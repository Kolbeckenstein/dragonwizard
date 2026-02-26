"""
DragonWizardBot — discord.py bot client.

Manages the full bot lifecycle:
- Initializes shared services (RAG engine, LLM orchestrator, dice tool) once at startup
- Loads command cogs (RulesCog, DiceCog)
- Syncs slash commands (guild-local for dev, global for production)
- Cleans up all resources on shutdown via AsyncExitStack
"""

from __future__ import annotations

import discord
from contextlib import AsyncExitStack
from pathlib import Path

from discord.ext import commands

from dragonwizard.config.logging import get_logger
from dragonwizard.config.settings import Settings
from dragonwizard.llm import LLMOrchestrator
from dragonwizard.rag.components import RAGComponents
from dragonwizard.rag.engine import RAGEngine
from dragonwizard.tools.dice_roller import DiceRollerTool

logger = get_logger(__name__)


class DragonWizardBot(commands.Bot):
    """
    Discord bot for D&D 5e rules assistance.

    Holds shared application state (RAG engine, orchestrator) and exposes
    it to cogs. All async resources are managed via AsyncExitStack so they're
    properly cleaned up when the bot shuts down.

    Args:
        settings: Full application settings (bot token, LLM config, RAG config, etc.)
    """

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message text for mention handling
        super().__init__(
            command_prefix=settings.bot.command_prefix,
            intents=intents,
        )
        self.settings = settings
        self.rag_engine: RAGEngine | None = None
        self.orchestrator: LLMOrchestrator | None = None
        self._exit_stack = AsyncExitStack()

    async def setup_hook(self) -> None:
        """
        Called after login, before connecting to the Gateway.

        Initializes all services, loads cogs, and syncs slash commands.
        Runs in the bot's event loop so async operations are safe here.
        """
        # --- 1. RAG components (long-lived; kept alive for the bot's lifetime) ---
        logger.info("Initializing RAG components...")
        factory = RAGComponents(self.settings.rag)
        embedding_model = await self._exit_stack.enter_async_context(
            factory.create_embedding_model()
        )
        vector_store = await self._exit_stack.enter_async_context(
            factory.create_vector_store()
        )
        self.rag_engine = factory.create_engine(embedding_model, vector_store)
        logger.info("RAG engine ready")

        # --- 2. Dice tool (optional; enables LLM tool use and /roll command) ---
        tool_adapter = None
        if self.settings.tools.dice_server_path:
            logger.info(f"Initializing dice tool: {self.settings.tools.dice_server_path}")
            dice_tool = DiceRollerTool(self.settings.tools.dice_server_path)
            tool_adapter = await self._exit_stack.enter_async_context(dice_tool)
            logger.info("Dice tool ready")
        else:
            logger.info("Dice tool not configured (TOOL__DICE_SERVER_PATH not set) — /roll disabled")

        # --- 3. LLM orchestrator ---
        system_template_path = Path(__file__).parent.parent / "prompts" / "system.txt"
        system_template = system_template_path.read_text()
        self.orchestrator = LLMOrchestrator(
            settings=self.settings.llm,
            system_template=system_template,
            tool_adapter=tool_adapter,
        )
        logger.info(f"LLM orchestrator ready (model: {self.settings.llm.model})")

        # --- 4. Load cogs ---
        from dragonwizard.bot.cogs.rules import RulesCog
        from dragonwizard.bot.cogs.dice import DiceCog
        await self.add_cog(RulesCog(self))
        await self.add_cog(DiceCog(self))
        logger.info("Cogs loaded")

        # --- 5. Sync slash commands ---
        try:
            if self.settings.bot.dev_guild_id:
                guild = discord.Object(id=self.settings.bot.dev_guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info(f"Slash commands synced to dev guild {self.settings.bot.dev_guild_id} (instant)")
            else:
                await self.tree.sync()
                logger.info("Slash commands synced globally (may take up to 1 hour to propagate)")
        except discord.errors.Forbidden:
            logger.warning(
                "Could not sync slash commands (403 Forbidden). "
                "The bot is missing the 'applications.commands' OAuth2 scope. "
                "Re-invite the bot using an OAuth2 URL that includes both 'bot' "
                "and 'applications.commands' scopes. "
                "The bot will still respond to @mentions while slash commands are unavailable."
            )
        except Exception as e:
            logger.warning(f"Slash command sync failed: {e}. The bot will still start.")

    async def on_ready(self) -> None:
        """Called when the bot successfully connects to Discord."""
        logger.info(f"Logged in as {self.user} (id: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

    async def close(self) -> None:
        """Graceful shutdown — clean up all async resources before disconnecting."""
        logger.info("Shutting down DragonWizard...")
        await self._exit_stack.aclose()
        await super().close()

    def is_allowed_channel(self, channel_id: int) -> bool:
        """
        Return True if the bot should respond in this channel.

        If `allowed_channel_ids` is empty (the default), the bot responds everywhere.
        If it's non-empty, the bot only responds in the listed channel IDs.
        """
        allowed = self.settings.bot.allowed_channel_ids
        return not allowed or channel_id in allowed
