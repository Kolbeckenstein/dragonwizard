"""
RulesCog — D&D rules Q&A via /ask and @mention.

Two entry points, one answer pipeline:
  - /ask question:<str>  (slash command)
  - @DragonWizard <question>  (mention in any message)

Both call _answer() which runs RAG search → LLM generation and returns
a discord.Embed. The embed contains the answer, source citations, and
any dice rolls the LLM made while generating the response.
"""

from __future__ import annotations

import re

import discord
from discord import app_commands
from discord.ext import commands

from dragonwizard.config.logging import get_logger
from dragonwizard.llm import LLMError

logger = get_logger(__name__)

# Matches <@USER_ID> and <@!USER_ID> (standard Discord mention formats)
_MENTION_RE = re.compile(r"<@!?\d+>")


def _error_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(
        title=title,
        description=description,
        color=discord.Color.red(),
    )


class RulesCog(commands.Cog):
    """Handles D&D rules questions via /ask and @mention."""

    def __init__(self, bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # Slash command
    # ------------------------------------------------------------------

    @app_commands.command(name="ask", description="Ask a D&D 5e rules question")
    @app_commands.describe(question="Your D&D rules question")
    async def ask(self, interaction: discord.Interaction, question: str) -> None:
        """
        /ask question:<your question>

        Searches the D&D rules knowledge base and generates an answer with
        source citations. The LLM may roll dice mid-answer if relevant.
        """
        if not self.bot.is_allowed_channel(interaction.channel_id):
            await interaction.response.send_message(
                "I'm not configured to respond in this channel.", ephemeral=True
            )
            return

        # Defer immediately — RAG + LLM will take well over Discord's 3s limit
        await interaction.response.defer()

        embed = await self._answer(question)
        await interaction.followup.send(embed=embed)

    # ------------------------------------------------------------------
    # Mention listener
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """
        Respond to @DragonWizard <question> mentions.

        Ignores:
        - Messages from bots (including ourselves)
        - Messages that don't mention this bot
        - Messages in non-allowed channels (if restriction is configured)
        """
        if message.author.bot:
            return
        if not self.bot.user.mentioned_in(message):
            return
        if not self.bot.is_allowed_channel(message.channel.id):
            return

        question = _strip_mention(message.clean_content, self.bot.user)
        if not question:
            await message.reply("What's your question? (e.g. `@DragonWizard How does advantage work?`)")
            return

        async with message.channel.typing():
            embed = await self._answer(question)

        await message.reply(embed=embed)

    # ------------------------------------------------------------------
    # Shared answer pipeline
    # ------------------------------------------------------------------

    async def _answer(self, question: str) -> discord.Embed:
        """
        Run RAG search → LLM generation and return a formatted Embed.

        Returns an error embed on failure rather than raising, so the
        bot never crashes on a bad query.
        """
        try:
            results = await self.bot.rag_engine.search(question)
            context = self.bot.rag_engine.format_context(results) if results else None
            response = await self.bot.orchestrator.generate_response(question, context)
        except LLMError as e:
            logger.warning(f"LLM error for question {question!r}: {e}")
            return _error_embed("LLM Error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error answering {question!r}: {e}")
            return _error_embed("Error", "Something went wrong. Please try again.")

        embed = discord.Embed(
            title=question[:256],
            description=response.text[:4096],
            color=discord.Color.purple(),
        )

        if results:
            sources = "\n".join(f"• {r.citation}" for r in results[:5])
            embed.add_field(name="Sources", value=sources[:1024], inline=False)

        if response.tool_calls:
            rolls = "\n".join(
                f"• `{tc.name}({tc.arguments})` → {tc.result}"
                for tc in response.tool_calls
            )
            embed.add_field(name="Dice Rolls", value=rolls[:1024], inline=False)

        embed.set_footer(text=f"{response.usage.total_tokens} tokens | {response.model}")
        return embed


def _strip_mention(text: str, bot_user: discord.ClientUser) -> str:
    """
    Remove all @mentions from text and return the trimmed remainder.

    Handles both the clean-text form (@DisplayName) and the raw Discord
    form (<@USER_ID> / <@!USER_ID>) since `message.clean_content` uses
    the display name form but we strip both to be safe.
    """
    # Remove raw mention tags (<@123> and <@!123>)
    text = _MENTION_RE.sub("", text)
    # Remove display-name form (@BotName)
    text = text.replace(f"@{bot_user.display_name}", "")
    return text.strip()
