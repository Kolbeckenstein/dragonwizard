"""
DiceCog — standalone /roll slash command.

Lets users roll dice directly without going through the LLM. Routes to the
same MCP dice tool adapter used by the orchestrator's tool-use loop.

If the dice tool is not configured (TOOL__DICE_SERVER_PATH not set), the
command sends an ephemeral error message rather than crashing.
"""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from dragonwizard.config.logging import get_logger

logger = get_logger(__name__)


class DiceCog(commands.Cog):
    """Provides the /roll slash command for direct dice rolling."""

    def __init__(self, bot) -> None:
        self.bot = bot

    @app_commands.command(name="roll", description="Roll dice (e.g. 2d6+3, d20, 4d6kh3)")
    @app_commands.describe(expression="Dice expression to evaluate")
    async def roll(self, interaction: discord.Interaction, expression: str) -> None:
        """
        /roll expression:<dice expression>

        Evaluates a dice expression via the MCP dice server. Requires
        TOOL__DICE_SERVER_PATH to be configured.

        Examples:
          /roll expression:d20
          /roll expression:2d6+3
          /roll expression:4d6kh3   (roll 4d6, keep highest 3)
        """
        if not self.bot.is_allowed_channel(interaction.channel_id):
            await interaction.response.send_message(
                "I'm not configured to respond in this channel.", ephemeral=True
            )
            return

        tool_adapter = self.bot.orchestrator._tool_adapter if self.bot.orchestrator else None
        if tool_adapter is None:
            await interaction.response.send_message(
                "Dice roller not configured. Ask the bot admin to set `TOOL__DICE_SERVER_PATH`.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            result = await tool_adapter.call("dice_roll", {"notation": expression})
            text = result.get("text", str(result))
        except Exception as e:
            logger.warning(f"Dice roll failed for {expression!r}: {e}")
            await interaction.followup.send(f"Roll failed: {e}")
            return

        embed = discord.Embed(
            title=f"🎲 {expression}",
            description=text,
            color=discord.Color.green(),
        )
        embed.set_footer(text=f"Requested by {interaction.user.display_name}")
        await interaction.followup.send(embed=embed)
