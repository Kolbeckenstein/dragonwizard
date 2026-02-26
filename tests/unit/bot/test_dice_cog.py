"""
Tests for DiceCog.

Covers:
- No tool adapter → ephemeral config message
- Successful roll → green embed with expression as title
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import discord

from dragonwizard.bot.cogs.dice import DiceCog


def _make_interaction(channel_id=100):
    interaction = MagicMock(spec=discord.Interaction)
    interaction.channel_id = channel_id
    interaction.user = MagicMock()
    interaction.user.display_name = "Tester"
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _make_bot(tool_adapter=None, allowed=True):
    bot = MagicMock()
    bot.is_allowed_channel.return_value = allowed
    bot.orchestrator = MagicMock()
    bot.orchestrator._tool_adapter = tool_adapter
    return bot


class TestDiceCog:
    @pytest.mark.asyncio
    async def test_no_adapter_sends_ephemeral_config_message(self):
        """When no dice tool is configured, an ephemeral error message is sent."""
        bot = _make_bot(tool_adapter=None)
        cog = DiceCog(bot)
        interaction = _make_interaction()

        await cog.roll.callback(cog, interaction, "2d6")

        interaction.response.defer.assert_not_called()
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        assert call_kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_successful_roll_sends_green_embed(self):
        """A successful dice roll response is sent as a green embed."""
        adapter = MagicMock()
        adapter.call = AsyncMock(return_value={"text": "[12, 5] → 17"})
        bot = _make_bot(tool_adapter=adapter)
        cog = DiceCog(bot)
        interaction = _make_interaction()

        await cog.roll.callback(cog, interaction, "2d10+5")

        adapter.call.assert_awaited_once_with("dice_roll", {"notation": "2d10+5"})
        interaction.response.defer.assert_called_once()
        interaction.followup.send.assert_called_once()
        embed = interaction.followup.send.call_args[1]["embed"]
        assert embed.title == "🎲 2d10+5"
        assert embed.color == discord.Color.green()

    @pytest.mark.asyncio
    async def test_blocked_channel_sends_ephemeral(self):
        """Channel restriction sends ephemeral reply without rolling."""
        bot = _make_bot(allowed=False)
        cog = DiceCog(bot)
        interaction = _make_interaction()

        await cog.roll.callback(cog, interaction, "d20")

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        assert call_kwargs.get("ephemeral") is True
