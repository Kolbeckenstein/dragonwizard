"""
Tests for DragonWizardBot.is_allowed_channel() logic.

We test the pure channel-restriction method in isolation — no Discord
connection required.
"""

from unittest.mock import MagicMock

import pytest

from dragonwizard.bot.client import DragonWizardBot
from dragonwizard.config.settings import BotSettings, Settings


def _make_bot(allowed_channel_ids: list[int]) -> DragonWizardBot:
    """Create a DragonWizardBot with the given channel restriction list."""
    settings = MagicMock(spec=Settings)
    settings.bot = MagicMock(spec=BotSettings)
    settings.bot.command_prefix = "!"
    settings.bot.allowed_channel_ids = allowed_channel_ids
    # Patch discord internals so __init__ doesn't require a real connection
    bot = DragonWizardBot.__new__(DragonWizardBot)
    bot.settings = settings
    return bot


class TestBotChannelRestriction:
    def test_empty_list_allows_all_channels(self):
        """When allowed_channel_ids is empty the bot responds everywhere."""
        bot = _make_bot([])
        assert bot.is_allowed_channel(111) is True
        assert bot.is_allowed_channel(999999) is True

    def test_listed_channel_is_allowed(self):
        """A channel ID in the list returns True."""
        bot = _make_bot([111, 222, 333])
        assert bot.is_allowed_channel(111) is True
        assert bot.is_allowed_channel(333) is True

    def test_unlisted_channel_is_blocked(self):
        """A channel ID not in the list returns False when the list is non-empty."""
        bot = _make_bot([111, 222])
        assert bot.is_allowed_channel(999) is False
