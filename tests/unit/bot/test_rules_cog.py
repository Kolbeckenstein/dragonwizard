"""
Tests for RulesCog.

Covers:
- _strip_mention helper (pure function, no Discord required)
- _answer embed structure (mocked RAG engine + orchestrator)
- on_message gate conditions (bot messages, non-mentions, blocked channels)
- /ask channel restriction
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from dragonwizard.bot.cogs.rules import RulesCog, _strip_mention
from dragonwizard.llm import LLMError, LLMResponse, TokenUsage, ToolCall
from dragonwizard.rag.base import SearchResult, ChunkMetadata
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_metadata(**kwargs) -> ChunkMetadata:
    defaults = dict(
        chunk_id="abc",
        document_id="doc1",
        source_file="phb.pdf",
        source_type="pdf",
        title="Player's Handbook",
        chunk_index=0,
        total_chunks=1,
        token_count=100,
        page_number=1,
        section=None,
        char_start=None,
        char_end=None,
        edition="5e",
        ingestion_timestamp=datetime(2024, 1, 1),
        pipeline_version="1.0",
    )
    defaults.update(kwargs)
    return ChunkMetadata(**defaults)


def _make_search_result(text="Some rules text") -> SearchResult:
    return SearchResult(
        text=text,
        score=0.9,
        metadata=_make_chunk_metadata(),
        citation="Player's Handbook p.1",
    )


def _make_llm_response(text="Here is the answer.", tool_calls=None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        model="anthropic/claude-3-5-sonnet",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
    )


def _make_bot(allowed_channel_ids=None, rag_results=None, llm_response=None):
    """Create a mock bot with configurable RAG/LLM behaviour."""
    bot = MagicMock()
    bot.is_allowed_channel.return_value = True
    if allowed_channel_ids is not None:
        bot.is_allowed_channel.side_effect = lambda ch_id: (
            not allowed_channel_ids or ch_id in allowed_channel_ids
        )
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.display_name = "DragonWizard"

    results = rag_results if rag_results is not None else [_make_search_result()]
    bot.rag_engine.search = AsyncMock(return_value=results)
    bot.rag_engine.format_context.return_value = "Some context"

    response = llm_response or _make_llm_response()
    bot.orchestrator.generate_response = AsyncMock(return_value=response)
    return bot


# ---------------------------------------------------------------------------
# _strip_mention
# ---------------------------------------------------------------------------

class TestStripMention:
    def _bot_user(self, user_id=12345, display_name="DragonWizard"):
        u = MagicMock(spec=discord.ClientUser)
        u.id = user_id
        u.display_name = display_name
        return u

    def test_strips_raw_mention_format(self):
        """<@123> is removed from the text."""
        user = self._bot_user(user_id=123)
        result = _strip_mention("<@123> How does fireball work?", user)
        assert result == "How does fireball work?"

    def test_strips_nickname_mention_format(self):
        """<@!123> is also removed."""
        user = self._bot_user(user_id=123)
        result = _strip_mention("<@!123> What is a saving throw?", user)
        assert result == "What is a saving throw?"

    def test_strips_display_name_form(self):
        """@DisplayName is removed from clean_content."""
        user = self._bot_user(display_name="DragonWizard")
        result = _strip_mention("@DragonWizard How does grappling work?", user)
        assert result == "How does grappling work?"

    def test_returns_empty_string_for_mention_only(self):
        """A message that is only a mention strips to empty string."""
        user = self._bot_user(user_id=123)
        result = _strip_mention("<@123>", user)
        assert result == ""

    def test_leaves_non_mention_text_intact(self):
        """Text without a mention is returned unchanged."""
        user = self._bot_user(user_id=999)
        result = _strip_mention("How does advantage work?", user)
        assert result == "How does advantage work?"


# ---------------------------------------------------------------------------
# _answer embed structure
# ---------------------------------------------------------------------------

class TestAnswerEmbed:
    @pytest.mark.asyncio
    async def test_embed_has_question_as_title_and_answer_as_description(self):
        bot = _make_bot()
        cog = RulesCog(bot)
        embed = await cog._answer("How does fireball work?")
        assert embed.title == "How does fireball work?"
        assert embed.description == "Here is the answer."

    @pytest.mark.asyncio
    async def test_embed_includes_sources_field_when_rag_returns_results(self):
        result = _make_search_result()
        bot = _make_bot(rag_results=[result])
        cog = RulesCog(bot)
        embed = await cog._answer("How does advantage work?")
        field_names = [f.name for f in embed.fields]
        assert "Sources" in field_names

    @pytest.mark.asyncio
    async def test_embed_omits_sources_field_when_no_rag_results(self):
        bot = _make_bot(rag_results=[])
        cog = RulesCog(bot)
        embed = await cog._answer("What is a spell?")
        field_names = [f.name for f in embed.fields]
        assert "Sources" not in field_names

    @pytest.mark.asyncio
    async def test_embed_includes_dice_rolls_field_when_tool_calls_present(self):
        tool_call = ToolCall(name="roll_dice", arguments={"expression": "1d20"}, result="15")
        response = _make_llm_response(tool_calls=[tool_call])
        bot = _make_bot(llm_response=response)
        cog = RulesCog(bot)
        embed = await cog._answer("Roll for initiative")
        field_names = [f.name for f in embed.fields]
        assert "Dice Rolls" in field_names

    @pytest.mark.asyncio
    async def test_embed_omits_dice_rolls_field_when_no_tool_calls(self):
        bot = _make_bot()
        cog = RulesCog(bot)
        embed = await cog._answer("What is AC?")
        field_names = [f.name for f in embed.fields]
        assert "Dice Rolls" not in field_names

    @pytest.mark.asyncio
    async def test_llm_error_returns_red_error_embed(self):
        bot = _make_bot()
        bot.orchestrator.generate_response = AsyncMock(
            side_effect=LLMError("API key not configured")
        )
        cog = RulesCog(bot)
        embed = await cog._answer("What is a paladin?")
        assert embed.color == discord.Color.red()
        assert "LLM Error" in embed.title

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_error_embed_not_raised(self):
        bot = _make_bot()
        bot.rag_engine.search = AsyncMock(side_effect=RuntimeError("DB unavailable"))
        cog = RulesCog(bot)
        embed = await cog._answer("What is a barbarian?")
        assert embed.color == discord.Color.red()


# ---------------------------------------------------------------------------
# on_message gate conditions
# ---------------------------------------------------------------------------

class TestOnMessageGating:
    def _make_message(self, is_bot=False, mentions_bot=True, channel_id=100):
        msg = MagicMock(spec=discord.Message)
        msg.author = MagicMock()
        msg.author.bot = is_bot
        msg.channel = MagicMock()
        msg.channel.id = channel_id
        msg.clean_content = "<@12345> How does grappling work?"
        msg.reply = AsyncMock()
        # channel.typing() is used as an async context manager
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=False)
        msg.channel.typing.return_value = typing_cm
        return msg

    @pytest.mark.asyncio
    async def test_ignores_bot_messages(self):
        bot = _make_bot()
        cog = RulesCog(bot)
        message = self._make_message(is_bot=True)
        await cog.on_message(message)
        bot.rag_engine.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_mention_messages(self):
        bot = _make_bot()
        bot.user.mentioned_in.return_value = False
        cog = RulesCog(bot)
        message = self._make_message()
        await cog.on_message(message)
        bot.rag_engine.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_messages_in_blocked_channels(self):
        bot = _make_bot(allowed_channel_ids=[999])  # 100 not in list
        bot.user.mentioned_in.return_value = True
        cog = RulesCog(bot)
        message = self._make_message(channel_id=100)
        await cog.on_message(message)
        bot.rag_engine.search.assert_not_called()
