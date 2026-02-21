"""
Unit tests for the LLM Orchestration Layer.

Tests cover:
- Initialization and configuration
- System prompt construction
- Basic response generation (no tools)
- Tool use loop
- Tool round limiting
- Error handling
- Response model structure
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dragonwizard.config.settings import LLMSettings
from dragonwizard.llm.orchestrator import LLMOrchestrator
from dragonwizard.llm.models import LLMResponse, ToolCall, TokenUsage


# ---------------------------------------------------------------------------
# Helpers for building mock LiteLLM responses
# ---------------------------------------------------------------------------

def _make_text_response(text: str, model: str = "anthropic/claude-sonnet-4-5-20250929") -> MagicMock:
    """Build a mock LiteLLM response that contains only text (no tool calls)."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


def _make_tool_call_response(
    tool_name: str,
    arguments: dict,
    tool_call_id: str = "call_123",
    model: str = "anthropic/claude-sonnet-4-5-20250929",
) -> MagicMock:
    """Build a mock LiteLLM response that requests a tool call."""
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(arguments)
    tool_call.type = "function"

    choice = MagicMock()
    choice.message.content = None
    choice.message.tool_calls = [tool_call]
    # LiteLLM expects the raw message to be appended back to the conversation
    choice.message.role = "assistant"

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 30
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Default LLM settings for tests."""
    return LLMSettings(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.3,
        api_key="test-api-key",
    )


@pytest.fixture
def system_template():
    """A minimal system prompt template for tests."""
    return (
        "You are DragonWizard, a D&D 5e rules expert.\n"
        "\n"
        "{context_block}"
        "\n"
        "Answer based on the provided context. Cite sources using [1], [2], etc."
    )


@pytest.fixture
def mock_tool_adapter():
    """Mock ToolAdapter that exposes a dice roller tool."""
    adapter = AsyncMock()
    adapter.list_tools.return_value = [
        {
            "name": "roll_dice",
            "description": "Roll dice using standard notation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "notation": {"type": "string", "description": "Dice notation e.g. 2d6+3"},
                },
                "required": ["notation"],
            },
        }
    ]
    adapter.call.return_value = {"text": "Rolled 2d6+3: [4, 5] + 3 = 12"}
    return adapter


@pytest.fixture
def orchestrator(settings, system_template):
    """Orchestrator with no tools."""
    return LLMOrchestrator(
        settings=settings,
        system_template=system_template,
    )


@pytest.fixture
def orchestrator_with_tools(settings, system_template, mock_tool_adapter):
    """Orchestrator with a mock tool adapter."""
    return LLMOrchestrator(
        settings=settings,
        system_template=system_template,
        tool_adapter=mock_tool_adapter,
    )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestOrchestratorInitialization:
    """Tests for LLMOrchestrator construction."""

    def test_initialize_with_settings(self, settings, system_template):
        orch = LLMOrchestrator(settings=settings, system_template=system_template)
        assert orch._settings is settings
        assert orch._system_template == system_template

    def test_initialize_without_tool_adapter(self, settings, system_template):
        orch = LLMOrchestrator(settings=settings, system_template=system_template)
        assert orch._tool_adapter is None

    def test_initialize_with_tool_adapter(self, settings, system_template, mock_tool_adapter):
        orch = LLMOrchestrator(
            settings=settings,
            system_template=system_template,
            tool_adapter=mock_tool_adapter,
        )
        assert orch._tool_adapter is mock_tool_adapter

    def test_default_max_tool_rounds(self, orchestrator):
        assert orchestrator._max_tool_rounds == 5

    def test_custom_max_tool_rounds(self, settings, system_template):
        orch = LLMOrchestrator(
            settings=settings,
            system_template=system_template,
            max_tool_rounds=10,
        )
        assert orch._max_tool_rounds == 10


class TestSystemPromptConstruction:
    """Tests for _build_system_prompt method."""

    def test_injects_context_into_template(self, orchestrator):
        context = "[1] Fireball deals 8d6 fire damage.\n    Source: PHB p.241"
        prompt = orchestrator._build_system_prompt(context)
        assert "## Reference Material" in prompt
        assert "Fireball deals 8d6 fire damage" in prompt
        assert "DragonWizard" in prompt

    def test_no_context_omits_reference_section(self, orchestrator):
        prompt = orchestrator._build_system_prompt(None)
        assert "## Reference Material" not in prompt
        assert "None" not in prompt
        assert "DragonWizard" in prompt

    def test_empty_string_context_omits_reference_section(self, orchestrator):
        prompt = orchestrator._build_system_prompt("")
        assert "## Reference Material" not in prompt

    def test_preserves_template_instructions(self, orchestrator):
        prompt = orchestrator._build_system_prompt("some context")
        assert "Cite sources using [1], [2]" in prompt


class TestBasicResponseGeneration:
    """Tests for generate_response without tool use."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, orchestrator):
        mock_resp = _make_text_response("Fireball deals 8d6 fire damage in a 20-foot radius.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp):
            result = await orchestrator.generate_response(
                query="How does fireball work?",
                context="[1] Fireball: 8d6 fire damage, 20-foot radius.",
            )

        assert isinstance(result, LLMResponse)
        assert "8d6 fire damage" in result.text

    @pytest.mark.asyncio
    async def test_passes_model_from_settings(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator.generate_response(query="test", context=None)

        call_kwargs = mock_call.call_args
        assert "claude-sonnet-4-5-20250929" in call_kwargs.kwargs["model"]

    @pytest.mark.asyncio
    async def test_passes_temperature_and_max_tokens(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator.generate_response(query="test", context=None)

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_sends_system_prompt_with_context(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator.generate_response(
                query="How does fireball work?",
                context="[1] Fireball: 8d6 fire damage.",
            )

        call_kwargs = mock_call.call_args.kwargs
        # System prompt should contain the context
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Fireball" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_sends_user_query_as_user_message(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator.generate_response(
                query="How does fireball work?",
                context=None,
            )

        messages = mock_call.call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "How does fireball work?"

    @pytest.mark.asyncio
    async def test_no_tools_omits_tools_param(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator.generate_response(query="test", context=None)

        call_kwargs = mock_call.call_args.kwargs
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_response_includes_token_usage(self, orchestrator):
        mock_resp = _make_text_response("Answer.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp):
            result = await orchestrator.generate_response(query="test", context=None)

        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_response_includes_model_name(self, orchestrator):
        mock_resp = _make_text_response("Answer.", model="anthropic/claude-sonnet-4-5-20250929")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp):
            result = await orchestrator.generate_response(query="test", context=None)

        assert result.model == "anthropic/claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_no_context_still_works(self, orchestrator):
        mock_resp = _make_text_response("I don't have specific rules context.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp):
            result = await orchestrator.generate_response(query="What is D&D?", context=None)

        assert isinstance(result, LLMResponse)
        assert len(result.text) > 0


class TestToolUseLoop:
    """Tests for the tool call loop in generate_response."""

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_final_text(self, orchestrator_with_tools, mock_tool_adapter):
        tool_response = _make_tool_call_response("roll_dice", {"notation": "2d6+3"})
        text_response = _make_text_response("You rolled 12 on 2d6+3.")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_response, text_response],
        ):
            result = await orchestrator_with_tools.generate_response(
                query="Roll 2d6+3 for damage",
                context=None,
            )

        assert "rolled 12" in result.text
        mock_tool_adapter.call.assert_called_once_with("roll_dice", {"notation": "2d6+3"})

    @pytest.mark.asyncio
    async def test_records_tool_calls_in_response(self, orchestrator_with_tools, mock_tool_adapter):
        tool_response = _make_tool_call_response("roll_dice", {"notation": "1d20"})
        text_response = _make_text_response("You rolled a 15.")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_response, text_response],
        ):
            result = await orchestrator_with_tools.generate_response(
                query="Roll a d20",
                context=None,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "roll_dice"
        assert result.tool_calls[0].arguments == {"notation": "1d20"}

    @pytest.mark.asyncio
    async def test_includes_tool_definitions_in_api_call(self, orchestrator_with_tools, mock_tool_adapter):
        mock_resp = _make_text_response("No tools needed.")

        with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
            await orchestrator_with_tools.generate_response(query="test", context=None)

        call_kwargs = mock_call.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "roll_dice"

    @pytest.mark.asyncio
    async def test_multiple_tool_rounds(self, orchestrator_with_tools, mock_tool_adapter):
        """LLM calls tool twice before producing final text."""
        tool_resp_1 = _make_tool_call_response("roll_dice", {"notation": "1d20"}, tool_call_id="call_1")
        tool_resp_2 = _make_tool_call_response("roll_dice", {"notation": "2d6"}, tool_call_id="call_2")
        text_resp = _make_text_response("Attack hit for 9 damage.")

        mock_tool_adapter.call.side_effect = [
            {"text": "Rolled 1d20: 18"},
            {"text": "Rolled 2d6: [4, 5] = 9"},
        ]

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_resp_1, tool_resp_2, text_resp],
        ):
            result = await orchestrator_with_tools.generate_response(
                query="Roll to attack and damage",
                context=None,
            )

        assert len(result.tool_calls) == 2
        assert mock_tool_adapter.call.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_result_sent_back_to_llm(self, orchestrator_with_tools, mock_tool_adapter):
        """After tool execution, the result is appended to messages for the next LLM call."""
        tool_response = _make_tool_call_response("roll_dice", {"notation": "1d20"}, tool_call_id="call_abc")
        text_response = _make_text_response("You rolled 18.")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_response, text_response],
        ) as mock_call:
            await orchestrator_with_tools.generate_response(
                query="Roll a d20",
                context=None,
            )

        # The second call should include the tool result in messages
        second_call_messages = mock_call.call_args_list[1].kwargs["messages"]
        tool_result_msg = next(
            (m for m in second_call_messages if m["role"] == "tool"), None
        )
        assert tool_result_msg is not None
        assert tool_result_msg["tool_call_id"] == "call_abc"
        assert "Rolled" in tool_result_msg["content"]


class TestToolRoundLimit:
    """Tests for the max tool rounds safety limit."""

    @pytest.mark.asyncio
    async def test_stops_after_max_rounds(self, settings, system_template, mock_tool_adapter):
        orch = LLMOrchestrator(
            settings=settings,
            system_template=system_template,
            tool_adapter=mock_tool_adapter,
            max_tool_rounds=2,
        )

        # LLM keeps requesting tools beyond the limit
        tool_resp_1 = _make_tool_call_response("roll_dice", {"notation": "1d20"}, tool_call_id="call_1")
        tool_resp_2 = _make_tool_call_response("roll_dice", {"notation": "1d20"}, tool_call_id="call_2")
        # After limit, orchestrator sends a "no more tools" message; LLM responds with text
        final_resp = _make_text_response("Based on what I have so far...")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_resp_1, tool_resp_2, final_resp],
        ):
            result = await orch.generate_response(query="Roll many dice", context=None)

        assert isinstance(result, LLMResponse)
        assert len(result.tool_calls) == 2
        assert "Based on what I have" in result.text

    @pytest.mark.asyncio
    async def test_limit_message_sent_to_llm(self, settings, system_template, mock_tool_adapter):
        """When limit is hit, a user-role message tells the LLM to stop requesting tools."""
        orch = LLMOrchestrator(
            settings=settings,
            system_template=system_template,
            tool_adapter=mock_tool_adapter,
            max_tool_rounds=1,
        )

        tool_resp = _make_tool_call_response("roll_dice", {"notation": "1d20"}, tool_call_id="call_1")
        final_resp = _make_text_response("OK, here's my answer.")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_resp, final_resp],
        ) as mock_call:
            await orch.generate_response(query="test", context=None)

        # The final call should NOT include tools parameter (prevent further tool calls)
        final_call_kwargs = mock_call.call_args_list[-1].kwargs
        assert "tools" not in final_call_kwargs


class TestToolErrorHandling:
    """Tests for handling tool execution failures."""

    @pytest.mark.asyncio
    async def test_tool_failure_sent_as_error_result(self, orchestrator_with_tools, mock_tool_adapter):
        """If a tool raises an exception, the error is sent back to the LLM as the tool result."""
        mock_tool_adapter.call.side_effect = RuntimeError("Dice server crashed")

        tool_response = _make_tool_call_response("roll_dice", {"notation": "1d20"})
        text_response = _make_text_response("I wasn't able to roll dice, but here's the rule...")

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=[tool_response, text_response],
        ) as mock_call:
            result = await orchestrator_with_tools.generate_response(
                query="Roll a d20",
                context=None,
            )

        # Should NOT raise — error is passed to LLM
        assert isinstance(result, LLMResponse)

        # The tool result message should contain the error
        second_call_messages = mock_call.call_args_list[1].kwargs["messages"]
        tool_result_msg = next(m for m in second_call_messages if m["role"] == "tool")
        assert "error" in tool_result_msg["content"].lower()


class TestErrorHandling:
    """Tests for orchestrator-level error handling."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self, orchestrator):
        with pytest.raises(ValueError, match="[Qq]uery"):
            await orchestrator.generate_response(query="", context=None)

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_value_error(self, orchestrator):
        with pytest.raises(ValueError, match="[Qq]uery"):
            await orchestrator.generate_response(query="   ", context=None)

    @pytest.mark.asyncio
    async def test_api_failure_raises_llm_error(self, orchestrator):
        from dragonwizard.llm.models import LLMError

        with patch(
            "dragonwizard.llm.orchestrator.acompletion",
            side_effect=Exception("API rate limit exceeded"),
        ):
            with pytest.raises(LLMError):
                await orchestrator.generate_response(query="test", context=None)

    @pytest.mark.asyncio
    async def test_no_api_key_raises_llm_error(self, system_template):
        from dragonwizard.llm.models import LLMError

        settings = LLMSettings(api_key="")
        orch = LLMOrchestrator(settings=settings, system_template=system_template)

        with pytest.raises(LLMError, match="[Aa]pi.key|API.key"):
            await orch.generate_response(query="test", context=None)


class TestResponseModel:
    """Tests for the LLMResponse Pydantic model."""

    def test_response_with_no_tool_calls(self):
        resp = LLMResponse(
            text="Fireball deals 8d6 fire damage.",
            tool_calls=[],
            model="anthropic/claude-sonnet-4-5-20250929",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        assert resp.text == "Fireball deals 8d6 fire damage."
        assert resp.tool_calls == []

    def test_response_with_tool_calls(self):
        resp = LLMResponse(
            text="You rolled 18.",
            tool_calls=[
                ToolCall(name="roll_dice", arguments={"notation": "1d20"}, result="18"),
            ],
            model="anthropic/claude-sonnet-4-5-20250929",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "roll_dice"

    def test_token_usage_total(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_tool_call_model(self):
        tc = ToolCall(
            name="roll_dice",
            arguments={"notation": "2d6+3"},
            result="Rolled 12",
        )
        assert tc.name == "roll_dice"
        assert tc.arguments == {"notation": "2d6+3"}
        assert tc.result == "Rolled 12"
