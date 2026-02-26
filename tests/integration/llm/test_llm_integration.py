"""
Integration tests for the LLM Orchestration Layer.

These tests verify that the orchestrator correctly wires together with
adjacent layers (RAG engine, tool adapters). The LLM API itself is always
mocked — we never burn real API tokens in tests.

The value of these tests vs. unit tests:
- Unit tests mock everything and verify orchestrator logic in isolation.
- These integration tests use REAL components (real RAG engine with real
  embeddings, real MCP dice server) and only mock the LLM API call.
  This catches wiring bugs: wrong field names, incompatible schemas,
  format mismatches between layers.

Test groups:
1. RAG → Orchestrator: Real RAG context flows into the system prompt correctly.
2. Tool Adapter → Orchestrator: Real MCP tool schemas are formatted for LiteLLM.
3. Full stack (RAG + Tools + Orchestrator): All real components except the LLM.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dragonwizard.config.settings import LLMSettings, RAGSettings
from dragonwizard.llm.models import LLMResponse
from dragonwizard.llm.orchestrator import LLMOrchestrator
from dragonwizard.rag import RAGComponents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = (
    "You are DragonWizard, a D&D 5e rules expert.\n"
    "\n"
    "{context_block}"
    "\n"
    "Answer based on the provided context. Cite sources using [1], [2], etc."
)


def _make_text_response(text: str) -> MagicMock:
    """Build a mock LiteLLM text-only response."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None

    response = MagicMock()
    response.choices = [choice]
    response.model = "anthropic/claude-sonnet-4-5-20250929"
    response.usage.prompt_tokens = 200
    response.usage.completion_tokens = 100
    return response


def _make_tool_call_response(tool_name: str, arguments: dict, tool_call_id: str = "call_int_1") -> MagicMock:
    """Build a mock LiteLLM response requesting a tool call."""
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(arguments)
    tool_call.type = "function"

    choice = MagicMock()
    choice.message.content = None
    choice.message.tool_calls = [tool_call]
    choice.message.role = "assistant"

    response = MagicMock()
    response.choices = [choice]
    response.model = "anthropic/claude-sonnet-4-5-20250929"
    response.usage.prompt_tokens = 200
    response.usage.completion_tokens = 50
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_settings():
    return LLMSettings(
        model="anthropic/claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.3,
        api_key="test-integration-key",
    )


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent.parent / "fixtures" / "sample_documents"


@pytest.fixture
def sample_text_file(test_data_dir):
    return test_data_dir / "sample_rules.txt"


@pytest.fixture
def rag_settings(tmp_path):
    return RAGSettings(
        vector_db_path=str(tmp_path / "vector_db"),
        processed_data_path=str(tmp_path / "processed"),
    )


@pytest.fixture
def rag_factory(rag_settings):
    return RAGComponents(rag_settings)


@pytest.fixture
def dice_server_path():
    """Get path to real MCP dice server, skip if not available."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    server_path = os.path.join(repo_root, "external", "dice-rolling-mcp", "dist", "index.js")

    if not os.path.exists(server_path):
        pytest.skip("Dice server not found. Run 'make install-dice-server' first.")

    return server_path


# ---------------------------------------------------------------------------
# Contract Test 1: RAG context flows into the orchestrator's system prompt
# ---------------------------------------------------------------------------

class TestRAGToOrchestrator:
    """
    Verify that real RAG engine output (format_context) is correctly
    injected into the LLM system prompt.

    Why this matters: The unit tests mock the context as a plain string.
    These tests use REAL embeddings and vector search to produce context,
    ensuring the format_context() output is compatible with the orchestrator's
    {context_block} injection — no field name mismatches, no encoding issues.
    """

    @pytest.mark.asyncio
    async def test_rag_context_appears_in_system_prompt(
        self, sample_text_file, rag_factory, llm_settings
    ):
        """Real RAG context should appear in the system prompt sent to the LLM."""
        async with rag_factory.create_embedding_model() as embedding_model, \
                   rag_factory.create_vector_store() as vector_store:

            # Ingest real documents
            pipeline = rag_factory.create_pipeline(embedding_model, vector_store)
            await pipeline.ingest_file(sample_text_file)

            # Search and format context — this is what the orchestrator receives
            engine = rag_factory.create_engine(embedding_model, vector_store)
            results = await engine.search("fireball spell damage", k=3)
            context = engine.format_context(results)

            # Wire into orchestrator with mocked LLM
            orchestrator = LLMOrchestrator(
                settings=llm_settings,
                system_template=SYSTEM_TEMPLATE,
            )

            mock_resp = _make_text_response("Fireball deals 8d6 fire damage.")

            with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
                await orchestrator.generate_response(
                    query="How does fireball work?",
                    context=context,
                )

            # Verify: the actual RAG context (with real chunk text and citations)
            # made it into the system prompt
            messages = mock_call.call_args.kwargs["messages"]
            system_msg = next(m for m in messages if m["role"] == "system")

            # Should contain the reference material header
            assert "## Reference Material" in system_msg["content"]
            # Should contain actual text from the ingested document
            assert "[1]" in system_msg["content"]
            # Should contain source attribution from format_context()
            assert "score:" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_no_rag_results_produces_clean_prompt(
        self, rag_factory, llm_settings
    ):
        """
        When RAG returns no results, format_context() returns a fallback string.
        The orchestrator should handle this gracefully — no crashes, no empty
        reference section confusing the LLM.
        """
        async with rag_factory.create_embedding_model() as embedding_model, \
                   rag_factory.create_vector_store() as vector_store:

            # Search an empty store — no documents ingested
            engine = rag_factory.create_engine(embedding_model, vector_store)
            results = await engine.search("something totally irrelevant", k=3)
            context = engine.format_context(results)

            orchestrator = LLMOrchestrator(
                settings=llm_settings,
                system_template=SYSTEM_TEMPLATE,
            )

            mock_resp = _make_text_response("I don't have information on that.")

            with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
                result = await orchestrator.generate_response(
                    query="What is the meaning of life?",
                    context=context,
                )

            assert isinstance(result, LLMResponse)
            messages = mock_call.call_args.kwargs["messages"]
            system_msg = next(m for m in messages if m["role"] == "system")
            # "No relevant results found." is truthy, so it will appear in context_block
            # This is fine — the LLM sees it and knows there's no context
            assert "DragonWizard" in system_msg["content"]


# ---------------------------------------------------------------------------
# Contract Test 2: Real MCP tool schemas flow into the orchestrator
# ---------------------------------------------------------------------------

class TestToolAdapterToOrchestrator:
    """
    Verify that real MCP tool adapter schemas are correctly formatted
    for LiteLLM's tool use API.

    Why this matters: The unit tests mock list_tools() with hand-crafted dicts.
    The real DiceRollerTool returns schemas from the MCP server, which have a
    slightly different shape (e.g., 'input_schema' vs 'parameters'). The
    orchestrator's _get_tool_definitions() must translate correctly.
    """

    @pytest.mark.asyncio
    async def test_real_tool_schemas_formatted_for_litellm(
        self, dice_server_path, llm_settings
    ):
        """Real MCP tool schemas should be wrapped in LiteLLM's OpenAI-compatible format."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as dice_tool:
            orchestrator = LLMOrchestrator(
                settings=llm_settings,
                system_template=SYSTEM_TEMPLATE,
                tool_adapter=dice_tool,
            )

            mock_resp = _make_text_response("No dice needed for this answer.")

            with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
                await orchestrator.generate_response(
                    query="What is armor class?",
                    context=None,
                )

            # Verify: tools were included in the API call
            call_kwargs = mock_call.call_args.kwargs
            assert "tools" in call_kwargs

            tools = call_kwargs["tools"]
            assert len(tools) >= 1

            # Verify: LiteLLM-compatible format (OpenAI tool schema)
            dice_tool_def = tools[0]
            assert dice_tool_def["type"] == "function"
            assert "function" in dice_tool_def
            assert "name" in dice_tool_def["function"]
            assert "description" in dice_tool_def["function"]
            assert "parameters" in dice_tool_def["function"]

    @pytest.mark.asyncio
    async def test_real_tool_execution_in_loop(
        self, dice_server_path, llm_settings
    ):
        """
        Orchestrator should execute a real MCP tool call and feed the result
        back to the (mocked) LLM.

        This verifies the full tool-use loop with a real tool subprocess:
        1. Mock LLM requests a dice roll
        2. Real DiceRollerTool executes it via MCP
        3. Real result is appended to messages
        4. Mock LLM produces final text response
        """
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as dice_tool:
            orchestrator = LLMOrchestrator(
                settings=llm_settings,
                system_template=SYSTEM_TEMPLATE,
                tool_adapter=dice_tool,
            )

            # The real MCP server's tool is named "dice_roll", not "roll_dice"
            # This is exactly the kind of mismatch integration tests catch!
            tool_response = _make_tool_call_response("dice_roll", {"notation": "2d6+3"})
            text_response = _make_text_response("You rolled 2d6+3 and got a total of 12.")

            with patch(
                "dragonwizard.llm.orchestrator.acompletion",
                side_effect=[tool_response, text_response],
            ) as mock_call:
                result = await orchestrator.generate_response(
                    query="Roll 2d6+3 for damage",
                    context=None,
                )

            # Verify: got a response
            assert isinstance(result, LLMResponse)
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "dice_roll"

            # Verify: the REAL dice result was sent back to the LLM
            second_call_messages = mock_call.call_args_list[1].kwargs["messages"]
            tool_result_msg = next(m for m in second_call_messages if m["role"] == "tool")
            # Real MCP server returns actual dice notation in the result
            assert "2d6" in tool_result_msg["content"] or "d6" in tool_result_msg["content"]


# ---------------------------------------------------------------------------
# Contract Test 3: Full stack (RAG + Tools + Orchestrator)
# ---------------------------------------------------------------------------

class TestFullStack:
    """
    End-to-end test with all real components except the LLM API.

    This is the closest we can get to a real user interaction without
    burning API tokens. It verifies that RAG context AND tool definitions
    flow correctly into a single orchestrator call.

    Why this matters: Testing components in pairs (RAG+Orchestrator,
    Tool+Orchestrator) doesn't catch issues that only emerge when all
    three are combined — e.g., message ordering, token budget pressure,
    or tools param conflicting with system prompt format.
    """

    @pytest.mark.asyncio
    async def test_rag_context_and_tools_in_single_call(
        self, sample_text_file, rag_factory, dice_server_path, llm_settings
    ):
        """
        Full pipeline: real RAG context + real tool adapter + mocked LLM.
        Verifies all components wire together without conflicts.
        """
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with rag_factory.create_embedding_model() as embedding_model, \
                   rag_factory.create_vector_store() as vector_store, \
                   DiceRollerTool(server_path=dice_server_path) as dice_tool:

            # Ingest and search
            pipeline = rag_factory.create_pipeline(embedding_model, vector_store)
            await pipeline.ingest_file(sample_text_file)

            engine = rag_factory.create_engine(embedding_model, vector_store)
            results = await engine.search("fire damage spell", k=3)
            context = engine.format_context(results)

            # Build orchestrator with both RAG context and tools
            orchestrator = LLMOrchestrator(
                settings=llm_settings,
                system_template=SYSTEM_TEMPLATE,
                tool_adapter=dice_tool,
            )

            mock_resp = _make_text_response(
                "Fireball deals 8d6 fire damage [1]. "
                "The average damage is 28."
            )

            with patch("dragonwizard.llm.orchestrator.acompletion", return_value=mock_resp) as mock_call:
                result = await orchestrator.generate_response(
                    query="How much damage does fireball do?",
                    context=context,
                )

            # Verify: response came through
            assert isinstance(result, LLMResponse)
            assert "fire damage" in result.text.lower()

            # Verify: system prompt has RAG context
            call_kwargs = mock_call.call_args.kwargs
            messages = call_kwargs["messages"]
            system_msg = next(m for m in messages if m["role"] == "system")
            assert "## Reference Material" in system_msg["content"]
            assert "[1]" in system_msg["content"]

            # Verify: tools are present alongside context
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) >= 1
