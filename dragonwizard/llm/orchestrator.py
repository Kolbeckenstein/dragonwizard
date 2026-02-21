"""
LLM Orchestrator — core response generation engine.

This module sits between the RAG retrieval layer and the Discord bot layer.
It receives a user query + pre-formatted RAG context, constructs prompts,
manages the LLM API interaction (including tool-use loops), and returns
a structured LLMResponse.

Data flow:
    RAGEngine.format_context() → context string
                                      ↓
    User query + context → LLMOrchestrator.generate_response()
                                      ↓
                              LiteLLM acompletion()  ←→  ToolAdapter (optional loop)
                                      ↓
                                LLMResponse → Discord bot layer

Design decisions:
- Uses LiteLLM for provider abstraction — users can swap between Anthropic,
  OpenAI, local models (Ollama), etc. by changing a config string.
- RAG context is injected into the system prompt (not the user message)
  for authority framing: the model treats system content as more trustworthy,
  reducing hallucination. It also provides mild prompt-injection resistance.
- Tool errors are passed back to the LLM as error text (not raised) so the
  model can gracefully degrade — e.g., "I couldn't roll dice, but here's
  the rule about damage." Crashing on a tool failure would be a worse UX.
- A max_tool_rounds limit prevents runaway loops where the LLM keeps
  requesting tools indefinitely. When hit, we send a final request without
  tool definitions, forcing a text response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from litellm import acompletion

from dragonwizard.config.settings import LLMSettings
from dragonwizard.llm.models import LLMError, LLMResponse, TokenUsage, ToolCall
from dragonwizard.tools.base import ToolAdapter

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates LLM API calls with optional tool use.

    The orchestrator handles the full lifecycle of a single user query:
    1. Build a system prompt with RAG context injected
    2. Send to LLM via LiteLLM
    3. If the LLM requests a tool call, execute it and loop
    4. Return the final text response with metadata

    Each call to generate_response() is stateless — there is no conversation
    memory across calls. Every API request sends the full message history
    from scratch (system prompt + user query + any tool round-trips).
    This is by design for v1; multi-turn memory is a future enhancement.

    Args:
        settings: LLM configuration (model, temperature, max_tokens, api_key)
        system_template: Prompt template string with a {context_block} placeholder
        tool_adapter: Optional adapter for external tools (e.g., MCP dice roller)
        max_tool_rounds: Safety limit on tool-use loop iterations (default: 5)
    """

    def __init__(
        self,
        settings: LLMSettings,
        system_template: str,
        tool_adapter: ToolAdapter | None = None,
        max_tool_rounds: int = 5,
    ):
        self._settings = settings
        self._system_template = system_template
        self._tool_adapter = tool_adapter
        self._max_tool_rounds = max_tool_rounds

    def _build_system_prompt(self, context: str | None) -> str:
        """
        Build the final system prompt by injecting RAG context into the template.

        When context is provided, it's wrapped in a "## Reference Material" section
        so the LLM knows to treat it as authoritative source material. When there's
        no context (e.g., general question with no RAG results), the section is
        omitted entirely — no empty headers or "None" strings.

        Args:
            context: Pre-formatted RAG context from RAGEngine.format_context(),
                     or None if no relevant documents were found.
        """
        context_block = ""
        if context:
            context_block = f"## Reference Material\n{context}\n"
        return self._system_template.replace("{context_block}", context_block)

    async def _get_tool_definitions(self) -> list[dict[str, Any]] | None:
        """
        Fetch tool schemas from the adapter in LiteLLM's expected format.

        LiteLLM uses the OpenAI tool format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Our ToolAdapter.list_tools() returns a slightly flatter format
        (name, description, input_schema), so we wrap it here.

        Returns None if no tool adapter is configured.
        """
        if self._tool_adapter is None:
            return None

        raw_tools = await self._tool_adapter.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in raw_tools
        ]

    async def generate_response(
        self,
        query: str,
        context: str | None,
    ) -> LLMResponse:
        """
        Generate a response to a user query, optionally using RAG context and tools.

        This is the main entry point. It validates input, constructs the prompt,
        calls the LLM, handles any tool-use rounds, and returns a structured response.

        Args:
            query: The user's question (must be non-empty after stripping whitespace)
            context: Pre-formatted RAG context, or None for context-free queries

        Returns:
            LLMResponse with the answer text, tool call records, model info, and usage

        Raises:
            ValueError: If query is empty or whitespace-only
            LLMError: If the API key is missing or the LLM API call fails
        """
        # Validate query
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        # Validate API key early — better error message than a cryptic 401
        if not self._settings.api_key:
            raise LLMError("API key not configured. Set LLM_API_KEY in your environment.")

        # Build messages
        system_prompt = self._build_system_prompt(context)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Get tool definitions (None if no adapter)
        tool_definitions = await self._get_tool_definitions()

        # Track tool calls and cumulative token usage across rounds
        recorded_tool_calls: list[ToolCall] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        model_name = ""

        # --- Tool-use loop ---
        # The LLM may respond with a tool call instead of text. When it does,
        # we execute the tool, append the result to the conversation, and call
        # the LLM again. This repeats until we get a text response or hit the
        # round limit.
        rounds_used = 0

        while True:
            # Build API call kwargs
            call_kwargs: dict[str, Any] = {
                "model": self._settings.model,
                "messages": messages,
                "temperature": self._settings.temperature,
                "max_tokens": self._settings.max_tokens,
                "api_key": self._settings.api_key,
            }

            # Include tools only if we have them AND haven't hit the limit
            if tool_definitions and rounds_used < self._max_tool_rounds:
                call_kwargs["tools"] = tool_definitions

            try:
                response = await acompletion(**call_kwargs)
            except Exception as e:
                raise LLMError(f"LLM API call failed: {e}", cause=e)

            # Track usage and model
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            model_name = response.model

            choice = response.choices[0]
            assistant_message = choice.message

            # Check if the LLM wants to call a tool
            if assistant_message.tool_calls and rounds_used < self._max_tool_rounds:
                # Process each tool call in this response
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Execute the tool, catching errors gracefully
                    try:
                        tool_result = await self._tool_adapter.call(tool_name, tool_args)
                        result_text = tool_result.get("text", str(tool_result))
                    except Exception as e:
                        # Pass error back to the LLM — let it decide how to handle it
                        # rather than crashing the whole request
                        logger.warning(f"Tool '{tool_name}' failed: {e}")
                        result_text = f"Error: Tool '{tool_name}' failed: {e}"

                    # Record for the response
                    recorded_tool_calls.append(
                        ToolCall(name=tool_name, arguments=tool_args, result=result_text)
                    )

                    # Append assistant's tool call + tool result to conversation
                    # so the LLM sees what happened on the next round
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                rounds_used += 1
                continue  # Loop back for the next LLM call

            # No tool call (or limit reached) — we have our text response
            text = assistant_message.content or ""
            break

        return LLMResponse(
            text=text,
            tool_calls=recorded_tool_calls,
            model=model_name,
            usage=TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            ),
        )
