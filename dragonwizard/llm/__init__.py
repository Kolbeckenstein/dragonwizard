"""
LLM Orchestration Layer.

Manages interactions with LLM APIs (Claude, GPT, local models via LiteLLM),
prompt construction, tool-call handling, and response generation.

This layer sits between the RAG Engine and the Discord Bot:

    RAGEngine.format_context()  →  context string
                                        ↓
    LLMOrchestrator.generate_response(query, context)
                                        ↓
                                   LLMResponse  →  Discord bot formats and sends

Key responsibilities:
- Construct system prompts with injected RAG context
- Call LLM APIs via LiteLLM (provider-agnostic)
- Handle tool-use loops (e.g., LLM requests a dice roll mid-response)
- Return structured responses with text, tool call records, and token usage

The orchestrator is stateless per call — every generate_response() builds the
full message history from scratch. Multi-turn conversation memory is a future
enhancement (see architecture.md backlog).
"""

from dragonwizard.llm.models import LLMError, LLMResponse, TokenUsage, ToolCall
from dragonwizard.llm.orchestrator import LLMOrchestrator

__all__ = [
    "LLMOrchestrator",
    "LLMResponse",
    "LLMError",
    "TokenUsage",
    "ToolCall",
]
