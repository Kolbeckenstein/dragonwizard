"""
Base classes for tool adapters.

Provides abstract interfaces for integrating external tools (MCP servers, APIs, etc.)
that can be invoked by the LLM during response generation.
"""

from abc import ABC, abstractmethod
from typing import Any


class ToolAdapter(ABC):
    """
    Abstract base class for tool adapters.

    Tool adapters provide a uniform interface for calling external tools,
    whether they're MCP servers, REST APIs, or other integrations.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the tool adapter.

        This may involve starting subprocesses, establishing connections,
        or performing handshakes with external services.

        Raises:
            ConnectionError: If initialization fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanly shut down the tool adapter.

        This should close connections, terminate subprocesses,
        and release any resources.
        """
        pass

    @abstractmethod
    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a tool with the given arguments.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool-specific arguments

        Returns:
            Tool execution result as a dictionary

        Raises:
            ValueError: If tool_name is unknown or arguments are invalid
            RuntimeError: If tool execution fails
        """
        pass

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools from this adapter.

        Returns:
            List of tool schemas compatible with LLM tool use format.
            Each schema includes name, description, and input schema.

        Example:
            [
                {
                    "name": "roll_dice",
                    "description": "Roll dice using standard notation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Dice notation (e.g., 2d20kh1+5)"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            ]
        """
        pass

    async def __aenter__(self):
        """Context manager entry - initialize the adapter."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown the adapter."""
        await self.shutdown()
        return False
