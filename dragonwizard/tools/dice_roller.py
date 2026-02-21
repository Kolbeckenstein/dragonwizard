"""
MCP-based dice roller tool adapter.

Provides dice rolling functionality via an MCP server subprocess.
"""

from typing import Any
import os
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from dragonwizard.tools.base import ToolAdapter


class DiceRollerTool(ToolAdapter):
    """
    Dice rolling tool using MCP protocol.

    Spawns an MCP dice server as a subprocess and communicates
    via JSON-RPC over stdio.
    """

    def __init__(self, server_path: str | None = None):
        """Initialize the dice roller tool.

        Args:
            server_path: Path to the dice server index.js file.
                        If None, uses environment variable DICE_SERVER_PATH.
        """
        self._initialized = False
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._stdio_context = None
        self._session_context = None

        # Determine server path
        if server_path is None:
            server_path = os.environ.get("DICE_SERVER_PATH")

        self._server_path = server_path

    async def initialize(self) -> None:
        """Initialize the MCP dice server subprocess."""
        # Validate server path
        if self._server_path is None:
            raise ValueError(
                "Dice server path not specified. "
                "Provide server_path argument to __init__ or set DICE_SERVER_PATH environment variable."
            )

        server_path = Path(self._server_path)
        if not server_path.exists():
            raise FileNotFoundError(f"Dice server not found at: {server_path}")

        # Create server parameters for the dice rolling MCP server
        server_params = StdioServerParameters(
            command="node",
            args=[str(server_path)]
        )

        # Start the MCP server subprocess and store context manager
        self._stdio_context = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._stdio_context.__aenter__()

        # Create and initialize the client session and store context manager
        self._session_context = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_context.__aenter__()

        # Perform MCP initialization handshake
        await self._session.initialize()

        # Mark as initialized
        self._initialized = True

    async def shutdown(self) -> None:
        """Cleanly shut down the MCP server subprocess."""
        if not self._initialized:
            return  # Already shut down or never initialized

        # Exit the session context manager (closes session)
        if self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            self._session = None

        # Exit the stdio context manager (terminates subprocess)
        if self._stdio_context is not None:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None
            self._read_stream = None
            self._write_stream = None

        # Mark as no longer initialized
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry - initialize the tool."""
        await self.initialize()
        return self

    async def __aexit__(self, *_args):
        """Async context manager exit - shutdown the tool."""
        await self.shutdown()
        return None  # Don't suppress exceptions

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self._initialized:
            raise RuntimeError("Tool adapter not initialized")

        # Call the tool on the MCP server
        result = await self._session.call_tool(tool_name, arguments)

        # Extract text from the result content
        # MCP returns content as a list of content blocks
        text_parts = []
        for content in result.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)

        # Return formatted result compatible with our tool interface
        return {
            "text": " ".join(text_parts) if text_parts else ""
        }

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self._initialized:
            raise RuntimeError("Tool adapter not initialized")

        # Call the MCP server's list_tools method
        result = await self._session.list_tools()

        # Convert MCP tools to dictionaries compatible with Claude API
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            })

        return tools
