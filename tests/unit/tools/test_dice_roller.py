"""
Unit tests for DiceRollerTool adapter.

These tests focus on the wrapper logic: initialization, shutdown,
error handling, and state management. For end-to-end functionality
with the actual dice server, see tests/integration/test_dice_roller_integration.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test data for mocking MCP responses
MOCK_INITIALIZE_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "dice-roller", "version": "1.0.0"},
    },
}


class TestDiceRollerToolInitialization:
    """Test initialization and shutdown behavior."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should successfully initialize and connect to MCP server."""
        import os
        import tempfile
        from unittest.mock import AsyncMock, patch

        from mcp.client.stdio import StdioServerParameters

        from dragonwizard.tools.dice_roller import DiceRollerTool

        # Create a temporary file to represent the dice server
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="index.js", delete=False
        ) as f:
            temp_server_path = f.name
            f.write("// mock server")

        try:
            # Mock the MCP stdio_client context manager
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_session = AsyncMock()

            # Mock successful initialization response
            mock_session.initialize = AsyncMock(
                return_value=MOCK_INITIALIZE_RESPONSE["result"]
            )

            with patch(
                "dragonwizard.tools.dice_roller.stdio_client"
            ) as mock_stdio_client:
                # Setup the async context manager mock
                mock_stdio_client.return_value.__aenter__ = AsyncMock(
                    return_value=(mock_read_stream, mock_write_stream)
                )
                mock_stdio_client.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch(
                    "dragonwizard.tools.dice_roller.ClientSession"
                ) as mock_client_session:
                    mock_client_session.return_value.__aenter__ = AsyncMock(
                        return_value=mock_session
                    )
                    mock_client_session.return_value.__aexit__ = AsyncMock(
                        return_value=None
                    )

                    tool = DiceRollerTool(server_path=temp_server_path)
                    await tool.initialize()

                    # Verify tool is now initialized
                    assert tool._initialized is True

                    # Verify MCP client was created with correct server parameters
                    # Using jimmcq/dice-rolling-mcp server: node dist/index.js
                    call_args = mock_stdio_client.call_args
                    assert call_args is not None
                    server_params = call_args[0][0]  # First positional argument
                    assert isinstance(server_params, StdioServerParameters)
                    assert server_params.command == "node"
                    # The path to index.js will be configurable, but verify args structure
                    assert len(server_params.args) > 0
                    assert server_params.args[-1].endswith("index.js")

                    # Verify session was initialized
                    mock_session.initialize.assert_called_once()
        finally:
            # Clean up temp file
            os.unlink(temp_server_path)

    @pytest.mark.asyncio
    async def test_initialize_server_not_found(self):
        """Should raise FileNotFoundError if MCP server executable not found."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        # Use a path that doesn't exist
        tool = DiceRollerTool(server_path="/nonexistent/path/to/dice/server.js")

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Dice server not found at"):
            await tool.initialize()

    @pytest.mark.asyncio
    async def test_initialize_no_server_path(self):
        """Should raise ValueError if no server path provided."""
        from dragonwizard.tools.dice_roller import DiceRollerTool
        import os

        # Clear environment variable if it exists
        old_env = os.environ.pop("DICE_SERVER_PATH", None)
        try:
            tool = DiceRollerTool()

            with pytest.raises(ValueError, match="Dice server path not specified"):
                await tool.initialize()
        finally:
            # Restore environment variable
            if old_env is not None:
                os.environ["DICE_SERVER_PATH"] = old_env

    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Should cleanly shut down MCP server subprocess."""
        import os
        import tempfile

        from dragonwizard.tools.dice_roller import DiceRollerTool

        # Create a temporary file to represent the dice server
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="index.js", delete=False
        ) as f:
            temp_server_path = f.name
            f.write("// mock server")

        try:
            # Mock the MCP stdio_client context manager
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_session = AsyncMock()
            mock_stdio_context = MagicMock()
            mock_session_context = MagicMock()

            # Mock successful initialization response
            mock_session.initialize = AsyncMock(
                return_value=MOCK_INITIALIZE_RESPONSE["result"]
            )

            with patch(
                "dragonwizard.tools.dice_roller.stdio_client"
            ) as mock_stdio_client:
                # Setup the async context manager mock
                mock_stdio_context.__aenter__ = AsyncMock(
                    return_value=(mock_read_stream, mock_write_stream)
                )
                mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
                mock_stdio_client.return_value = mock_stdio_context

                with patch(
                    "dragonwizard.tools.dice_roller.ClientSession"
                ) as mock_client_session:
                    mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session_context.__aexit__ = AsyncMock(return_value=None)
                    mock_client_session.return_value = mock_session_context

                    tool = DiceRollerTool(server_path=temp_server_path)
                    await tool.initialize()

                    # Verify tool is initialized
                    assert tool._initialized is True

                    # Now shut down
                    await tool.shutdown()

                    # Verify __aexit__ was called on both context managers
                    mock_stdio_context.__aexit__.assert_called_once()
                    mock_session_context.__aexit__.assert_called_once()

                    # Verify tool is no longer initialized
                    assert tool._initialized is False
        finally:
            # Clean up temp file
            os.unlink(temp_server_path)

    @pytest.mark.asyncio
    async def test_shutdown_when_not_initialized(self):
        """Should handle shutdown gracefully when not initialized."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        tool = DiceRollerTool()

        # Should not raise an error
        await tool.shutdown()

        # Tool should still be not initialized
        assert tool._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Should work as async context manager."""
        import os
        import tempfile

        from dragonwizard.tools.dice_roller import DiceRollerTool

        # Create a temporary file to represent the dice server
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix="index.js", delete=False
        )
        temp_server_path = temp_file.name
        temp_file.write("// mock server")
        temp_file.close()

        try:
            # Mock the MCP stdio_client context manager
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_session = AsyncMock()
            mock_stdio_context = MagicMock()
            mock_session_context = MagicMock()

            # Mock successful initialization response
            mock_session.initialize = AsyncMock(
                return_value=MOCK_INITIALIZE_RESPONSE["result"]
            )

            with patch(
                "dragonwizard.tools.dice_roller.stdio_client"
            ) as mock_stdio_client:
                # Setup the async context manager mock
                mock_stdio_context.__aenter__ = AsyncMock(
                    return_value=(mock_read_stream, mock_write_stream)
                )
                mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
                mock_stdio_client.return_value = mock_stdio_context

                with patch(
                    "dragonwizard.tools.dice_roller.ClientSession"
                ) as mock_client_session:
                    mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session_context.__aexit__ = AsyncMock(return_value=None)
                    mock_client_session.return_value = mock_session_context

                    # Use tool as context manager
                    async with DiceRollerTool(server_path=temp_server_path) as tool:
                        # Tool should be initialized inside the context
                        assert tool._initialized is True

                    # Tool should be shut down after exiting the context
                    assert tool._initialized is False

                    # Verify __aexit__ was called on both context managers
                    mock_stdio_context.__aexit__.assert_called_once()
                    mock_session_context.__aexit__.assert_called_once()
        finally:
            # Clean up temp file
            os.unlink(temp_server_path)


class TestDiceRollerToolErrorHandling:
    """Test error handling for wrapper methods."""

    @pytest.mark.asyncio
    async def test_list_tools_before_initialization(self):
        """Should raise RuntimeError if list_tools called before initialize."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        tool = DiceRollerTool()

        # Should raise RuntimeError when calling list_tools before initialize
        with pytest.raises(RuntimeError, match="not initialized"):
            await tool.list_tools()

    @pytest.mark.asyncio
    async def test_call_before_initialization(self):
        """Should raise RuntimeError if call invoked before initialize."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        tool = DiceRollerTool()

        # Should raise RuntimeError when calling tool before initialize
        with pytest.raises(RuntimeError, match="not initialized"):
            await tool.call("roll_dice", {"expression": "1d20"})
