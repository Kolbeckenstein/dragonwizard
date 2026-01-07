"""
Unit tests for DiceRollerTool adapter.

Tests mock at the JSON-RPC message level to ensure correct behavior
and document expected MCP protocol interactions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Test data: MCP JSON-RPC messages
MOCK_INITIALIZE_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {
            "name": "dice-roller",
            "version": "1.0.0"
        }
    }
}

MOCK_TOOLS_LIST_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "tools": [
            {
                "name": "roll_dice",
                "description": "Roll dice using standard notation (e.g., 2d20, 3d6+5, 4d6kh3)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Dice expression in standard notation"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    }
}

# Happy path test cases
MOCK_SIMPLE_ROLL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Rolling 1d20: [15] = 15"
            }
        ]
    }
}

MOCK_ADVANTAGE_ROLL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 4,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Rolling 2d20kh1+5 (advantage): [18, 7] -> 18 + 5 = 23"
            }
        ]
    }
}

MOCK_DISADVANTAGE_ROLL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 5,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Rolling 2d20kl1+3 (disadvantage): [18, 7] -> 7 + 3 = 10"
            }
        ]
    }
}

MOCK_ABILITY_SCORE_ROLL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 6,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Rolling 4d6kh3 (ability score): [6, 5, 4, 2] -> [6, 5, 4] = 15"
            }
        ]
    }
}

MOCK_MULTIPLE_DICE_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 7,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Rolling 8d6: [5, 4, 3, 6, 2, 1, 4, 5] = 30"
            }
        ]
    }
}

# Error case test data
MOCK_INVALID_EXPRESSION_ERROR = {
    "jsonrpc": "2.0",
    "id": 8,
    "error": {
        "code": -32602,
        "message": "Invalid dice expression",
        "data": {
            "expression": "2d20kh1kl2",
            "reason": "Cannot use both kh (keep highest) and kl (keep lowest)"
        }
    }
}

MOCK_INVALID_DICE_COUNT_ERROR = {
    "jsonrpc": "2.0",
    "id": 9,
    "error": {
        "code": -32602,
        "message": "Invalid parameters",
        "data": {
            "expression": "99d99999",
            "reason": "Dice count or faces exceeds maximum allowed"
        }
    }
}

MOCK_MALFORMED_EXPRESSION_ERROR = {
    "jsonrpc": "2.0",
    "id": 10,
    "error": {
        "code": -32602,
        "message": "Invalid parameters",
        "data": {
            "expression": "not_a_dice_expression",
            "reason": "Expression does not match dice notation pattern"
        }
    }
}

MOCK_METHOD_NOT_FOUND_ERROR = {
    "jsonrpc": "2.0",
    "id": 11,
    "error": {
        "code": -32601,
        "message": "Method not found",
        "data": {
            "method": "unknown_tool"
        }
    }
}


class TestDiceRollerToolInitialization:
    """Test initialization and shutdown behavior."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should successfully initialize and connect to MCP server."""
        from dragonwizard.tools.dice_roller import DiceRollerTool
        from mcp.client.stdio import StdioServerParameters
        from unittest.mock import patch, AsyncMock
        import tempfile
        import os

        # Create a temporary file to represent the dice server
        with tempfile.NamedTemporaryFile(mode='w', suffix='index.js', delete=False) as f:
            temp_server_path = f.name
            f.write("// mock server")

        try:
            # Mock the MCP stdio_client context manager
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_session = AsyncMock()

            # Mock successful initialization response
            mock_session.initialize = AsyncMock(return_value=MOCK_INITIALIZE_RESPONSE["result"])

            with patch("dragonwizard.tools.dice_roller.stdio_client") as mock_stdio_client:
                # Setup the async context manager mock
                mock_stdio_client.return_value.__aenter__ = AsyncMock(
                    return_value=(mock_read_stream, mock_write_stream)
                )
                mock_stdio_client.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("dragonwizard.tools.dice_roller.ClientSession") as mock_client_session:
                    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)

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
        # TODO: Implement
        # Expected: FileNotFoundError with helpful message about server path
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_initialize_timeout(self):
        """Should raise TimeoutError if server doesn't respond to initialize."""
        # TODO: Implement
        # Expected: TimeoutError after configured timeout period
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Should cleanly shut down MCP server subprocess."""
        # TODO: Implement
        # Expected behavior:
        # 1. Send shutdown notification (if protocol supports)
        # 2. Terminate subprocess gracefully
        # 3. Wait for process to exit
        # 4. Clean up resources
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Should work as async context manager."""
        # TODO: Implement
        # Expected behavior:
        # async with DiceRollerTool() as tool:
        #     # tool is initialized here
        #     pass
        # # tool is shut down here
        pytest.skip("Implementation pending")


class TestDiceRollerToolListTools:
    """Test tool listing functionality."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_correct_schema(self):
        """Should return roll_dice tool with correct schema."""
        # TODO: Implement
        # Expected: list_tools() returns tool schema matching MOCK_TOOLS_LIST_RESPONSE
        # Schema should be compatible with Claude API tool format
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_list_tools_before_initialization(self):
        """Should raise RuntimeError if called before initialize."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        tool = DiceRollerTool()

        # Should raise RuntimeError when calling list_tools before initialize
        with pytest.raises(RuntimeError, match="not initialized"):
            await tool.list_tools()


class TestDiceRollerToolHappyPath:
    """Test successful dice rolling scenarios."""

    @pytest.mark.asyncio
    async def test_simple_roll(self):
        """Should successfully roll simple dice expression: 1d20."""
        # TODO: Implement with mock
        # Mock MCP response: MOCK_SIMPLE_ROLL_RESPONSE
        # Expected: call("roll_dice", {"expression": "1d20"}) returns parsed result
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_advantage_roll(self):
        """Should successfully roll with advantage: 2d20kh1+5."""
        # TODO: Implement
        # Mock response: MOCK_ADVANTAGE_ROLL_RESPONSE
        # This is the canonical D&D advantage roll
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_disadvantage_roll(self):
        """Should successfully roll with disadvantage: 2d20kl1+3."""
        # TODO: Implement
        # Mock response: MOCK_DISADVANTAGE_ROLL_RESPONSE
        # This is the canonical D&D disadvantage roll
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_ability_score_roll(self):
        """Should successfully roll for ability scores: 4d6kh3."""
        # TODO: Implement
        # Mock response: MOCK_ABILITY_SCORE_ROLL_RESPONSE
        # Standard D&D ability score generation
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_multiple_dice_roll(self):
        """Should successfully roll multiple dice: 8d6."""
        # TODO: Implement
        # Mock response: MOCK_MULTIPLE_DICE_RESPONSE
        # Common for damage rolls
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_roll_with_modifier(self):
        """Should successfully roll with positive modifier: 1d20+5."""
        # TODO: Implement
        # Test that modifiers are correctly included
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_roll_with_negative_modifier(self):
        """Should successfully roll with negative modifier: 1d20-2."""
        # TODO: Implement
        # Test that negative modifiers work
        pytest.skip("Implementation pending")


class TestDiceRollerToolErrorCases:
    """Test error handling for invalid inputs and edge cases."""

    @pytest.mark.asyncio
    async def test_conflicting_keep_modifiers(self):
        """Should raise ValueError for conflicting kh/kl: 2d20kh1kl2."""
        # TODO: Implement
        # Mock response: MOCK_INVALID_EXPRESSION_ERROR
        # Expected: ValueError with clear message
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_excessive_dice_count(self):
        """Should raise ValueError for excessive dice: 99d99999."""
        # TODO: Implement
        # Mock response: MOCK_INVALID_DICE_COUNT_ERROR
        # Expected: ValueError about exceeding limits
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_malformed_expression(self):
        """Should raise ValueError for malformed expression: 'not_a_dice'."""
        # TODO: Implement
        # Mock response: MOCK_MALFORMED_EXPRESSION_ERROR
        # Expected: ValueError with parsing error
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        """Should raise ValueError for empty expression."""
        # TODO: Implement
        # Expected: ValueError("expression cannot be empty")
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_wrong_argument_type(self):
        """Should raise TypeError for non-string expression."""
        # TODO: Implement
        # call("roll_dice", {"expression": 123}) should raise TypeError
        # Expected: TypeError with message about expecting string
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_missing_required_argument(self):
        """Should raise ValueError for missing expression argument."""
        # TODO: Implement
        # call("roll_dice", {}) should raise ValueError
        # Expected: ValueError("missing required argument: expression")
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_unknown_tool_name(self):
        """Should raise ValueError for unknown tool name."""
        # TODO: Implement
        # Mock response: MOCK_METHOD_NOT_FOUND_ERROR
        # call("unknown_tool", {}) should raise ValueError
        # Expected: ValueError("Unknown tool: unknown_tool")
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_call_before_initialization(self):
        """Should raise RuntimeError if call before initialize."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        tool = DiceRollerTool()

        # Should raise RuntimeError when calling tool before initialize
        with pytest.raises(RuntimeError, match="not initialized"):
            await tool.call("roll_dice", {"expression": "1d20"})

    @pytest.mark.asyncio
    async def test_zero_dice(self):
        """Should raise ValueError for zero dice: 0d20."""
        # TODO: Implement
        # Expected: ValueError about dice count being at least 1
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_zero_sides(self):
        """Should raise ValueError for zero sides: 1d0."""
        # TODO: Implement
        # Expected: ValueError about dice faces being at least 1
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_negative_dice_count(self):
        """Should raise ValueError for negative dice count: -1d20."""
        # TODO: Implement
        # Expected: ValueError about negative dice count
        pytest.skip("Implementation pending")


class TestDiceRollerToolEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_die_single_side(self):
        """Should handle degenerate case: 1d1 (always rolls 1)."""
        # TODO: Implement
        # Expected: Valid roll result of 1
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_maximum_reasonable_dice(self):
        """Should handle maximum reasonable dice count."""
        # TODO: Implement - what's the max we support?
        # Maybe 100d100 or similar
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_keep_more_than_rolled(self):
        """Should handle kh when k > dice rolled: 2d20kh5."""
        # TODO: Implement
        # Expected behavior: keep all dice (effectively 2d20)
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_multiple_sequential_rolls(self):
        """Should handle multiple sequential roll requests."""
        # TODO: Implement
        # Ensure state doesn't leak between calls
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_concurrent_rolls(self):
        """Should handle concurrent roll requests safely."""
        # TODO: Implement
        # If we support concurrent calls, test thread safety
        pytest.skip("Implementation pending")


class TestDiceRollerToolResultFormat:
    """Test that results are correctly formatted for consumption."""

    @pytest.mark.asyncio
    async def test_result_contains_text_field(self):
        """Should return dict with 'text' field containing roll result."""
        # TODO: Implement
        # Expected result format:
        # {"text": "Rolling 1d20: [15] = 15"}
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_result_is_human_readable(self):
        """Should return human-readable text suitable for LLM."""
        # TODO: Implement
        # Result should be formatted for inclusion in LLM response
        pytest.skip("Implementation pending")

    @pytest.mark.asyncio
    async def test_result_includes_breakdown(self):
        """Should include breakdown of individual die results."""
        # TODO: Implement
        # For 2d20kh1: should show both rolls and which was kept
        pytest.skip("Implementation pending")
