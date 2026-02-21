"""
Integration tests for DiceRollerTool.

These tests use the actual dice-rolling-mcp server subprocess
to verify end-to-end functionality.
"""

import os

import pytest


@pytest.fixture
def dice_server_path():
    """Get the path to the dice server index.js file."""
    # Assuming the dice server is at external/dice-rolling-mcp/dist/index.js
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    server_path = os.path.join(
        repo_root, "external", "dice-rolling-mcp", "dist", "index.js"
    )

    if not os.path.exists(server_path):
        pytest.skip(
            f"Dice server not found at {server_path}. Run 'make install-dice-server' first."
        )

    return server_path


class TestDiceRollerIntegration:
    """Integration tests with real dice server."""

    @pytest.mark.asyncio
    async def test_list_tools_real_server(self, dice_server_path):
        """Should list tools from real dice server."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            tools = await tool.list_tools()

            # Verify we got the dice_roll tool
            assert len(tools) >= 1
            dice_roll_tool = next((t for t in tools if t["name"] == "dice_roll"), None)
            assert dice_roll_tool is not None
            assert "description" in dice_roll_tool
            assert "input_schema" in dice_roll_tool

    @pytest.mark.asyncio
    async def test_simple_roll(self, dice_server_path):
        """Should successfully roll simple dice: 1d20."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            result = await tool.call("dice_roll", {"notation": "1d20"})

            # Verify result structure
            assert "text" in result
            assert "1d20" in result["text"]
            # Result should contain a number between 1 and 20
            assert any(str(i) in result["text"] for i in range(1, 21))

    @pytest.mark.asyncio
    async def test_advantage_roll(self, dice_server_path):
        """Should successfully roll with advantage: 2d20kh1."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            result = await tool.call("dice_roll", {"notation": "2d20kh1"})

            assert "text" in result
            # Should show it rolled 2 dice
            assert "2d20" in result["text"] or "kh1" in result["text"]

    @pytest.mark.asyncio
    async def test_disadvantage_roll(self, dice_server_path):
        """Should successfully roll with disadvantage: 2d20kl1."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            result = await tool.call("dice_roll", {"notation": "2d20kl1"})

            assert "text" in result
            # Should show it rolled 2 dice
            assert "2d20" in result["text"] or "kl1" in result["text"]

    @pytest.mark.asyncio
    async def test_ability_score_roll(self, dice_server_path):
        """Should successfully roll for ability scores: 4d6kh3."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            result = await tool.call("dice_roll", {"notation": "4d6kh3"})

            assert "text" in result
            assert "4d6" in result["text"] or "kh3" in result["text"]

    @pytest.mark.asyncio
    async def test_roll_with_modifier(self, dice_server_path):
        """Should successfully roll with modifier: 1d20+5."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            result = await tool.call("dice_roll", {"notation": "1d20+5"})

            assert "text" in result
            assert "1d20" in result["text"]
            assert "+5" in result["text"] or "5" in result["text"]

    @pytest.mark.asyncio
    async def test_invalid_notation(self, dice_server_path):
        """Should handle invalid dice notation gracefully."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            # This should raise an error or return an error message
            # The exact behavior depends on the dice server implementation
            with pytest.raises(Exception):  # May be ValueError, RuntimeError, etc.
                await tool.call("dice_roll", {"notation": "not_a_dice_notation"})

    @pytest.mark.asyncio
    async def test_multiple_sequential_rolls(self, dice_server_path):
        """Should handle multiple sequential roll requests."""
        from dragonwizard.tools.dice_roller import DiceRollerTool

        async with DiceRollerTool(server_path=dice_server_path) as tool:
            # Roll multiple times
            result1 = await tool.call("dice_roll", {"notation": "1d6"})
            result2 = await tool.call("dice_roll", {"notation": "1d8"})
            result3 = await tool.call("dice_roll", {"notation": "1d10"})

            # All should succeed
            assert "text" in result1
            assert "text" in result2
            assert "text" in result3

            # Verify they're different rolls
            assert "1d6" in result1["text"]
            assert "1d8" in result2["text"]
            assert "1d10" in result3["text"]
