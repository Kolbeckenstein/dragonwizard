#!/usr/bin/env python3
"""
Manual test script to validate the dice rolling MCP server interface.

This script connects to the MCP dice server and tests its actual behavior
to ensure our mocks and implementation are accurate.
"""

import asyncio
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def test_dice_server():
    """Test the dice rolling MCP server."""
    # Path to the dice server
    server_path = Path(__file__).parent.parent.parent.parent / "external" / "dice-rolling-mcp" / "dist" / "index.js"

    if not server_path.exists():
        print(f"❌ Server not found at: {server_path}")
        print("Run: make install-dice-server")
        return

    print(f"🎲 Testing MCP dice server at: {server_path}")
    print()

    # Create server parameters
    server_params = StdioServerParameters(
        command="node",
        args=[str(server_path)]
    )

    print("📡 Connecting to MCP server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            print("🤝 Initializing session...")
            init_result = await session.initialize()
            print(f"✅ Initialized: {json.dumps(init_result.model_dump(), indent=2)}")
            print()

            # List available tools
            print("🔍 Listing available tools...")
            tools_result = await session.list_tools()
            print(f"✅ Found {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Schema: {json.dumps(tool.inputSchema, indent=6)}")
            print()

            # Test a simple dice roll
            print("🎲 Testing simple roll: 1d20...")
            roll_result = await session.call_tool("dice_roll", arguments={"notation": "1d20"})
            print(f"✅ Roll result:")
            for content_item in roll_result.content:
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
            print()

            # Test advantage roll (D&D 5e)
            print("🎲 Testing advantage roll: 2d20kh1+5...")
            adv_result = await session.call_tool("dice_roll", arguments={"notation": "2d20kh1+5"})
            print(f"✅ Advantage result:")
            for content_item in adv_result.content:
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
            print()

            # Test disadvantage roll (D&D 5e)
            print("🎲 Testing disadvantage roll: 2d20kl1+3...")
            disadv_result = await session.call_tool("dice_roll", arguments={"notation": "2d20kl1+3"})
            print(f"✅ Disadvantage result:")
            for content_item in disadv_result.content:
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
            print()

            # Test ability score roll (D&D 5e)
            print("🎲 Testing ability score roll: 4d6kh3...")
            ability_result = await session.call_tool("dice_roll", arguments={"notation": "4d6kh3"})
            print(f"✅ Ability score result:")
            for content_item in ability_result.content:
                if hasattr(content_item, 'text'):
                    print(f"  Text: {content_item.text}")
            print()

            print("✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_dice_server())
