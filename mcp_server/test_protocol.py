#!/usr/bin/env python3
"""
Test MCP protocol communication with the Synthony MCP Server

This script properly tests the server without causing JSON parsing errors.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_mcp_protocol():
    """Test basic MCP protocol messages."""

    test_messages = [
        {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        },
        {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        },
        {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "params": {},
            "id": 3
        },
        {
            "jsonrpc": "2.0",
            "method": "prompts/list",
            "params": {},
            "id": 4
        }
    ]

    print("MCP Protocol Test Messages")
    print("=" * 60)

    for i, msg in enumerate(test_messages, 1):
        print(f"\n{i}. {msg['method']}")
        print("-" * 60)
        print(json.dumps(msg, indent=2))

    print("\n" + "=" * 60)
    print("To test these messages with the MCP server:")
    print("1. Start the server: python -m mcp_server.server")
    print("2. Pipe each message (one at a time):")
    print("   echo '<json>' | python -m mcp_server.server")
    print("\nNote: The server uses stdio transport, so it reads from stdin")
    print("and writes to stdout using JSON-RPC 2.0 protocol.")


if __name__ == "__main__":
    asyncio.run(test_mcp_protocol())
