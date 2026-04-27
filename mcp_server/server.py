#!/usr/bin/env python3
"""
Synthony MCP Server - Main Entry Point

This MCP server exposes Synthony's data profiling and model recommendation
capabilities to AI agents through the Model Context Protocol.

Protocol: JSON-RPC 2.0 over stdio transport
Target: Local AI agent integration (e.g., Claude Code)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
)

# Import Synthony core components
from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine

# Import database functions
from synthony.api.database import get_active_prompt

# Import MCP server components
from mcp_server.tools.data_tools import DataTools
from mcp_server.tools.profiling_tools import ProfilingTools
from mcp_server.tools.model_tools import ModelTools
from mcp_server.tools.recommendation_tools import RecommendationTools
from mcp_server.tools.benchmark_tools import BenchmarkTools
from mcp_server.resources.model_registry import ModelRegistry
from mcp_server.resources.profile_cache import ProfileCache
from mcp_server.resources.benchmark_data import BenchmarkData
from mcp_server.prompts.workflows import WorkflowPrompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("synthony-mcp")


class SynthonyMCPServer:
    """
    Synthony MCP Server

    Provides AI agents with access to:
    - Tools: Executable functions for profiling and recommendation
    - Resources: Read-only access to model registry, cached profiles, benchmarks
    - Prompts: Guided workflows for common tasks
    """

    def __init__(self, verbose: bool = False):
        """Initialize the MCP server with Synthony components.

        Args:
            verbose: If True, log all tool commands and JSON outputs to stderr.
                     Also enabled by setting the MCP_DEBUG environment variable.
        """
        self.server = Server("synthony_mcp")
        self.verbose = verbose or bool(os.getenv("MCP_DEBUG"))

        # Initialize Synthony core components
        self.analyzer = StochasticDataAnalyzer()
        self.column_analyzer = ColumnAnalyzer()

        # Configure LLM from environment (same pattern as API server)
        vllm_url = os.getenv("VLLM_URL")
        if vllm_url:
            openai_api_key = os.getenv("VLLM_API_KEY")
            openai_base_url = vllm_url
            openai_model = os.getenv("VLLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_base_url = os.getenv("OPENAI_URL")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.recommender = ModelRecommendationEngine(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
        )

        # Initialize MCP components
        self.data_tools = DataTools()
        self.profiling_tools = ProfilingTools(self.analyzer, self.column_analyzer)
        self.model_tools = ModelTools(self.recommender)
        self.recommendation_tools = RecommendationTools(self.recommender)
        self.benchmark_tools = BenchmarkTools()
        self.model_registry = ModelRegistry(self.recommender)
        self.profile_cache = ProfileCache()
        self.benchmark_data = BenchmarkData()
        self.workflow_prompts = WorkflowPrompts()

        # Register handlers
        self._register_handlers()

        if self.verbose:
            print(f"[VERBOSE] Synthony MCP Server initialized", file=sys.stderr)
        logger.info("Synthony MCP Server initialized")

    def _register_handlers(self):
        """Register all MCP protocol handlers."""

        # Tools handlers
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            tools = []

            # Data tools
            tools.extend(self.data_tools.get_tool_definitions())

            # Profiling tools
            tools.extend(self.profiling_tools.get_tool_definitions())

            # Model tools
            tools.extend(self.model_tools.get_tool_definitions())

            # Recommendation tools
            tools.extend(self.recommendation_tools.get_tool_definitions())

            # Benchmark tools
            tools.extend(self.benchmark_tools.get_tool_definitions())

            logger.info(f"Listed {len(tools)} tools")
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Execute a tool with given arguments."""
            logger.info(f"Calling tool: {name} with arguments: {arguments}")

            # Verbose logging: Print tool command received
            if self.verbose:
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"[VERBOSE] Tool Called: {name}", file=sys.stderr)
                print(f"[VERBOSE] Arguments:", file=sys.stderr)
                print(json.dumps(arguments, indent=2), file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)

            try:
                # Route to appropriate tool handler
                if name in self.data_tools.get_tool_names():
                    result = await self.data_tools.execute_tool(name, arguments)
                elif name in self.profiling_tools.get_tool_names():
                    result = await self.profiling_tools.execute_tool(name, arguments)
                elif name in self.model_tools.get_tool_names():
                    result = await self.model_tools.execute_tool(name, arguments)
                elif name in self.recommendation_tools.get_tool_names():
                    result = await self.recommendation_tools.execute_tool(name, arguments)
                elif name in self.benchmark_tools.get_tool_names():
                    result = await self.benchmark_tools.execute_tool(name, arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Verbose logging: Print JSON output
                if self.verbose:
                    print(f"\n{'-'*80}", file=sys.stderr)
                    print(f"[VERBOSE] Tool Response: {name}", file=sys.stderr)
                    print(f"[VERBOSE] Result:", file=sys.stderr)
                    print(json.dumps(result, indent=2), file=sys.stderr)
                    print(f"{'-'*80}\n", file=sys.stderr)

                # Format result as TextContent
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )
                ]

            except Exception as e:
                logger.error(f"Tool execution error: {e}")

                # Verbose logging: Print error
                if self.verbose:
                    print(f"\n{'!'*80}", file=sys.stderr)
                    print(f"[VERBOSE] Tool Error: {name} — {e}", file=sys.stderr)
                    print(f"{'!'*80}\n", file=sys.stderr)

                # Re-raise so the MCP framework returns isError=True in the protocol response
                raise

        # Resources handlers
        @self.server.list_resources()
        async def list_resources() -> List[Dict[str, str]]:
            """List all available resources."""
            resources = []

            # Model registry resources
            resources.extend(self.model_registry.get_resource_definitions())

            # Profile cache resources
            resources.extend(self.profile_cache.get_resource_definitions())

            # Benchmark data resources
            resources.extend(self.benchmark_data.get_resource_definitions())

            logger.info(f"Listed {len(resources)} resources")
            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI."""
            logger.info(f"Reading resource: {uri}")

            try:
                # Route to appropriate resource handler
                if uri.startswith("models://"):
                    result = await self.model_registry.read_resource(uri)
                elif uri.startswith("datasets://profiles/"):
                    result = await self.profile_cache.read_resource(uri)
                elif uri.startswith("benchmarks://"):
                    result = await self.benchmark_data.read_resource(uri)
                elif uri.startswith("guidelines://"):
                    result = await self._read_guidelines(uri)
                else:
                    raise ValueError(f"Unknown resource URI: {uri}")

                return json.dumps(result, indent=2)

            except Exception as e:
                logger.error(f"Resource read error: {e}")
                return json.dumps({
                    "error": str(e),
                    "uri": uri
                }, indent=2)

        # Prompts handlers
        @self.server.list_prompts()
        async def list_prompts() -> List[Dict[str, Any]]:
            """List all available prompts."""
            prompts = self.workflow_prompts.get_prompt_definitions()
            logger.info(f"Listed {len(prompts)} prompts")
            return prompts

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
            """Get a prompt by name with optional arguments."""
            logger.info(f"Getting prompt: {name} with arguments: {arguments}")

            try:
                result = await self.workflow_prompts.get_prompt(name, arguments or {})
                return result

            except Exception as e:
                logger.error(f"Prompt retrieval error: {e}")
                return {
                    "error": str(e),
                    "prompt": name,
                    "arguments": arguments
                }

    async def _read_guidelines(self, uri: str) -> Dict[str, Any]:
        """Read system guidelines from the active system prompt in database."""
        if uri == "guidelines://system-prompt":
            # Get active prompt from database
            prompt = get_active_prompt()

            if not prompt:
                raise ValueError("No active system prompt found in database")

            return {
                "uri": uri,
                "type": "guidelines",
                "content": prompt.content,
                "version": prompt.version,
                "prompt_id": prompt.prompt_id,
                "created_at": prompt.created_at.isoformat(),
                "description": "Current knowledge base for model capability scoring"
            }

        raise ValueError(f"Unknown guidelines URI: {uri}")

    async def run(self):
        """Run the MCP server with stdio transport."""
        logger.info("Starting Synthony MCP Server on stdio transport")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Synthony MCP Server - Model Context Protocol server for synthetic data model recommendations"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging: print all tool commands and JSON outputs to stderr"
    )

    args = parser.parse_args()

    # Initialize server with verbose flag
    server = SynthonyMCPServer(verbose=args.verbose)

    if args.verbose:
        print(f"[VERBOSE] Starting Synthony MCP Server in verbose mode", file=sys.stderr)
        print(f"[VERBOSE] All tool calls and responses will be logged to stderr\n", file=sys.stderr)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
