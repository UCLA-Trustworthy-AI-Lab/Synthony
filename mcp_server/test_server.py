#!/usr/bin/env python3
"""
Test script for Synthony MCP Server

Validates that all components are properly structured and importable.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all MCP server components can be imported."""
    print("Testing MCP server imports...")

    from mcp_server.tools.profiling_tools import ProfilingTools  # noqa: F401
    print("✓ ProfilingTools imported successfully")

    from mcp_server.tools.model_tools import ModelTools  # noqa: F401
    print("✓ ModelTools imported successfully")

    from mcp_server.tools.recommendation_tools import RecommendationTools  # noqa: F401
    print("✓ RecommendationTools imported successfully")

    from mcp_server.resources.model_registry import ModelRegistry  # noqa: F401
    print("✓ ModelRegistry imported successfully")

    from mcp_server.resources.profile_cache import ProfileCache  # noqa: F401
    print("✓ ProfileCache imported successfully")

    from mcp_server.resources.benchmark_data import BenchmarkData  # noqa: F401
    print("✓ BenchmarkData imported successfully")

    from mcp_server.prompts.workflows import WorkflowPrompts  # noqa: F401
    print("✓ WorkflowPrompts imported successfully")


def test_tool_definitions():
    """Test that tool definitions are properly structured."""
    print("\nTesting tool definitions...")

    from synthony.core.analyzer import StochasticDataAnalyzer
    from synthony.core.column_analyzer import ColumnAnalyzer
    from synthony.recommender.engine import ModelRecommendationEngine
    from mcp_server.tools.profiling_tools import ProfilingTools
    from mcp_server.tools.model_tools import ModelTools
    from mcp_server.tools.recommendation_tools import RecommendationTools

    analyzer = StochasticDataAnalyzer()
    column_analyzer = ColumnAnalyzer()
    recommender = ModelRecommendationEngine()

    profiling_tools = ProfilingTools(analyzer, column_analyzer)
    tool_defs = profiling_tools.get_tool_definitions()
    print(f"✓ ProfilingTools: {len(tool_defs)} tools defined")

    model_tools = ModelTools(recommender)
    tool_defs = model_tools.get_tool_definitions()
    print(f"✓ ModelTools: {len(tool_defs)} tools defined")

    recommendation_tools = RecommendationTools(recommender)
    tool_defs = recommendation_tools.get_tool_definitions()
    print(f"✓ RecommendationTools: {len(tool_defs)} tools defined")


def test_resource_definitions():
    """Test that resource definitions are properly structured."""
    print("\nTesting resource definitions...")

    from synthony.recommender.engine import ModelRecommendationEngine
    from mcp_server.resources.model_registry import ModelRegistry
    from mcp_server.resources.profile_cache import ProfileCache
    from mcp_server.resources.benchmark_data import BenchmarkData

    recommender = ModelRecommendationEngine()

    model_registry = ModelRegistry(recommender)
    resource_defs = model_registry.get_resource_definitions()
    print(f"✓ ModelRegistry: {len(resource_defs)} resources defined")

    profile_cache = ProfileCache()
    resource_defs = profile_cache.get_resource_definitions()
    print(f"✓ ProfileCache: {len(resource_defs)} resources defined")

    benchmark_data = BenchmarkData()
    resource_defs = benchmark_data.get_resource_definitions()
    print(f"✓ BenchmarkData: {len(resource_defs)} resources defined")


def test_prompt_definitions():
    """Test that prompt definitions are properly structured."""
    print("\nTesting prompt definitions...")

    from mcp_server.prompts.workflows import WorkflowPrompts

    workflow_prompts = WorkflowPrompts()
    prompt_defs = workflow_prompts.get_prompt_definitions()
    print(f"✓ WorkflowPrompts: {len(prompt_defs)} prompts defined")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Synthony MCP Server - Component Tests")
    print("=" * 60)

    all_passed = True

    for check in (
        test_imports,
        test_tool_definitions,
        test_resource_definitions,
        test_prompt_definitions,
    ):
        try:
            check()
        except Exception as e:
            print(f"✗ {check.__name__} failed: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
