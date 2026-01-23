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

    try:
        from mcp_server.tools.profiling_tools import ProfilingTools
        print("✓ ProfilingTools imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ProfilingTools: {e}")
        return False

    try:
        from mcp_server.tools.model_tools import ModelTools
        print("✓ ModelTools imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ModelTools: {e}")
        return False

    try:
        from mcp_server.tools.recommendation_tools import RecommendationTools
        print("✓ RecommendationTools imported successfully")
    except Exception as e:
        print(f"✗ Failed to import RecommendationTools: {e}")
        return False

    try:
        from mcp_server.resources.model_registry import ModelRegistry
        print("✓ ModelRegistry imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ModelRegistry: {e}")
        return False

    try:
        from mcp_server.resources.profile_cache import ProfileCache
        print("✓ ProfileCache imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ProfileCache: {e}")
        return False

    try:
        from mcp_server.resources.benchmark_data import BenchmarkData
        print("✓ BenchmarkData imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BenchmarkData: {e}")
        return False

    try:
        from mcp_server.prompts.workflows import WorkflowPrompts
        print("✓ WorkflowPrompts imported successfully")
    except Exception as e:
        print(f"✗ Failed to import WorkflowPrompts: {e}")
        return False

    return True


def test_tool_definitions():
    """Test that tool definitions are properly structured."""
    print("\nTesting tool definitions...")

    from synthony.core.analyzer import StochasticDataAnalyzer
    from synthony.core.column_analyzer import ColumnAnalyzer
    from synthony.recommender.engine import ModelRecommendationEngine
    from mcp_server.tools.profiling_tools import ProfilingTools
    from mcp_server.tools.model_tools import ModelTools
    from mcp_server.tools.recommendation_tools import RecommendationTools

    try:
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

        return True
    except Exception as e:
        print(f"✗ Failed to get tool definitions: {e}")
        return False


def test_resource_definitions():
    """Test that resource definitions are properly structured."""
    print("\nTesting resource definitions...")

    from synthony.recommender.engine import ModelRecommendationEngine
    from mcp_server.resources.model_registry import ModelRegistry
    from mcp_server.resources.profile_cache import ProfileCache
    from mcp_server.resources.benchmark_data import BenchmarkData

    try:
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

        return True
    except Exception as e:
        print(f"✗ Failed to get resource definitions: {e}")
        return False


def test_prompt_definitions():
    """Test that prompt definitions are properly structured."""
    print("\nTesting prompt definitions...")

    from mcp_server.prompts.workflows import WorkflowPrompts

    try:
        workflow_prompts = WorkflowPrompts()
        prompt_defs = workflow_prompts.get_prompt_definitions()
        print(f"✓ WorkflowPrompts: {len(prompt_defs)} prompts defined")

        return True
    except Exception as e:
        print(f"✗ Failed to get prompt definitions: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Synthony MCP Server - Component Tests")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_tool_definitions()
    all_passed &= test_resource_definitions()
    all_passed &= test_prompt_definitions()

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
