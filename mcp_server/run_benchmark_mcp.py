#!/usr/bin/env python3
"""
MCP Benchmark Tool Runner

Demonstrates how to use the benchmark_compare MCP tool with path templates:
- original_path = './dataset/input_data'
- synthetic_path = './dataset/synth_data/spark/'
- evaluation_path = './dataset/synth_data/spark/{name}/test_data.csv'
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkMCPRunner:
    """Helper to run benchmark comparisons using the MCP tool."""
    
    def __init__(
        self,
        original_base: str = "./dataset/input_data",
        synthetic_base: str = "./dataset/synth_data/spark",
    ):
        """
        Initialize with path templates.
        
        Args:
            original_base: Base directory for original datasets
            synthetic_base: Base directory for synthetic datasets
        """
        self.original_base = Path(original_base)
        self.synthetic_base = Path(synthetic_base)
    
    def resolve_paths(
        self,
        dataset_name: str,
        synthetic_model: str = "aim",
    ) -> Optional[Dict[str, str]]:
        """
        Resolve full paths for a dataset and model combination.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'abalone', 'Bean')
            synthetic_model: Name of the synthetic model (e.g., 'aim', 'ctgan')
        
        Returns:
            Dict with 'original_path', 'synthetic_path', 'evaluation_path' keys,
            or None if files don't exist.
        """
        # Original dataset file
        original_path = self.original_base / f"{dataset_name}.csv"
        
        # Synthetic dataset file (note: may have dataset name prefix in filename)
        # Try both patterns: {dataset_name}_synthetic_{model}_1000.csv and {synthetic_model}_1000.csv
        synth_dir = self.synthetic_base / dataset_name
        synthetic_path = synth_dir / f"{dataset_name}_synthetic_{synthetic_model}_1000.csv"
        
        if not synthetic_path.exists():
            # Try alternate naming pattern
            synthetic_path = synth_dir / f"synthetic_{synthetic_model}_1000.csv"
        
        # Evaluation (test) dataset
        evaluation_path = synth_dir / "test_data.csv"
        
        # Validate paths
        if not original_path.exists():
            print(f"⚠ Original file not found: {original_path}")
            return None
        
        if not synthetic_path.exists():
            print(f"⚠ Synthetic file not found: {synthetic_path}")
            return None
        
        if not evaluation_path.exists():
            print(f"⚠ Evaluation file not found: {evaluation_path}")
        
        return {
            "original_path": str(original_path),
            "synthetic_path": str(synthetic_path),
            "evaluation_path": str(evaluation_path) if evaluation_path.exists() else None,
        }
    
    async def run_benchmark(
        self,
        dataset_name: str,
        synthetic_model: str = "aim",
        include_dp: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Run benchmark comparison for a dataset/model pair.
        
        Args:
            dataset_name: Name of the dataset
            synthetic_model: Name of the synthetic model
            include_dp: Whether to include differential privacy metrics
        
        Returns:
            Benchmark result dict, or None if paths couldn't be resolved.
        """
        from mcp_server.tools.benchmark_tools import BenchmarkTools
        
        paths = self.resolve_paths(dataset_name, synthetic_model)
        if not paths:
            return None
        
        benchmark_tools = BenchmarkTools()
        
        # Prepare arguments
        args = {
            "original_path": paths["original_path"],
            "synthetic_path": paths["synthetic_path"],
        }
        
        # Add evaluation path if it exists and DP is requested
        if include_dp and paths.get("evaluation_path"):
            args["evaluation_path"] = paths["evaluation_path"]
        
        print(f"\nRunning benchmark: {dataset_name} + {synthetic_model}")
        print(f"  Original: {args['original_path']}")
        print(f"  Synthetic: {args['synthetic_path']}")
        if "evaluation_path" in args:
            print(f"  Evaluation: {args['evaluation_path']}")
        
        # Execute the tool
        try:
            result = await benchmark_tools.execute_tool("benchmark_compare", args)
            return result
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            return None
    
    async def run_batch(
        self,
        datasets: list,
        models: list,
        include_dp: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks for multiple dataset/model combinations.
        
        Args:
            datasets: List of dataset names
            models: List of model names
            include_dp: Whether to include differential privacy metrics
        
        Returns:
            Dict mapping "{dataset}_{model}" to benchmark results
        """
        results = {}
        total = len(datasets) * len(models)
        completed = 0
        
        for dataset_name in datasets:
            for model_name in models:
                key = f"{dataset_name}_{model_name}"
                result = await self.run_benchmark(dataset_name, model_name, include_dp)
                results[key] = result
                completed += 1
                
                if result:
                    quality = result.get("quality_score", result.get("fidelity", {}).get("overall_fidelity", "N/A"))
                    print(f"  ✓ Quality: {quality}")
                else:
                    print(f"  ✗ Failed")
                
                print(f"  Progress: {completed}/{total}")
        
        return results


async def test_single_benchmark():
    """Test: Run a single benchmark with path resolution."""
    print("=" * 70)
    print("Test 1: Single Benchmark Execution")
    print("=" * 70)
    
    runner = BenchmarkMCPRunner()
    result = await runner.run_benchmark("abalone", "aim", include_dp=True)
    
    if result:
        print("\n✓ Benchmark completed successfully")
        
        # Show summary metrics
        if "fidelity" in result:
            print(f"\nFidelity: {json.dumps(result['fidelity'], indent=2)}")
        
        if "utility" in result:
            print(f"\nUtility: {json.dumps(result['utility'], indent=2)}")
        
        if "privacy" in result:
            print(f"\nPrivacy: {json.dumps(result['privacy'], indent=2)}")
        
        if "differential_privacy" in result:
            print(f"\nDifferential Privacy: {json.dumps(result['differential_privacy'], indent=2)}")
    
    return result is not None


async def test_batch_benchmarks():
    """Test: Run benchmarks for multiple datasets and models."""
    print("\n" + "=" * 70)
    print("Test 2: Batch Benchmark Execution")
    print("=" * 70)
    
    runner = BenchmarkMCPRunner()
    
    # Test with a subset of datasets and models
    datasets = ["abalone", "Bean", "faults"]
    models = ["aim", "arf", "ctgan", "cart"]
    
    print(f"Running {len(datasets)} datasets × {len(models)} models = {len(datasets) * len(models)} benchmarks\n")
    
    results = await runner.run_batch(datasets, models, include_dp=True)
    
    # Print summary
    print("\n" + "-" * 70)
    print("Batch Results Summary")
    print("-" * 70)
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"Completed: {successful}/{total}")
    
    for key, result in sorted(results.items()):
        if result:
            quality = result.get("fidelity", {}).get("overall_fidelity", "N/A")
            print(f"  ✓ {key}: quality={quality:.3f}" if isinstance(quality, float) else f"  ✓ {key}: quality={quality}")
        else:
            print(f"  ✗ {key}: failed")
    
    return successful == total


async def test_custom_paths():
    """Test: Show how to manually construct and use paths."""
    print("\n" + "=" * 70)
    print("Test 3: Manual Path Construction")
    print("=" * 70)
    
    from mcp_server.tools.benchmark_tools import BenchmarkTools
    
    # Manually construct paths using the template patterns
    dataset_name = "Bean"
    synthetic_model = "ctgan"
    
    original_path = f"./dataset/input_data/{dataset_name}.csv"
    synthetic_path = f"./dataset/synth_data/spark/{dataset_name}/{dataset_name}_synthetic_{synthetic_model}_1000.csv"
    evaluation_path = f"./dataset/synth_data/spark/{dataset_name}/test_data.csv"
    
    print(f"\nManually constructed paths:")
    print(f"  original_path: {original_path}")
    print(f"  synthetic_path: {synthetic_path}")
    print(f"  evaluation_path: {evaluation_path}")
    
    # Check if paths exist
    paths_exist = all(Path(p).exists() for p in [original_path, synthetic_path])
    eval_exists = Path(evaluation_path).exists()
    
    print(f"\n  ✓ Original & Synthetic exist: {paths_exist}")
    print(f"  {'✓' if eval_exists else '⚠'} Evaluation exists: {eval_exists}")
    
    if not paths_exist:
        print("  Skipping execution (files not found)")
        return False
    
    benchmark_tools = BenchmarkTools()
    
    args = {
        "original_path": original_path,
        "synthetic_path": synthetic_path,
    }
    
    if eval_exists:
        args["evaluation_path"] = evaluation_path
    
    print(f"\nExecuting benchmark with these paths...")
    result = await benchmark_tools.execute_tool("benchmark_compare", args)
    
    if result:
        print("✓ Benchmark completed")
        utility = result.get("utility", {})
        print(f"  Overall Utility: {utility.get('overall_utility', 'N/A')}")
        return True
    
    return False


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Synthony MCP Benchmark Runner - Path Template Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("Single Benchmark", test_single_benchmark),
        ("Batch Benchmarks", test_batch_benchmarks),
        ("Custom Paths", test_custom_paths),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n✗ {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total - passed}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
