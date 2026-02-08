#!/usr/bin/env python3
"""
Test Synthony API with real datasets from dataset/input_data/.

Tests the API with actual CSV files and saves detailed results.
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
DATA_DIR = Path(__file__).parent.parent / "dataset" / "input_data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "api_test_results"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def wait_for_server(max_attempts: int = 30, delay: float = 1.0):
    """Wait for server to be ready."""
    print("Waiting for API server to start...")

    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server is ready after {attempt + 1} attempts")
                return True
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(delay)
        print(f"  Attempt {attempt + 1}/{max_attempts}...", end="\r")

    return False


def test_single_dataset(csv_path: Path, method: str = "rule_based"):
    """Test API with a single dataset."""
    dataset_name = csv_path.stem
    print(f"\n{'─' * 80}")
    print(f"Testing: {dataset_name}")
    print(f"{'─' * 80}")

    try:
        # Analyze and recommend
        with open(csv_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/analyze-and-recommend",
                params={
                    "dataset_id": dataset_name,
                    "method": method,
                    "top_n": 3,
                },
                files={"file": f},
                timeout=60,
            )

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()

        # Display results
        analysis = result["analysis"]
        recommendation = result["recommendation"]

        profile = analysis["dataset_profile"]
        print(f"\n📊 Dataset Profile:")
        print(f"  Size: {profile['row_count']:,} rows × {profile['column_count']} columns")

        stress_factors = profile["stress_factors"]
        active_stress = [k for k, v in stress_factors.items() if v]
        if active_stress:
            print(f"  ⚠️  Active stress factors: {', '.join(active_stress)}")
        else:
            print(f"  ✓ No stress factors detected (easy dataset)")

        col_analysis = analysis["column_analysis"]
        print(f"\n📈 Column Analysis:")
        print(f"  Max difficulty: {col_analysis['max_column_difficulty']}/4")
        if col_analysis["difficult_columns"]:
            print(f"  Difficult columns ({len(col_analysis['difficult_columns'])}): {', '.join(col_analysis['difficult_columns'][:3])}")

        rec_model = recommendation["recommended_model"]
        print(f"\n🎯 Recommendation ({recommendation['method']} mode):")
        print(f"  ✓ Model: {rec_model['model_name']}")
        print(f"  ✓ Confidence: {rec_model['confidence_score']:.2f}")
        print(f"  ✓ Type: {rec_model['model_info']['type']}")

        if rec_model["reasoning"]:
            print(f"\n  Reasoning:")
            for reason in rec_model["reasoning"][:3]:
                print(f"    {reason}")

        if rec_model["warnings"]:
            print(f"\n  Warnings:")
            for warning in rec_model["warnings"][:2]:
                print(f"    {warning}")

        alternatives = [m["model_name"] for m in recommendation["alternative_models"]]
        print(f"\n  Alternatives: {', '.join(alternatives)}")

        excluded = recommendation.get("excluded_models", {})
        if excluded:
            print(f"  Excluded: {len(excluded)} models")
            for model, reason in list(excluded.items())[:2]:
                print(f"    • {model}: {reason}")

        return result

    except requests.exceptions.Timeout:
        print(f"❌ Timeout - dataset might be too large or server is overloaded")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_result(dataset_name: str, result: dict, output_dir: Path):
    """Save individual result to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{dataset_name}_api_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  💾 Saved to: {output_file.name}")


def create_comparison_report(results: dict, output_dir: Path):
    """Create a comparison report across all datasets."""
    print_section("Creating Comparison Report")

    comparison = []

    for dataset_name, result in results.items():
        if result is None:
            continue

        analysis = result["analysis"]
        recommendation = result["recommendation"]

        profile = analysis["dataset_profile"]
        col_analysis = analysis["column_analysis"]
        rec_model = recommendation["recommended_model"]

        active_stress = [k for k, v in profile["stress_factors"].items() if v]

        comparison.append({
            "dataset": dataset_name,
            "rows": profile["row_count"],
            "columns": profile["column_count"],
            "max_difficulty": col_analysis["max_column_difficulty"],
            "stress_factors": active_stress,
            "recommended_model": rec_model["model_name"],
            "confidence": rec_model["confidence_score"],
            "model_type": rec_model["model_info"]["type"],
        })

    # Sort by difficulty
    comparison.sort(key=lambda x: x["max_difficulty"], reverse=True)

    # Print comparison table
    print(f"\n{'Dataset':<20} {'Rows':>8} {'Cols':>5} {'Diff':>5} {'Model':<15} {'Conf':>5} {'Stress Factors'}")
    print("─" * 100)

    for item in comparison:
        stress = ", ".join(item["stress_factors"]) if item["stress_factors"] else "None"
        if len(stress) > 30:
            stress = stress[:27] + "..."

        print(
            f"{item['dataset']:<20} "
            f"{item['rows']:>8,} "
            f"{item['columns']:>5} "
            f"{item['max_difficulty']:>5}/4 "
            f"{item['recommended_model']:<15} "
            f"{item['confidence']:>5.2f} "
            f"{stress}"
        )

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "comparison_report.json"

    with open(report_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✓ Comparison report saved to: {report_file}")

    # Print summary statistics
    print_section("Summary Statistics")

    model_counts = {}
    for item in comparison:
        model = item["recommended_model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    print(f"\nTotal datasets tested: {len(comparison)}")
    print(f"\nModel recommendations breakdown:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {model}: {count} datasets")

    difficulty_dist = {}
    for item in comparison:
        diff = item["max_difficulty"]
        difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

    print(f"\nDifficulty distribution:")
    for diff in sorted(difficulty_dist.keys(), reverse=True):
        count = difficulty_dist[diff]
        print(f"  • Difficulty {diff}/4: {count} datasets")

    # Identify hardest datasets
    hardest = [item for item in comparison if item["max_difficulty"] >= 3]
    if hardest:
        print(f"\nHardest datasets (difficulty ≥3):")
        for item in hardest[:5]:
            print(f"  • {item['dataset']}: Difficulty {item['max_difficulty']}/4, recommended {item['recommended_model']}")


def main():
    """Run comprehensive API tests with real datasets."""
    print_section("Synthony API Test Suite - Real Datasets")

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in: {DATA_DIR}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} datasets to test")
    print(f"Output directory: {OUTPUT_DIR}")

    # Wait for server
    if not wait_for_server():
        print(f"\n❌ Cannot connect to server at {BASE_URL}")
        print("Please start the server with: synthony-api")
        sys.exit(1)

    # Test server health
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    print(f"\n✓ Server version: {health['version']}")
    print(f"✓ Models available: {health['models_count']}")
    print(f"✓ LLM mode: {'enabled' if health['llm_available'] else 'disabled'}")

    # Determine method based on LLM availability
    method = "hybrid" if health["llm_available"] else "rule_based"
    print(f"\nUsing recommendation method: {method}")

    # Test each dataset
    print_section(f"Testing {len(csv_files)} Datasets")

    results = {}
    successful = 0
    failed = 0

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}]", end=" ")

        result = test_single_dataset(csv_file, method=method)

        if result:
            results[csv_file.stem] = result
            save_result(csv_file.stem, result, OUTPUT_DIR)
            successful += 1
        else:
            failed += 1

        # Small delay between requests
        time.sleep(0.5)

    # Create comparison report
    if results:
        create_comparison_report(results, OUTPUT_DIR)

    # Final summary
    print_section("Test Complete")
    print(f"\n✅ Successful: {successful}/{len(csv_files)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(csv_files)}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"\nYou can now:")
    print(f"  1. Review individual results in: {OUTPUT_DIR}")
    print(f"  2. Check comparison report: {OUTPUT_DIR}/comparison_report.json")
    print(f"  3. View interactive docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
