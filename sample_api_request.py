#!/usr/bin/env python3
"""
Sample script demonstrating Synthony API usage.

Usage:
    python sample_api_request.py                      # Use default sample data
    python sample_api_request.py path/to/your.csv    # Use your own CSV file
    python sample_api_request.py --method llm        # Use LLM recommendation
    python sample_api_request.py --help              # Show help

Requirements:
    pip install requests

Make sure the API server is running:
    python start_api.py
    # or
    synthony-api
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests")
    sys.exit(1)


BASE_URL = "http://localhost:8000"


def analyze_and_recommend(
    file_path: str,
    method: str = "hybrid",
    cpu_only: bool = True,
    strict_dp: bool = False,
    top_n: int = 3,
    dataset_id: str = None,
):
    """
    Analyze a CSV file and get model recommendations.

    Args:
        file_path: Path to CSV file
        method: Recommendation method ('rule_based', 'llm', 'hybrid')
        cpu_only: Only recommend CPU-compatible models
        strict_dp: Only recommend differential privacy models
        top_n: Number of alternative recommendations
        dataset_id: Optional identifier for the dataset

    Returns:
        API response as dictionary
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix.lower() == ".csv":
        raise ValueError("Only CSV files are supported")

    # Prepare request
    url = f"{BASE_URL}/analyze-and-recommend"
    params = {
        "method": method,
        "cpu_only": cpu_only,
        "strict_dp": strict_dp,
        "top_n": top_n,
    }

    if dataset_id:
        params["dataset_id"] = dataset_id

    # Send request
    print(f"Uploading: {file_path.name}")
    print(f"Parameters: method={method}, cpu_only={cpu_only}, strict_dp={strict_dp}")
    print("-" * 60)

    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": (file_path.name, f, "text/csv")}
        )

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    return response.json()


def print_results(result: dict):
    """Pretty print the analysis and recommendation results."""

    # Analysis Summary
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    profile = result["analysis"]["dataset_profile"]
    col_analysis = result["analysis"]["column_analysis"]

    print(f"\nDataset: {result.get('dataset_id', 'unknown')}")
    print(f"Size: {profile['row_count']:,} rows × {profile['column_count']} columns")
    print(f"Max Column Difficulty: {col_analysis['max_column_difficulty']}/4")

    # Stress Factors
    print("\nStress Factors:")
    stress = profile["stress_factors"]
    for factor, active in stress.items():
        status = "🔴 YES" if active else "🟢 no"
        print(f"  {factor}: {status}")

    # Difficult Columns
    if col_analysis.get("difficult_columns"):
        print(f"\nDifficult Columns ({len(col_analysis['difficult_columns'])}):")
        for col in col_analysis["difficult_columns"][:5]:
            print(f"  - {col}")
        if len(col_analysis["difficult_columns"]) > 5:
            print(f"  ... and {len(col_analysis['difficult_columns']) - 5} more")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    rec = result["recommendation"]
    primary = rec["recommended_model"]

    print(f"\nMethod: {rec['method']}")
    print(f"\n🏆 Recommended Model: {primary['model_name']}")
    print(f"   Confidence: {primary['confidence_score']:.0%}")
    print(f"   Type: {primary['model_info']['type']}")
    print(f"   Training Speed: {primary['model_info']['performance']['training_speed']}")

    # Reasoning
    print("\n   Reasoning:")
    for reason in primary["reasoning"][:3]:
        print(f"   {reason}")

    # Warnings
    if primary.get("warnings"):
        print("\n   Warnings:")
        for warning in primary["warnings"][:2]:
            print(f"   {warning}")

    # Alternatives
    if rec.get("alternative_models"):
        print("\n📋 Alternative Models:")
        for i, alt in enumerate(rec["alternative_models"], 1):
            print(f"   {i}. {alt['model_name']} ({alt['confidence_score']:.0%})")

    # Excluded Models
    if rec.get("excluded_models"):
        excluded_count = len(rec["excluded_models"])
        print(f"\n⛔ Excluded Models: {excluded_count}")
        for name, reason in list(rec["excluded_models"].items())[:3]:
            print(f"   - {name}: {reason}")

    # LLM Reasoning (if available)
    if rec.get("llm_reasoning"):
        print("\n🤖 LLM Reasoning:")
        print(f"   {rec['llm_reasoning'][:300]}...")

    print("\n" + "=" * 60)


def check_server():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is running")
            print(f"  LLM available: {data['llm_available']}")
            print(f"  Models loaded: {data['models_count']}")
            return True
    except requests.exceptions.ConnectionError:
        pass

    print("✗ Server is not running")
    print(f"  Start with: python start_api.py")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CSV data and get synthetic data model recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sample_api_request.py dataset/input_data/Titanic.csv
  python sample_api_request.py data.csv --method llm
  python sample_api_request.py data.csv --cpu-only false
  python sample_api_request.py data.csv --strict-dp
        """
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="dataset/input_data/Titanic.csv",
        help="Path to CSV file (default: dataset/input_data/Titanic.csv)"
    )

    parser.add_argument(
        "--method", "-m",
        choices=["rule_based", "llm", "hybrid"],
        default="hybrid",
        help="Recommendation method (default: hybrid)"
    )

    parser.add_argument(
        "--cpu-only",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Only CPU-compatible models (default: true)"
    )

    parser.add_argument(
        "--strict-dp",
        action="store_true",
        help="Only differential privacy models"
    )

    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=3,
        help="Number of alternatives (default: 3)"
    )

    parser.add_argument(
        "--dataset-id",
        help="Optional dataset identifier"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted results"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Synthony - Model Recommendation API Client")
    print("=" * 60)
    print()

    # Check server
    if not check_server():
        sys.exit(1)

    print()

    # Run analysis
    try:
        result = analyze_and_recommend(
            file_path=args.file,
            method=args.method,
            cpu_only=args.cpu_only,
            strict_dp=args.strict_dp,
            top_n=args.top_n,
            dataset_id=args.dataset_id,
        )

        if result:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_results(result)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
