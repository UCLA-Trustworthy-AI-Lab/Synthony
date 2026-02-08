#!/usr/bin/env python3
"""
Example usage of the analysis results for model recommendation.

Demonstrates how to programmatically access and use the column-level
analysis results for making model selection decisions.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_analysis(dataset_name: str) -> Dict:
    """Load analysis results for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'abalone.csv')

    Returns:
        Dictionary containing full analysis results
    """
    analysis_dir = Path(__file__).parent.parent / "output" / "analysis_results"
    stem = dataset_name.replace(".csv", "")
    analysis_path = analysis_dir / f"{stem}_analysis.json"

    with open(analysis_path, "r") as f:
        return json.load(f)


def get_recommended_models(analysis: Dict) -> List[str]:
    """Extract recommended models from analysis.

    Args:
        analysis: Analysis results dictionary

    Returns:
        List of recommended model names (deduplicated)
    """
    recommendations = set()

    for col_profile in analysis["column_analysis"]["columns"].values():
        # Only consider difficult columns (≥3)
        if col_profile["difficulty"]["overall_difficulty"] >= 3:
            for rec in col_profile["recommended_model_types"]:
                # Extract model names from recommendation strings
                if "GReaT" in rec:
                    recommendations.add("GReaT")
                if "TabDDPM" in rec:
                    recommendations.add("TabDDPM")
                if "TabSyn" in rec:
                    recommendations.add("TabSyn")
                if "CTGAN" in rec:
                    recommendations.add("CTGAN")
                if "AutoDiff" in rec:
                    recommendations.add("AutoDiff")
                if "ARF" in rec:
                    recommendations.add("ARF")

    return sorted(recommendations)


def get_difficulty_breakdown(analysis: Dict) -> Dict[str, int]:
    """Get distribution of column difficulties.

    Args:
        analysis: Analysis results dictionary

    Returns:
        Dictionary mapping difficulty level to count
    """
    difficulty_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for col_profile in analysis["column_analysis"]["columns"].values():
        difficulty = col_profile["difficulty"]["overall_difficulty"]
        difficulty_counts[difficulty] += 1

    return difficulty_counts


def get_stress_factor_columns(analysis: Dict) -> Dict[str, List[str]]:
    """Get columns grouped by stress factor.

    Args:
        analysis: Analysis results dictionary

    Returns:
        Dictionary mapping stress factor to list of affected columns
    """
    stress_columns = {
        "severe_skew": [],
        "high_cardinality": [],
        "zipfian": [],
    }

    for col_name, col_profile in analysis["column_analysis"]["columns"].items():
        factors = col_profile["stress_factors"]

        if factors["severe_skew"]:
            stress_columns["severe_skew"].append(col_name)
        if factors["high_cardinality"]:
            stress_columns["high_cardinality"].append(col_name)
        if factors["zipfian"]:
            stress_columns["zipfian"].append(col_name)

    return stress_columns


def print_analysis_summary(dataset_name: str) -> None:
    """Print human-readable summary of analysis.

    Args:
        dataset_name: Name of the dataset
    """
    analysis = load_analysis(dataset_name)

    print("=" * 80)
    print(f"ANALYSIS SUMMARY: {dataset_name}")
    print("=" * 80)

    # Basic info
    metadata = analysis["metadata"]
    print(f"\nDataset: {metadata['dataset_name']}")
    print(f"Size: {metadata['file_size_mb']:.2f} MB")
    print(f"Analysis time: {metadata['analysis_time_seconds']:.2f}s")

    # Dataset-level summary
    profile = analysis["dataset_profile"]
    print(f"\nRows: {profile['row_count']:,}")
    print(f"Columns: {profile['column_count']}")

    # Active stress factors
    stress_factors = profile["stress_factors"]
    active_stress = [name for name, value in stress_factors.items() if value]
    print(f"\nActive Stress Factors: {', '.join(active_stress) if active_stress else 'None'}")

    # Column difficulty distribution
    difficulty_dist = get_difficulty_breakdown(analysis)
    print("\nColumn Difficulty Distribution:")
    for level, count in sorted(difficulty_dist.items()):
        if count > 0:
            icon = "🔴" if level >= 3 else "🟢"
            print(f"  {icon} Difficulty {level}: {count} columns")

    # Stress factor breakdown
    stress_columns = get_stress_factor_columns(analysis)
    print("\nStress Factor Breakdown:")
    for factor, columns in stress_columns.items():
        if columns:
            print(f"  • {factor}: {len(columns)} columns - {', '.join(columns[:3])}{'...' if len(columns) > 3 else ''}")

    # Recommended models
    models = get_recommended_models(analysis)
    print(f"\nRecommended Models: {', '.join(models) if models else 'Any model'}")

    # Most difficult columns
    column_analysis = analysis["column_analysis"]
    if column_analysis["difficult_columns"]:
        print(f"\nMost Difficult Columns ({len(column_analysis['difficult_columns'])}):")
        for col_name in column_analysis["difficult_columns"][:5]:
            col = column_analysis["columns"][col_name]
            diff = col["difficulty"]
            print(f"  • {col_name}: Difficulty {diff['overall_difficulty']}/4")
            print(f"    - Skew: {diff['skew_difficulty']}, Cardinality: {diff['cardinality_difficulty']}, Zipfian: {diff['zipfian_difficulty']}")

    print("\n" + "=" * 80)


def compare_datasets(dataset_names: List[str]) -> None:
    """Compare multiple datasets.

    Args:
        dataset_names: List of dataset names to compare
    """
    print("=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)

    print(f"\n{'Dataset':<25} {'Rows':>8} {'Cols':>5} {'Difficulty':>11} {'Models':<30}")
    print("-" * 80)

    for name in dataset_names:
        try:
            analysis = load_analysis(name)
            profile = analysis["dataset_profile"]
            col_analysis = analysis["column_analysis"]
            models = get_recommended_models(analysis)

            print(
                f"{name:<25} {profile['row_count']:>8,} {profile['column_count']:>5} "
                f"{'🔴' if col_analysis['max_column_difficulty'] >= 3 else '🟢'} {col_analysis['max_column_difficulty']}/4      "
                f"{', '.join(models[:2]):<30}"
            )
        except Exception as e:
            print(f"{name:<25} Error: {e}")

    print("\n" + "=" * 80)


def main():
    """Main demonstration."""
    # Example 1: Single dataset analysis
    print("\n=== EXAMPLE 1: Single Dataset Analysis ===")
    print_analysis_summary("abalone.csv")

    # Example 2: Compare multiple datasets
    print("\n=== EXAMPLE 2: Compare Multiple Datasets ===")
    compare_datasets([
        "abalone.csv",
        "insurance.csv",
        "Titanic.csv",
        "Bean.csv",
        "HTRU2.csv",
    ])

    # Example 3: Find datasets requiring specific models
    print("\n=== EXAMPLE 3: Find Datasets Requiring GReaT ===")
    analysis_dir = Path(__file__).parent.parent / "output" / "analysis_results"
    datasets_needing_great = []

    for analysis_file in analysis_dir.glob("*_analysis.json"):
        if analysis_file.name == "comparison_report.json":
            continue

        with open(analysis_file, "r") as f:
            analysis = json.load(f)

        models = get_recommended_models(analysis)
        if "GReaT" in models:
            datasets_needing_great.append(analysis["metadata"]["dataset_name"])

    print(f"\nDatasets requiring GReaT (LLM-based model):")
    for dataset in sorted(datasets_needing_great):
        print(f"  • {dataset}")

    # Example 4: Find easiest datasets
    print("\n=== EXAMPLE 4: Easiest Datasets (Difficulty ≤2) ===")
    easy_datasets = []

    for analysis_file in analysis_dir.glob("*_analysis.json"):
        if analysis_file.name == "comparison_report.json":
            continue

        with open(analysis_file, "r") as f:
            analysis = json.load(f)

        max_diff = analysis["column_analysis"]["max_column_difficulty"]
        if max_diff <= 2:
            easy_datasets.append(
                (analysis["metadata"]["dataset_name"], max_diff)
            )

    if easy_datasets:
        for dataset, diff in sorted(easy_datasets, key=lambda x: x[1]):
            print(f"  • {dataset}: Difficulty {diff}/4")
    else:
        print("  No datasets with difficulty ≤2 found")


if __name__ == "__main__":
    main()
