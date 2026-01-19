#!/usr/bin/env python3
"""
Batch analysis script for all datasets in dataset/input_data/.

Runs StochasticDataAnalyzer and ColumnAnalyzer on all CSV files,
generating individual reports and a comprehensive comparison summary.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthony import ColumnAnalyzer, StochasticDataAnalyzer


class DatasetBatchAnalyzer:
    """Batch analyzer for multiple datasets."""

    def __init__(self, input_dir: Path, output_dir: Path):
        """Initialize batch analyzer.

        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save analysis results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = StochasticDataAnalyzer()
        self.column_analyzer = ColumnAnalyzer()

        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []

    def analyze_all(self) -> None:
        """Analyze all CSV files in input directory."""
        csv_files = sorted(self.input_dir.glob("*.csv"))

        if not csv_files:
            print(f"⚠️  No CSV files found in {self.input_dir}")
            return

        print("=" * 80)
        print(f"BATCH DATASET ANALYSIS - {len(csv_files)} datasets found")
        print("=" * 80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

        for idx, csv_path in enumerate(csv_files, 1):
            print(f"\n[{idx}/{len(csv_files)}] Analyzing: {csv_path.name}")
            print("-" * 80)

            try:
                result = self._analyze_dataset(csv_path)
                self.results.append(result)
                print(f"✓ Successfully analyzed {csv_path.name}")

            except Exception as e:
                error_info = {
                    "dataset": csv_path.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                self.errors.append(error_info)
                print(f"✗ Error analyzing {csv_path.name}: {e}")
                continue

        print("\n" + "=" * 80)
        print("BATCH ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"✓ Successful: {len(self.results)}/{len(csv_files)}")
        print(f"✗ Failed: {len(self.errors)}/{len(csv_files)}")

    def _analyze_dataset(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary containing analysis results and metadata
        """
        start_time = time.time()

        # Load dataset
        print(f"  Loading data... ", end="", flush=True)
        df = pd.read_csv(csv_path)
        print(f"✓ ({df.shape[0]:,} rows × {df.shape[1]} columns)")

        # Dataset-level analysis
        print(f"  Running dataset analysis... ", end="", flush=True)
        dataset_profile = self.analyzer.analyze(df)
        print("✓")

        # Column-level analysis
        print(f"  Running column analysis... ", end="", flush=True)
        column_analysis = self.column_analyzer.analyze(df, dataset_profile)
        print("✓")

        # Calculate processing time
        elapsed_time = time.time() - start_time

        # Save individual report
        output_path = self.output_dir / f"{csv_path.stem}_analysis.json"
        print(f"  Saving results to {output_path.name}... ", end="", flush=True)

        combined_output = {
            "metadata": {
                "dataset_name": csv_path.name,
                "file_size_mb": csv_path.stat().st_size / (1024 * 1024),
                "analysis_time_seconds": round(elapsed_time, 2),
                "analyzed_at": datetime.now().isoformat(),
            },
            "dataset_profile": dataset_profile.model_dump(),
            "column_analysis": column_analysis.model_dump(),
        }

        with open(output_path, "w") as f:
            json.dump(combined_output, f, indent=2, default=str)
        print("✓")

        # Print summary
        self._print_summary(csv_path.name, dataset_profile, column_analysis)

        # Return summary for comparison report
        return {
            "dataset_name": csv_path.name,
            "rows": dataset_profile.row_count,
            "columns": dataset_profile.column_count,
            "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2),
            "analysis_time_seconds": round(elapsed_time, 2),
            "stress_factors": dataset_profile.stress_factors.model_dump(),
            "max_column_difficulty": column_analysis.max_column_difficulty,
            "difficult_columns_count": len(column_analysis.difficult_columns),
            "difficult_columns": column_analysis.difficult_columns,
            "stress_factor_summary": column_analysis.stress_factor_summary,
        }

    def _print_summary(
        self, dataset_name: str, profile: Any, column_analysis: Any
    ) -> None:
        """Print analysis summary for a dataset.

        Args:
            dataset_name: Name of the dataset
            profile: DatasetProfile
            column_analysis: ColumnAnalysisResult
        """
        print(f"\n  📊 Summary for {dataset_name}:")
        print(f"     Rows: {profile.row_count:,} | Columns: {profile.column_count}")

        # Stress factors
        active_stress = [
            name
            for name, value in profile.stress_factors.model_dump().items()
            if value
        ]
        if active_stress:
            print(f"     ⚠️  Active stress factors: {', '.join(active_stress)}")

        # Column difficulty
        print(f"     🎯 Max column difficulty: {column_analysis.max_column_difficulty}/4")
        if column_analysis.difficult_columns:
            print(
                f"     🔴 Difficult columns ({len(column_analysis.difficult_columns)}): {', '.join(column_analysis.difficult_columns[:3])}{'...' if len(column_analysis.difficult_columns) > 3 else ''}"
            )

    def generate_comparison_report(self) -> None:
        """Generate comparison report across all analyzed datasets."""
        if not self.results:
            print("⚠️  No results to compare")
            return

        print("\n" + "=" * 80)
        print("CROSS-DATASET COMPARISON REPORT")
        print("=" * 80)

        # Sort by max difficulty (descending)
        sorted_results = sorted(
            self.results, key=lambda x: x["max_column_difficulty"], reverse=True
        )

        # Summary table
        print(f"\n{'Dataset':<25} {'Rows':>8} {'Cols':>5} {'Difficulty':>11} {'Hard Cols':>10}")
        print("-" * 80)

        for result in sorted_results:
            diff_icon = "🔴" if result["max_column_difficulty"] >= 3 else "🟢"
            print(
                f"{result['dataset_name']:<25} {result['rows']:>8,} {result['columns']:>5} "
                f"{diff_icon} {result['max_column_difficulty']}/4      {result['difficult_columns_count']:>10}"
            )

        # Stress factor distribution
        print("\n=== STRESS FACTOR DISTRIBUTION ===")
        stress_counts = {}
        for result in self.results:
            for factor, active in result["stress_factors"].items():
                if active:
                    stress_counts[factor] = stress_counts.get(factor, 0) + 1

        for factor, count in sorted(stress_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(self.results)) * 100
            print(f"  {factor:<30}: {count}/{len(self.results)} datasets ({pct:.1f}%)")

        # Most difficult datasets
        print("\n=== MOST DIFFICULT DATASETS (by column difficulty) ===")
        top_difficult = sorted_results[:5]
        for idx, result in enumerate(top_difficult, 1):
            print(f"\n{idx}. {result['dataset_name']} (Difficulty: {result['max_column_difficulty']}/4)")
            print(f"   Size: {result['rows']:,} rows × {result['columns']} columns")
            if result["difficult_columns"]:
                print(f"   Difficult columns: {', '.join(result['difficult_columns'])}")
            active_stress = [k for k, v in result["stress_factors"].items() if v]
            if active_stress:
                print(f"   Active stress factors: {', '.join(active_stress)}")

        # Performance statistics
        print("\n=== ANALYSIS PERFORMANCE ===")
        total_time = sum(r["analysis_time_seconds"] for r in self.results)
        total_rows = sum(r["rows"] for r in self.results)
        avg_time = total_time / len(self.results)

        print(f"  Total analysis time: {total_time:.2f} seconds")
        print(f"  Average time per dataset: {avg_time:.2f} seconds")
        print(f"  Total rows processed: {total_rows:,}")
        print(
            f"  Processing speed: {total_rows / total_time:,.0f} rows/second (avg)"
        )

        # Save comparison report
        comparison_path = self.output_dir / "comparison_report.json"
        comparison_data = {
            "generated_at": datetime.now().isoformat(),
            "total_datasets": len(self.results),
            "failed_datasets": len(self.errors),
            "summary_statistics": {
                "total_rows": total_rows,
                "total_analysis_time_seconds": round(total_time, 2),
                "average_analysis_time_seconds": round(avg_time, 2),
                "processing_speed_rows_per_second": round(total_rows / total_time, 2),
            },
            "stress_factor_distribution": stress_counts,
            "datasets": sorted_results,
            "errors": self.errors,
        }

        with open(comparison_path, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)

        print(f"\n✓ Comparison report saved to: {comparison_path}")

        # Error summary
        if self.errors:
            print("\n=== ERRORS ===")
            for error in self.errors:
                print(f"  ✗ {error['dataset']}: {error['error_type']} - {error['error']}")


def main():
    """Main entry point."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "dataset" / "input_data"
    output_dir = project_root / "output" / "analysis_results"

    # Run batch analysis
    batch_analyzer = DatasetBatchAnalyzer(input_dir, output_dir)
    batch_analyzer.analyze_all()
    batch_analyzer.generate_comparison_report()

    print("\n" + "=" * 80)
    print("All results saved to:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
