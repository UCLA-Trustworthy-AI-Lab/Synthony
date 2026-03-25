"""
Run Synthony recommendation engine on all datasets in data/input_data/.

For each CSV dataset:
1. Profile with StochasticDataAnalyzer
2. Analyze columns with ColumnAnalyzer
3. Generate rule-based recommendation with ModelRecommendationEngine
4. Save results to output/ folder as JSON

Usage:
    python run_recommendations.py                    # No focus (original behavior)
    python run_recommendations.py --focus privacy    # Single focus
    python run_recommendations.py --focus all        # All 3 focuses sequentially
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine


FOCUS_NAMES = ["privacy", "fidelity", "latency"]


def run_single(engine, analyzer, column_analyzer, csv_files, output_dir, focus=None):
    """Run recommendations for all datasets with an optional focus.

    Returns list of summary dicts.
    """
    suffix = f"_{focus}" if focus else ""
    summary_results = []

    for csv_path in csv_files:
        dataset_name = csv_path.stem
        print(f"{'='*60}")
        label = f"{dataset_name}" + (f" [focus={focus}]" if focus else "")
        print(f"Processing: {label}")
        print(f"{'='*60}")

        try:
            # Step 1: Load data
            df = pd.read_csv(csv_path)
            print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

            # Step 2: Profile dataset
            profile = analyzer.analyze(df)
            stress = profile.stress_factors.model_dump()
            active = [k for k, v in stress.items() if v]
            print(f"  Stress factors: {active if active else 'None detected'}")

            # Step 3: Column analysis
            col_analysis = column_analyzer.analyze(df, profile)
            print(f"  Max column difficulty: {col_analysis.max_column_difficulty}")
            if col_analysis.difficult_columns:
                print(f"  Difficult columns: {col_analysis.difficult_columns}")

            # Step 4: Generate recommendation
            recommendation = engine.recommend(
                dataset_profile=profile,
                column_analysis=col_analysis,
                constraints={},
                method="rule_based",
                top_n=3,
                focus=focus,
            )

            # Print summary
            primary = recommendation.recommended_model
            print(f"\n  >> Recommended: {primary.model_name} "
                  f"(confidence: {primary.confidence_score:.2%})")
            alts = [a.model_name for a in recommendation.alternative_models]
            print(f"  >> Alternatives: {alts}")
            print(f"  >> Method: {recommendation.method}")
            if recommendation.difficulty_summary.get("is_hard_problem"):
                print(f"  >> HARD PROBLEM detected!")
            print()

            # Step 5: Save individual result
            profile_dict = profile.model_dump()
            if profile_dict.get("correlation") and "correlation_matrix" in profile_dict["correlation"]:
                profile_dict["correlation"]["correlation_matrix"] = None

            result_data = {
                "dataset_name": dataset_name,
                "dataset_path": str(csv_path),
                "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "focus": focus,
                "profile": profile_dict,
                "column_analysis": col_analysis.model_dump(),
                "recommendation": recommendation.model_dump(),
            }

            output_path = output_dir / f"{dataset_name}{suffix}_recommendation.json"
            with open(output_path, "w") as f:
                json.dump(result_data, f, indent=2, default=str)
            print(f"  Saved: {output_path}")

            # Add to summary
            summary_results.append({
                "dataset": dataset_name,
                "focus": focus,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "stress_factors": active,
                "max_difficulty": col_analysis.max_column_difficulty,
                "is_hard_problem": recommendation.difficulty_summary.get("is_hard_problem", False),
                "recommended_model": primary.model_name,
                "confidence": round(primary.confidence_score, 4),
                "alternatives": alts,
                "method": recommendation.method,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            summary_results.append({
                "dataset": dataset_name,
                "focus": focus,
                "error": str(e),
            })

        print()

    return summary_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Synthony recommendations on all datasets."
    )
    parser.add_argument(
        "--focus",
        choices=FOCUS_NAMES + ["all"],
        default=None,
        help="Focus profile to use. 'all' runs privacy, fidelity, and latency sequentially.",
    )
    args = parser.parse_args()

    data_dir = Path("data/input_data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Initialize components
    analyzer = StochasticDataAnalyzer()
    column_analyzer = ColumnAnalyzer()
    engine = ModelRecommendationEngine()

    # Get all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} datasets to process\n")

    # Determine which focuses to run
    if args.focus == "all":
        focuses = FOCUS_NAMES
    elif args.focus:
        focuses = [args.focus]
    else:
        focuses = [None]  # No focus (original behavior)

    all_summaries = []

    for focus in focuses:
        if focus:
            print(f"\n{'#'*60}")
            print(f"# FOCUS: {focus}")
            print(f"{'#'*60}\n")
        results = run_single(engine, analyzer, column_analyzer, csv_files, output_dir, focus)
        all_summaries.extend(results)

    # Save summary
    if args.focus:
        summary_path = output_dir / "recommendation_summary_by_focus.json"
    else:
        summary_path = output_dir / "recommendation_summary.json"

    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    # Print summary table
    header_focus = "Focus" if args.focus else ""
    print(f"{'Dataset':<25} {'Focus':<10} {'Rows':>7} {'Recommended':<15} {'Confidence':>10} {'Hard?'}")
    print("-" * 80)
    for r in all_summaries:
        if "error" in r:
            print(f"{r['dataset']:<25} {(r.get('focus') or ''):<10} {'ERROR':>7}")
        else:
            hard = "YES" if r.get("is_hard_problem") else ""
            print(f"{r['dataset']:<25} {(r.get('focus') or ''):<10} {r['rows']:>7} "
                  f"{r['recommended_model']:<15} {r['confidence']:>9.2%} {hard}")


if __name__ == "__main__":
    main()
