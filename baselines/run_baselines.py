"""Orchestrator: run all 4 baselines and produce comparison table.

Optionally includes Synthony engine results for side-by-side comparison.

Usage:
    python baselines/run_baselines.py
    python baselines/run_baselines.py --include-synthony
"""

import argparse
import json

import pandas as pd

from baselines.common import (
    OVERLAP_MODELS,
    OUTPUT_DIR,
    evaluate_baseline,
    find_csv,
    load_ground_truth,
)
from baselines.static_heuristic import run_all as run_heuristic
from baselines.vanilla_llm import run_all as run_llm
from baselines.random_search import run_expected_value, run_single_shot
from baselines.knn_selector import run_all as run_knn


def run_synthony(gt, split):
    """Run Synthony engine with optimized scale factors."""
    sf_path = OUTPUT_DIR / "best_scale_factors.json"
    if not sf_path.exists():
        print("WARNING: best_scale_factors.json not found. Skipping Synthony.")
        return {}

    with open(sf_path) as f:
        best_sf = json.load(f)

    # Import Synthony components
    from synthony.core.analyzer import StochasticDataAnalyzer
    from synthony.core.column_analyzer import ColumnAnalyzer
    from synthony.recommender.engine import ModelRecommendationEngine

    analyzer = StochasticDataAnalyzer()
    col_analyzer = ColumnAnalyzer()
    engine = ModelRecommendationEngine()

    # Cache profiles
    profiles = {}
    datasets = sorted({v["dataset"] for v in gt.values()})
    for ds_name in datasets:
        csv_path = find_csv(ds_name)
        df = pd.read_csv(csv_path)
        profile = analyzer.analyze(df)
        col_analysis = col_analyzer.analyze(df, profile)
        profiles[ds_name] = (profile, col_analysis)

    predictions = {}
    for key, entry in gt.items():
        ds_name = entry["dataset"]
        focus = entry["focus"]
        profile, col_analysis = profiles[ds_name]
        result = engine.recommend(
            dataset_profile=profile,
            column_analysis=col_analysis,
            constraints={"allowed_models": OVERLAP_MODELS},
            method="rule_based",
            top_n=len(OVERLAP_MODELS),
            scale_factors=best_sf[focus],
        )
        ranking = [result.recommended_model.model_name]
        for alt in result.alternative_models:
            ranking.append(alt.model_name)
        predictions[key] = ranking

    return predictions


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    header = (f"{'Method':<25} | {'Train Top-1':>10} | {'Train Top-3':>10} | "
              f"{'Train Spear':>11} | {'Train NDCG':>10} | "
              f"{'Test Top-1':>10} | {'Test Top-3':>10} | "
              f"{'Test Spear':>10} | {'Test NDCG':>10}")
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        tr = res.get("train", {})
        te = res.get("test", {})
        row = (f"{name:<25} | "
               f"{tr.get('top1', 0):.3f}      | "
               f"{tr.get('top3', 0):.3f}      | "
               f"{tr.get('spearman', 0):.3f}       | "
               f"{tr.get('ndcg', 0):.3f}      | "
               f"{te.get('top1', 0):.3f}      | "
               f"{te.get('top3', 0):.3f}      | "
               f"{te.get('spearman', 0):.3f}      | "
               f"{te.get('ndcg', 0):.3f}")
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Run all baselines")
    parser.add_argument("--include-synthony", action="store_true",
                        help="Include Synthony engine results")
    args = parser.parse_args()

    gt, split = load_ground_truth()
    print(f"Ground truth: {len(gt)} pairs  "
          f"(train={len(split['train'])}, test={len(split['test'])})\n")

    all_results = {}

    # Synthony (optional)
    if args.include_synthony:
        print("Running Synthony (optimized)...")
        synthony_preds = run_synthony(gt, split)
        if synthony_preds:
            all_results["Synthony (optimized)"] = evaluate_baseline(
                "Synthony", synthony_preds, gt, split)
        print()

    # Baseline 1: Static Heuristic
    print("Running Baseline 1: Static Heuristic...")
    heuristic_preds = run_heuristic(gt, split)
    all_results["Static Heuristic"] = evaluate_baseline(
        "Static Heuristic", heuristic_preds, gt, split)

    # Baseline 2: Vanilla LLM
    print("\nRunning Baseline 2: Vanilla LLM...")
    llm_preds = run_llm(gt, split)
    if llm_preds:
        all_results["Vanilla LLM"] = evaluate_baseline(
            "Vanilla LLM", llm_preds, gt, split)
    else:
        print("  (Skipped — no API key)")

    # Baseline 3: Random Search
    print("\nRunning Baseline 3: Random Search...")
    random_preds = run_single_shot(gt, seed=42)
    all_results["Random (seed=42)"] = evaluate_baseline(
        "Random (seed=42)", random_preds, gt, split)

    random_ev = run_expected_value(gt, split, n_trials=1000)
    all_results["Random (E[1000])"] = random_ev

    # Baseline 4: kNN
    print("\nRunning Baseline 4: kNN Selector...")
    knn_preds = run_knn(gt, split, k=3)
    all_results["kNN (k=3)"] = evaluate_baseline(
        "kNN (k=3)", knn_preds, gt, split)

    # Print comparison
    print(f"\n{'='*120}")
    print("BASELINE COMPARISON")
    print(f"{'='*120}\n")
    print_comparison_table(all_results)

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "baseline_comparison.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
