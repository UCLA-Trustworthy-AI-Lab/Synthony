"""
Build ground truth rankings from evaluation data for focus-based optimization.

Reads the Excel evaluation results, filters to the 6 overlapping models,
and produces ground truth rankings for privacy, fidelity, and latency focuses.
Also creates a 70/30 train/test split for optimization.

Usage:
    python scripts/build_ground_truth.py
"""

import json
import random
from pathlib import Path

import pandas as pd

# 6 models that overlap between model_capabilities.json and evaluation data
OVERLAP_MODELS = ["AIM", "AutoDiff", "DPCART", "TabDDPM", "TVAE", "ARF"]

# Focus → (metric_column, ascending?)
# Privacy:  "Proportion Closer to Real" — smaller is better (less identifiable)
# Fidelity: "Column Shape Score (Average)" — larger is better
# Latency:  "Latency (Seconds)" — smaller is better
FOCUS_METRICS = {
    "privacy":  ("Proportion Closer to Real", True),    # ascending = best first
    "fidelity": ("Column Shape Score (Average)", False), # descending = best first
    "latency":  ("Latency (Seconds)", True),             # ascending = best first
}

EVAL_PATH = Path("evaluation_metrics/Table-Synthesizers Evaluation Results.xlsx")
OUTPUT_DIR = Path("output")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read evaluation data
    df = pd.read_excel(EVAL_PATH, sheet_name="T1_Eval_clean")
    print(f"Loaded {len(df)} rows from T1_Eval_clean")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Models:   {sorted(df['Model'].unique())}")

    # Filter to overlapping models only
    df = df[df["Model"].isin(OVERLAP_MODELS)].copy()
    print(f"\nAfter filtering to {OVERLAP_MODELS}: {len(df)} rows")

    datasets = sorted(df["Dataset"].unique())
    print(f"Datasets with data: {datasets}")

    # Build ground truth
    ground_truth = {}

    for dataset in datasets:
        ds_df = df[df["Dataset"] == dataset]

        for focus, (metric_col, ascending) in FOCUS_METRICS.items():
            key = f"{dataset}_{focus}"

            # Sort by metric (ascending=True means smallest first = best)
            sorted_df = ds_df.sort_values(metric_col, ascending=ascending)
            ranking = sorted_df["Model"].tolist()
            best_model = ranking[0]

            # Also store the metric values for inspection
            metric_values = {
                row["Model"]: round(row[metric_col], 6)
                for _, row in sorted_df.iterrows()
            }

            ground_truth[key] = {
                "dataset": dataset,
                "focus": focus,
                "metric": metric_col,
                "ascending": ascending,
                "best_model": best_model,
                "ranking": ranking,
                "metric_values": metric_values,
            }

    print(f"\nGround truth entries: {len(ground_truth)} (expected 21 = 7 datasets × 3 focuses)")

    # Print summary
    print(f"\n{'Key':<35} {'Best Model':<12} {'Metric Value'}")
    print("-" * 65)
    for key, entry in sorted(ground_truth.items()):
        best = entry["best_model"]
        val = entry["metric_values"][best]
        print(f"{key:<35} {best:<12} {val}")

    # Save ground truth
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nSaved: {gt_path}")

    # Create train/test split (70/30)
    keys = sorted(ground_truth.keys())
    random.seed(42)
    random.shuffle(keys)

    split_idx = int(len(keys) * 0.7)
    train_keys = sorted(keys[:split_idx])
    test_keys = sorted(keys[split_idx:])

    split = {
        "train": train_keys,
        "test": test_keys,
        "seed": 42,
        "train_size": len(train_keys),
        "test_size": len(test_keys),
    }

    split_path = OUTPUT_DIR / "ground_truth_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved: {split_path}")
    print(f"Train: {len(train_keys)} pairs, Test: {len(test_keys)} pairs")
    print(f"  Train: {train_keys}")
    print(f"  Test:  {test_keys}")


if __name__ == "__main__":
    main()
