"""Baseline 1: Static Heuristic (Rule-of-Thumb) decision tree.

A simple practitioner heuristic that uses only row_count and focus to rank
the 6 models. Does NOT use Synthony stress profiles or capability scores.

Usage:
    python baselines/static_heuristic.py
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from baselines.common import (
    DATA_DIR,
    FOCUS_NAMES,
    OVERLAP_MODELS,
    evaluate_baseline,
    find_csv,
    load_ground_truth,
)


def predict(csv_path: Path, focus: str) -> List[str]:
    """Return ranked list of 6 models using simple heuristic rules.

    Only uses row_count and focus — no statistical profiling.
    """
    df = pd.read_csv(csv_path)
    rows = len(df)

    if focus == "privacy":
        # DP models first; prefer DPCART for small data, AIM for large
        if rows < 1000:
            return ["DPCART", "AIM", "TVAE", "ARF", "AutoDiff", "TabDDPM"]
        else:
            return ["AIM", "DPCART", "AutoDiff", "TabDDPM", "TVAE", "ARF"]

    elif focus == "fidelity":
        if rows < 500:
            return ["ARF", "TVAE", "TabDDPM", "AutoDiff", "AIM", "DPCART"]
        elif rows > 10000:
            return ["TabDDPM", "AutoDiff", "ARF", "TVAE", "AIM", "DPCART"]
        else:
            return ["ARF", "TabDDPM", "AutoDiff", "TVAE", "AIM", "DPCART"]

    elif focus == "latency":
        # Fixed ranking by known speed
        return ["DPCART", "AIM", "TVAE", "ARF", "TabDDPM", "AutoDiff"]

    else:
        raise ValueError(f"Unknown focus: {focus}")


def run_all(gt: Dict, split: Dict) -> Dict[str, List[str]]:
    """Run heuristic on all 21 dataset-focus pairs."""
    predictions = {}
    for key, entry in gt.items():
        csv_path = find_csv(entry["dataset"])
        ranking = predict(csv_path, entry["focus"])
        predictions[key] = ranking
    return predictions


if __name__ == "__main__":
    gt, split = load_ground_truth()
    predictions = run_all(gt, split)

    print("Static Heuristic Rankings:")
    print(f"{'Key':<35} {'Predicted #1':<12} {'GT Best':<12} {'Match'}")
    print("-" * 65)
    for key in sorted(predictions):
        pred = predictions[key]
        gt_best = gt[key]["best_model"]
        match = "Y" if pred[0] == gt_best else ""
        print(f"{key:<35} {pred[0]:<12} {gt_best:<12} {match}")

    results = evaluate_baseline("Static Heuristic", predictions, gt, split)
    print(f"\nTrain: Top-1={results['train']['top1']:.3f}  "
          f"Top-3={results['train']['top3']:.3f}  "
          f"Spearman={results['train']['spearman']:.3f}  "
          f"NDCG={results['train']['ndcg']:.3f}")
    print(f"Test:  Top-1={results['test']['top1']:.3f}  "
          f"Top-3={results['test']['top3']:.3f}  "
          f"Spearman={results['test']['spearman']:.3f}  "
          f"NDCG={results['test']['ndcg']:.3f}")
