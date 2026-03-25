"""Baseline 3: Random Search (lower-bound baseline).

Samples uniformly random permutations of the 6 models.

Reports both:
  1. Single-shot (seed=42): one deterministic ranking per pair
  2. Expected-value (1000 trials): average metrics over random seeds

Usage:
    python baselines/random_search.py
"""

import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from baselines.common import (
    OVERLAP_MODELS,
    evaluate_baseline,
    load_ground_truth,
    ndcg,
    spearman_rank_correlation,
    top_k_accuracy,
)


def predict(csv_path: Path, focus: str, seed: int = None) -> List[str]:
    """Return a random permutation of 6 models."""
    rng = random.Random(seed)
    models = list(OVERLAP_MODELS)
    rng.shuffle(models)
    return models


def run_single_shot(gt: Dict, seed: int = 42) -> Dict[str, List[str]]:
    """Run one random ranking per pair with a fixed seed.

    Each key gets its own deterministic shuffle derived from the global seed
    and the key index, so results are reproducible.
    """
    predictions = {}
    rng = random.Random(seed)
    for key in sorted(gt.keys()):
        models = list(OVERLAP_MODELS)
        rng.shuffle(models)
        predictions[key] = models
    return predictions


def run_expected_value(gt: Dict, split: Dict,
                       n_trials: int = 1000) -> Dict[str, Dict]:
    """Run n_trials random rankings and average metrics per split."""
    results = {}

    for split_name in ["train", "test"]:
        keys = split[split_name]
        all_top1, all_top3, all_sp, all_ndcg = [], [], [], []

        for trial in range(n_trials):
            rng = random.Random(trial)
            pred_rankings = []
            gt_best_models = []
            gt_rankings = []

            for key in keys:
                entry = gt[key]
                models = list(OVERLAP_MODELS)
                rng.shuffle(models)
                pred_rankings.append(models)
                gt_best_models.append(entry["best_model"])
                gt_rankings.append(entry["ranking"])

            all_top1.append(top_k_accuracy(pred_rankings, gt_best_models, k=1))
            all_top3.append(top_k_accuracy(pred_rankings, gt_best_models, k=3))
            sp_scores = [
                spearman_rank_correlation(p, g)
                for p, g in zip(pred_rankings, gt_rankings)
            ]
            ndcg_scores = [
                ndcg(p, g)
                for p, g in zip(pred_rankings, gt_rankings)
            ]
            all_sp.append(float(np.mean(sp_scores)))
            all_ndcg.append(float(np.mean(ndcg_scores)))

        results[split_name] = {
            "top1": float(np.mean(all_top1)),
            "top3": float(np.mean(all_top3)),
            "spearman": float(np.mean(all_sp)),
            "ndcg": float(np.mean(all_ndcg)),
            "n": len(keys),
        }

    return results


if __name__ == "__main__":
    gt, split = load_ground_truth()

    # Single-shot
    predictions = run_single_shot(gt, seed=42)
    results_single = evaluate_baseline("Random (seed=42)", predictions, gt, split)

    print("Random Baseline (single shot, seed=42):")
    print(f"  Train: Top-1={results_single['train']['top1']:.3f}  "
          f"Top-3={results_single['train']['top3']:.3f}  "
          f"Spearman={results_single['train']['spearman']:.3f}  "
          f"NDCG={results_single['train']['ndcg']:.3f}")
    print(f"  Test:  Top-1={results_single['test']['top1']:.3f}  "
          f"Top-3={results_single['test']['top3']:.3f}  "
          f"Spearman={results_single['test']['spearman']:.3f}  "
          f"NDCG={results_single['test']['ndcg']:.3f}")

    # Expected value
    print("\nRandom Baseline (E[1000 trials]):")
    results_ev = run_expected_value(gt, split, n_trials=1000)
    print(f"  Train: Top-1={results_ev['train']['top1']:.3f}  "
          f"Top-3={results_ev['train']['top3']:.3f}  "
          f"Spearman={results_ev['train']['spearman']:.3f}  "
          f"NDCG={results_ev['train']['ndcg']:.3f}")
    print(f"  Test:  Top-1={results_ev['test']['top1']:.3f}  "
          f"Top-3={results_ev['test']['top3']:.3f}  "
          f"Spearman={results_ev['test']['spearman']:.3f}  "
          f"NDCG={results_ev['test']['ndcg']:.3f}")
