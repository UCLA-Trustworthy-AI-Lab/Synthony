"""Shared utilities for baseline evaluation.

Provides ground truth loading, metric functions, dataset summary extraction,
and the evaluation harness used by all baselines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OVERLAP_MODELS = ["AIM", "AutoDiff", "DPCART", "TabDDPM", "TVAE", "ARF"]
FOCUS_NAMES = ["privacy", "fidelity", "latency"]
DATA_DIR = Path("data/input_data")
OUTPUT_DIR = Path("output")


# ── Ground truth loading ─────────────────────────────────────────────────


def load_ground_truth() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load ground_truth.json and ground_truth_split.json.

    Returns:
        (ground_truth dict, split dict)
    """
    with open(OUTPUT_DIR / "ground_truth.json") as f:
        gt = json.load(f)
    with open(OUTPUT_DIR / "ground_truth_split.json") as f:
        split = json.load(f)
    return gt, split


def find_csv(dataset_name: str) -> Path:
    """Case-insensitive CSV lookup in DATA_DIR."""
    exact = DATA_DIR / f"{dataset_name}.csv"
    if exact.exists():
        return exact
    for p in DATA_DIR.glob("*.csv"):
        if p.stem.lower() == dataset_name.lower():
            return p
    raise FileNotFoundError(f"No CSV found for '{dataset_name}' in {DATA_DIR}")


# ── Dataset summary (for LLM baseline) ──────────────────────────────────


def load_dataset_summary(csv_path: Path) -> Dict[str, Any]:
    """Load CSV and compute basic summary stats for prompting.

    Returns a plain dict with:
        rows, columns, column_names, dtypes,
        numeric_stats (per-column mean/std/min/max/skew),
        categorical_stats (per-column nunique, top values).
    """
    df = pd.read_csv(csv_path)

    summary: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        numeric_stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(s.skew()), 4),
        }
    summary["numeric_stats"] = numeric_stats

    # Categorical / object columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_stats = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        cat_stats[col] = {
            "nunique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
        }
    summary["categorical_stats"] = cat_stats

    return summary


# ── Metrics (shared with optimize_scaling.py) ────────────────────────────


def top_k_accuracy(predicted_models: List[List[str]],
                   gt_best_models: List[str],
                   k: int = 1) -> float:
    """Fraction of cases where ground truth best is in top-k predicted."""
    correct = 0
    for pred_ranking, gt_best in zip(predicted_models, gt_best_models):
        if gt_best in pred_ranking[:k]:
            correct += 1
    return correct / len(gt_best_models) if gt_best_models else 0.0


def spearman_rank_correlation(predicted_ranking: List[str],
                              gt_ranking: List[str]) -> float:
    """Spearman correlation between predicted and ground truth rankings."""
    common = [m for m in gt_ranking if m in predicted_ranking]
    if len(common) < 2:
        return 0.0
    gt_ranks = [gt_ranking.index(m) for m in common]
    pred_ranks = [predicted_ranking.index(m) for m in common]
    corr, _ = spearmanr(gt_ranks, pred_ranks)
    return corr if not np.isnan(corr) else 0.0


def ndcg(predicted_ranking: List[str], gt_ranking: List[str]) -> float:
    """Normalized Discounted Cumulative Gain.

    Relevance of position i in gt_ranking = len(gt_ranking) - i.
    """
    n = len(gt_ranking)
    if n == 0:
        return 0.0
    relevance = {model: n - i for i, model in enumerate(gt_ranking)}

    dcg = 0.0
    for i, model in enumerate(predicted_ranking):
        rel = relevance.get(model, 0)
        dcg += rel / np.log2(i + 2)

    idcg = 0.0
    for i in range(n):
        rel = n - i
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ── Evaluation harness ───────────────────────────────────────────────────


def evaluate_baseline(name: str,
                      predictions: Dict[str, List[str]],
                      gt: Dict[str, Any],
                      split: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a baseline's predictions against ground truth.

    Args:
        name: Baseline name (for display).
        predictions: {key: [ranked model list]} for all 21 keys.
        gt: Ground truth dict.
        split: Train/test split dict.

    Returns:
        {train: {top1, top3, spearman, ndcg}, test: {...}}.
    """
    results = {}
    for split_name in ["train", "test"]:
        keys = split[split_name]
        pred_rankings = []
        gt_best_models = []
        gt_rankings = []

        for key in keys:
            entry = gt[key]
            pred = predictions.get(key)
            if pred is None:
                pred = list(OVERLAP_MODELS)  # fallback: arbitrary order
            pred_rankings.append(pred)
            gt_best_models.append(entry["best_model"])
            gt_rankings.append(entry["ranking"])

        top1 = top_k_accuracy(pred_rankings, gt_best_models, k=1)
        top3 = top_k_accuracy(pred_rankings, gt_best_models, k=3)

        sp_scores = [
            spearman_rank_correlation(p, g)
            for p, g in zip(pred_rankings, gt_rankings)
        ]
        ndcg_scores = [
            ndcg(p, g)
            for p, g in zip(pred_rankings, gt_rankings)
        ]

        results[split_name] = {
            "top1": top1,
            "top3": top3,
            "spearman": float(np.mean(sp_scores)),
            "ndcg": float(np.mean(ndcg_scores)),
            "n": len(keys),
        }

    return results
