"""Baseline 4: Meta-feature kNN Selector.

Non-LLM meta-learning baseline. Extracts dataset meta-features, finds k nearest
training datasets, and recommends models based on neighbor ground truth rankings.

Uses leave-one-out: when predicting for dataset X, only uses other datasets.

Usage:
    python baselines/knn_selector.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from baselines.common import (
    OVERLAP_MODELS,
    evaluate_baseline,
    find_csv,
    load_ground_truth,
)


def extract_meta_features(csv_path: Path) -> np.ndarray:
    """Extract 9 dataset meta-features from a CSV file.

    Features:
        0. log(row_count)
        1. column_count
        2. numeric_ratio (fraction of numeric columns)
        3. max_skewness (absolute)
        4. mean_skewness (absolute)
        5. log(max_cardinality)
        6. log(mean_cardinality)
        7. max_null_pct
        8. correlation_density (fraction of |corr| > 0.1 pairs)

    Returns:
        1D numpy array of 9 features.
    """
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    numeric_cols = df.select_dtypes(include="number").columns
    n_numeric = len(numeric_cols)
    numeric_ratio = n_numeric / n_cols if n_cols > 0 else 0.0

    # Skewness
    if n_numeric > 0:
        skews = df[numeric_cols].skew().abs()
        max_skew = float(skews.max()) if not skews.empty else 0.0
        mean_skew = float(skews.mean()) if not skews.empty else 0.0
    else:
        max_skew = 0.0
        mean_skew = 0.0

    # Cardinality (all columns)
    cardinalities = df.nunique()
    max_card = float(cardinalities.max()) if not cardinalities.empty else 1.0
    mean_card = float(cardinalities.mean()) if not cardinalities.empty else 1.0

    # Null percentage
    null_pcts = df.isnull().mean()
    max_null = float(null_pcts.max()) if not null_pcts.empty else 0.0

    # Correlation density
    if n_numeric >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        n_pairs = n_numeric * (n_numeric - 1) / 2
        # Upper triangle, excluding diagonal
        upper = np.triu(corr_matrix.values, k=1)
        dense_count = (upper > 0.1).sum()
        corr_density = dense_count / n_pairs if n_pairs > 0 else 0.0
    else:
        corr_density = 0.0

    return np.array([
        np.log1p(n_rows),
        n_cols,
        numeric_ratio,
        max_skew,
        mean_skew,
        np.log1p(max_card),
        np.log1p(mean_card),
        max_null,
        corr_density,
    ])


def _build_meta_feature_matrix(gt: Dict) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Build meta-feature vectors for all unique datasets in ground truth.

    Returns:
        (features_dict: {dataset_name: feature_vector}, dataset_names: sorted list)
    """
    datasets = sorted({v["dataset"] for v in gt.values()})
    features = {}
    for ds_name in datasets:
        csv_path = find_csv(ds_name)
        features[ds_name] = extract_meta_features(csv_path)
    return features, datasets


def predict(query_features: np.ndarray,
            query_dataset: str,
            focus: str,
            all_features: Dict[str, np.ndarray],
            gt: Dict,
            k: int = 3) -> List[str]:
    """Use kNN on meta-features to rank models.

    Args:
        query_features: Meta-feature vector for query dataset.
        query_dataset: Name of query dataset (excluded from neighbors).
        focus: Focus name.
        all_features: {dataset_name: feature_vector} for all datasets.
        gt: Ground truth dict.
        k: Number of nearest neighbors.

    Returns:
        Ranked list of 6 models.
    """
    # Collect neighbor features (exclude query dataset for leave-one-out)
    neighbor_names = [ds for ds in all_features if ds != query_dataset]
    if not neighbor_names:
        return list(OVERLAP_MODELS)

    # Stack features and standardize (z-score using neighbor stats)
    neighbor_matrix = np.array([all_features[ds] for ds in neighbor_names])
    all_matrix = np.vstack([neighbor_matrix, query_features.reshape(1, -1)])

    mean = neighbor_matrix.mean(axis=0)
    std = neighbor_matrix.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero

    standardized = (all_matrix - mean) / std
    query_std = standardized[-1]
    neighbor_std = standardized[:-1]

    # Euclidean distances
    distances = np.linalg.norm(neighbor_std - query_std, axis=1)

    # Select k nearest
    k_actual = min(k, len(neighbor_names))
    nearest_idx = np.argsort(distances)[:k_actual]
    nearest_datasets = [neighbor_names[i] for i in nearest_idx]

    # Aggregate rankings: for each model, compute mean rank across neighbors
    model_ranks = {m: [] for m in OVERLAP_MODELS}
    for ds in nearest_datasets:
        key = f"{ds}_{focus}"
        if key in gt:
            gt_ranking = gt[key]["ranking"]
            for model in OVERLAP_MODELS:
                if model in gt_ranking:
                    model_ranks[model].append(gt_ranking.index(model))
                else:
                    model_ranks[model].append(len(OVERLAP_MODELS))

    # Sort by mean rank (lower is better)
    mean_ranks = {m: np.mean(ranks) if ranks else len(OVERLAP_MODELS)
                  for m, ranks in model_ranks.items()}
    sorted_models = sorted(mean_ranks.keys(), key=lambda m: mean_ranks[m])
    return sorted_models


def run_all(gt: Dict, split: Dict, k: int = 3) -> Dict[str, List[str]]:
    """Run kNN baseline on all 21 dataset-focus pairs."""
    all_features, _ = _build_meta_feature_matrix(gt)

    predictions = {}
    for key, entry in gt.items():
        ds_name = entry["dataset"]
        focus = entry["focus"]
        query_features = all_features[ds_name]
        ranking = predict(query_features, ds_name, focus, all_features, gt, k=k)
        predictions[key] = ranking

    return predictions


if __name__ == "__main__":
    gt, split = load_ground_truth()

    for k in [3, 5]:
        predictions = run_all(gt, split, k=k)
        results = evaluate_baseline(f"kNN (k={k})", predictions, gt, split)

        print(f"\nkNN Selector (k={k}):")
        print(f"{'Key':<35} {'Predicted #1':<12} {'GT Best':<12} {'Match'}")
        print("-" * 65)
        for key in sorted(predictions):
            pred = predictions[key]
            gt_best = gt[key]["best_model"]
            match = "Y" if pred[0] == gt_best else ""
            print(f"{key:<35} {pred[0]:<12} {gt_best:<12} {match}")

        print(f"\n  Train: Top-1={results['train']['top1']:.3f}  "
              f"Top-3={results['train']['top3']:.3f}  "
              f"Spearman={results['train']['spearman']:.3f}  "
              f"NDCG={results['train']['ndcg']:.3f}")
        print(f"  Test:  Top-1={results['test']['top1']:.3f}  "
              f"Top-3={results['test']['top3']:.3f}  "
              f"Spearman={results['test']['spearman']:.3f}  "
              f"NDCG={results['test']['ndcg']:.3f}")
