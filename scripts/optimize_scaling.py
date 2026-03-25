"""
Bayesian optimization of focus-based scale factors using Optuna.

Optimizes 18 scale factor parameters (6 capabilities × 3 focuses) to maximize
Top-1 accuracy against ground truth rankings on the training set.
Then evaluates on the held-out test set with Top-1, Top-3, Spearman, and NDCG.

Usage:
    python scripts/optimize_scaling.py

Dependencies:
    pip install optuna
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine
from synthony.recommender.focus_profiles import (
    CAPABILITY_NAMES,
    FOCUS_REGISTRY,
    register_focus,
)

# ── Constants ──────────────────────────────────────────────────────────────

OVERLAP_MODELS = ["AIM", "AutoDiff", "DPCART", "TabDDPM", "TVAE", "ARF"]
FOCUS_NAMES = ["privacy", "fidelity", "latency"]
DATA_DIR = Path("data/input_data")
OUTPUT_DIR = Path("output")

N_TRIALS = 200      # 30 random + 170 TPE
N_STARTUP = 30      # Random exploration trials
SF_LOW = 0.0
SF_HIGH = 10.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_csv(dataset_name: str) -> Path:
    """Find CSV file for a dataset name, handling case mismatches."""
    # Try exact match first
    exact = DATA_DIR / f"{dataset_name}.csv"
    if exact.exists():
        return exact
    # Case-insensitive fallback
    for p in DATA_DIR.glob("*.csv"):
        if p.stem.lower() == dataset_name.lower():
            return p
    raise FileNotFoundError(f"No CSV found for dataset '{dataset_name}' in {DATA_DIR}")


def load_profiles():
    """Pre-compute dataset profiles and column analyses for all ground truth datasets."""
    with open(OUTPUT_DIR / "ground_truth.json") as f:
        gt = json.load(f)

    # Unique dataset names
    datasets = sorted({v["dataset"] for v in gt.values()})

    analyzer = StochasticDataAnalyzer()
    column_analyzer = ColumnAnalyzer()

    profiles = {}
    for ds_name in datasets:
        csv_path = _find_csv(ds_name)
        df = pd.read_csv(csv_path)
        profile = analyzer.analyze(df)
        col_analysis = column_analyzer.analyze(df, profile)
        profiles[ds_name] = (profile, col_analysis)
        print(f"  Profiled: {ds_name} ({df.shape[0]} rows)")

    return profiles


def build_scale_factor_dicts(trial_params: dict) -> dict:
    """Convert flat trial params into {focus: {cap: sf}} structure."""
    sf_dicts = {}
    for focus in FOCUS_NAMES:
        sf_dicts[focus] = {
            cap: trial_params[f"{focus}__{cap}"]
            for cap in CAPABILITY_NAMES
        }
    return sf_dicts


def run_recommendation(engine, profile, col_analysis, scale_factors):
    """Run engine recommendation with given scale factors, restricted to overlap models."""
    result = engine.recommend(
        dataset_profile=profile,
        column_analysis=col_analysis,
        constraints={"allowed_models": OVERLAP_MODELS},
        method="rule_based",
        top_n=len(OVERLAP_MODELS),
        scale_factors=scale_factors,
    )
    return result


def get_predicted_ranking(result):
    """Extract full ranking from recommendation result."""
    ranking = [result.recommended_model.model_name]
    for alt in result.alternative_models:
        ranking.append(alt.model_name)
    return ranking


# ── Metrics ────────────────────────────────────────────────────────────────

def top_k_accuracy(predicted_models, gt_best_models, k=1):
    """Fraction of cases where ground truth best is in top-k predicted."""
    correct = 0
    for pred_ranking, gt_best in zip(predicted_models, gt_best_models):
        if gt_best in pred_ranking[:k]:
            correct += 1
    return correct / len(gt_best_models) if gt_best_models else 0.0


def spearman_rank_correlation(predicted_ranking, gt_ranking):
    """Spearman correlation between predicted and ground truth rankings.

    Only considers models present in both rankings.
    """
    from scipy.stats import spearmanr

    # Align to common models
    common = [m for m in gt_ranking if m in predicted_ranking]
    if len(common) < 2:
        return 0.0

    gt_ranks = [gt_ranking.index(m) for m in common]
    pred_ranks = [predicted_ranking.index(m) for m in common]

    corr, _ = spearmanr(gt_ranks, pred_ranks)
    return corr if not np.isnan(corr) else 0.0


def ndcg(predicted_ranking, gt_ranking):
    """Normalized Discounted Cumulative Gain.

    Relevance of position i in gt_ranking = len(gt_ranking) - i
    (best model gets highest relevance).
    """
    n = len(gt_ranking)
    if n == 0:
        return 0.0

    # Relevance: best model in gt gets score n, second gets n-1, etc.
    relevance = {model: n - i for i, model in enumerate(gt_ranking)}

    # DCG of predicted ranking
    dcg = 0.0
    for i, model in enumerate(predicted_ranking):
        rel = relevance.get(model, 0)
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (gt_ranking order)
    idcg = 0.0
    for i in range(n):
        rel = n - i
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ── Optimization ───────────────────────────────────────────────────────────

def optimize():
    """Run Bayesian optimization and final evaluation."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load ground truth and split
    with open(OUTPUT_DIR / "ground_truth.json") as f:
        gt = json.load(f)
    with open(OUTPUT_DIR / "ground_truth_split.json") as f:
        split = json.load(f)

    train_keys = split["train"]
    test_keys = split["test"]

    print(f"Ground truth: {len(gt)} pairs")
    print(f"Train: {len(train_keys)}, Test: {len(test_keys)}")

    # Pre-compute profiles
    print("\nProfiling datasets...")
    profiles = load_profiles()

    # Initialize engine (reused across all trials)
    engine = ModelRecommendationEngine()

    # ── Objective function ─────────────────────────────────────────────

    def objective(trial):
        # Sample 18 scale factor params
        params = {}
        for focus in FOCUS_NAMES:
            for cap in CAPABILITY_NAMES:
                name = f"{focus}__{cap}"
                params[name] = trial.suggest_float(name, SF_LOW, SF_HIGH)

        sf_dicts = build_scale_factor_dicts(params)

        # Evaluate on train set
        correct = 0
        for key in train_keys:
            entry = gt[key]
            ds_name = entry["dataset"]
            focus = entry["focus"]
            gt_best = entry["best_model"]

            profile, col_analysis = profiles[ds_name]
            result = run_recommendation(engine, profile, col_analysis, sf_dicts[focus])
            predicted = result.recommended_model.model_name

            if predicted == gt_best:
                correct += 1

        accuracy = correct / len(train_keys)
        return accuracy

    # ── Run optimization ───────────────────────────────────────────────

    print(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP,
        seed=42,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="synthony_scaling",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"\nBest trial #{best_trial.number}: accuracy={best_trial.value:.4f}")

    # Build best scale factor dicts
    best_sf = build_scale_factor_dicts(best_trial.params)

    # Save optimization history
    history = []
    for t in study.trials:
        history.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
        })
    history_path = OUTPUT_DIR / "optimization_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {history_path}")

    # Save best scale factors
    best_sf_path = OUTPUT_DIR / "best_scale_factors.json"
    with open(best_sf_path, "w") as f:
        json.dump(best_sf, f, indent=2)
    print(f"Saved: {best_sf_path}")

    # Update focus registry
    for focus_name, sf_dict in best_sf.items():
        register_focus(focus_name, sf_dict)
    print("Updated FOCUS_REGISTRY with optimized scale factors")

    # ── Final evaluation on test set ───────────────────────────────────

    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")

    # Also evaluate train set for comparison
    for split_name, keys in [("Train", train_keys), ("Test", test_keys)]:
        predicted_rankings = []
        gt_best_models = []
        gt_rankings = []
        details = []

        for key in keys:
            entry = gt[key]
            ds_name = entry["dataset"]
            focus = entry["focus"]
            gt_best = entry["best_model"]
            gt_rank = entry["ranking"]

            profile, col_analysis = profiles[ds_name]
            result = run_recommendation(engine, profile, col_analysis, best_sf[focus])
            pred_ranking = get_predicted_ranking(result)

            predicted_rankings.append(pred_ranking)
            gt_best_models.append(gt_best)
            gt_rankings.append(gt_rank)

            match = "✓" if pred_ranking[0] == gt_best else "✗"
            details.append({
                "key": key,
                "gt_best": gt_best,
                "predicted": pred_ranking[0],
                "match": pred_ranking[0] == gt_best,
                "pred_ranking": pred_ranking,
                "gt_ranking": gt_rank,
            })

        # Compute metrics
        top1 = top_k_accuracy(predicted_rankings, gt_best_models, k=1)
        top3 = top_k_accuracy(predicted_rankings, gt_best_models, k=3)

        spearman_scores = [
            spearman_rank_correlation(pred, gt_r)
            for pred, gt_r in zip(predicted_rankings, gt_rankings)
        ]
        avg_spearman = np.mean(spearman_scores)

        ndcg_scores = [
            ndcg(pred, gt_r)
            for pred, gt_r in zip(predicted_rankings, gt_rankings)
        ]
        avg_ndcg = np.mean(ndcg_scores)

        print(f"--- {split_name} Set ({len(keys)} pairs) ---")
        print(f"  Top-1 Accuracy: {top1:.4f} ({int(top1 * len(keys))}/{len(keys)})")
        print(f"  Top-3 Accuracy: {top3:.4f} ({int(top3 * len(keys))}/{len(keys)})")
        print(f"  Avg Spearman:   {avg_spearman:.4f}")
        print(f"  Avg NDCG:       {avg_ndcg:.4f}")
        print()

        # Print detail table
        print(f"  {'Key':<35} {'GT Best':<12} {'Predicted':<12} {'Match'}")
        print(f"  {'-'*70}")
        for d in details:
            mark = "✓" if d["match"] else "✗"
            print(f"  {d['key']:<35} {d['gt_best']:<12} {d['predicted']:<12} {mark}")
        print()

    # ── Save final report ──────────────────────────────────────────────

    report = {
        "best_trial": best_trial.number,
        "train_accuracy": best_trial.value,
        "best_scale_factors": best_sf,
        "n_trials": N_TRIALS,
        "evaluation": {},
    }

    for split_name, keys in [("train", train_keys), ("test", test_keys)]:
        pred_rankings = []
        gt_bests = []
        gt_ranks = []

        for key in keys:
            entry = gt[key]
            profile, col_analysis = profiles[entry["dataset"]]
            result = run_recommendation(engine, profile, col_analysis, best_sf[entry["focus"]])
            pred_ranking = get_predicted_ranking(result)
            pred_rankings.append(pred_ranking)
            gt_bests.append(entry["best_model"])
            gt_ranks.append(entry["ranking"])

        report["evaluation"][split_name] = {
            "top1_accuracy": top_k_accuracy(pred_rankings, gt_bests, k=1),
            "top3_accuracy": top_k_accuracy(pred_rankings, gt_bests, k=3),
            "avg_spearman": float(np.mean([
                spearman_rank_correlation(p, g)
                for p, g in zip(pred_rankings, gt_ranks)
            ])),
            "avg_ndcg": float(np.mean([
                ndcg(p, g)
                for p, g in zip(pred_rankings, gt_ranks)
            ])),
            "n_pairs": len(keys),
        }

    report_path = OUTPUT_DIR / "final_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Print best scale factors summary
    print(f"\n{'='*60}")
    print("OPTIMIZED SCALE FACTORS")
    print(f"{'='*60}")
    for focus in FOCUS_NAMES:
        print(f"\n  {focus}:")
        for cap in CAPABILITY_NAMES:
            val = best_sf[focus][cap]
            print(f"    {cap:<25} {val:.4f}")


if __name__ == "__main__":
    optimize()
