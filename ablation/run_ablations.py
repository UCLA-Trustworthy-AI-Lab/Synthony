"""Ablation experiments for Synthony.

Tests the contribution of each component:
  - Full (optimized): stress profiling + capability matching + focus scaling
  - No focus scaling: stress + capabilities, SF=1.0
  - No stress profiling: focus scaling but required_capabilities forced to 0
  - Vanilla LLM: no Synthony components (reference)

Also runs stress-prediction classifiers (majority, logistic regression, decision tree).

Usage:
    python -m ablation.run_ablations
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.common import (
    OVERLAP_MODELS,
    FOCUS_NAMES,
    OUTPUT_DIR,
    evaluate_baseline,
    find_csv,
    load_ground_truth,
    top_k_accuracy,
    spearman_rank_correlation,
    ndcg,
)
from baselines.knn_selector import extract_meta_features


# ── Synthony ablation helpers ────────────────────────────────────────────


def _get_engine_and_profiles(gt):
    """Initialize engine and precompute dataset profiles."""
    from synthony.core.analyzer import StochasticDataAnalyzer
    from synthony.core.column_analyzer import ColumnAnalyzer
    from synthony.recommender.engine import ModelRecommendationEngine

    analyzer = StochasticDataAnalyzer()
    col_analyzer = ColumnAnalyzer()
    engine = ModelRecommendationEngine()

    profiles = {}
    for ds_name in sorted({v["dataset"] for v in gt.values()}):
        csv_path = find_csv(ds_name)
        df = pd.read_csv(csv_path)
        profile = analyzer.analyze(df)
        col_analysis = col_analyzer.analyze(df, profile)
        profiles[ds_name] = (profile, col_analysis)

    return engine, profiles


def _run_synthony_variant(engine, profiles, gt, scale_factors_dict,
                          zero_stress=False):
    """Run a Synthony variant on all 21 pairs.

    Args:
        scale_factors_dict: {focus: {cap: float}} scale factors per focus.
        zero_stress: If True, force required_capabilities to all zeros
                     (bypasses stress profiling contribution).
    """
    predictions = {}

    # Monkey-patch if zero_stress
    original_calc = None
    if zero_stress:
        original_calc = engine._calculate_required_capabilities
        engine._calculate_required_capabilities = lambda *args, **kwargs: {
            "skew_handling": 0,
            "cardinality_handling": 0,
            "zipfian_handling": 0,
            "small_data": 0,
            "correlation_handling": 0,
        }

    try:
        for key, entry in gt.items():
            ds_name = entry["dataset"]
            focus = entry["focus"]
            profile, col_analysis = profiles[ds_name]
            sf = scale_factors_dict[focus]

            result = engine.recommend(
                dataset_profile=profile,
                column_analysis=col_analysis,
                constraints={"allowed_models": OVERLAP_MODELS},
                method="rule_based",
                top_n=len(OVERLAP_MODELS),
                scale_factors=sf,
            )
            ranking = [result.recommended_model.model_name]
            for alt in result.alternative_models:
                ranking.append(alt.model_name)
            predictions[key] = ranking
    finally:
        if original_calc is not None:
            engine._calculate_required_capabilities = original_calc

    return predictions


# ── Stress prediction classifiers ────────────────────────────────────────


def run_stress_prediction(gt):
    """Run stress-prediction classifiers with leave-one-dataset-out CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler

    datasets = sorted({v["dataset"] for v in gt.values()})

    # Extract meta-features
    features = {}
    for ds in datasets:
        features[ds] = extract_meta_features(find_csv(ds))

    results = {}

    for focus in FOCUS_NAMES:
        # Build labels: best model per dataset for this focus
        labels = {}
        for ds in datasets:
            key = f"{ds}_{focus}"
            if key in gt:
                labels[ds] = gt[key]["best_model"]

        ds_with_labels = [ds for ds in datasets if ds in labels]

        # Leave-one-dataset-out CV
        majority_preds = []
        logreg_preds = []
        dtree_preds = []
        true_labels = []

        for i, test_ds in enumerate(ds_with_labels):
            train_ds = [ds for ds in ds_with_labels if ds != test_ds]
            X_train = np.array([features[ds] for ds in train_ds])
            y_train = [labels[ds] for ds in train_ds]
            X_test = features[test_ds].reshape(1, -1)
            y_true = labels[test_ds]

            true_labels.append(y_true)

            # Majority vote
            counter = Counter(y_train)
            majority_preds.append(counter.most_common(1)[0][0])

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_s, y_train)
            logreg_preds.append(lr.predict(X_test_s)[0])

            # Decision tree
            dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt.fit(X_train_s, y_train)
            dtree_preds.append(dt.predict(X_test_s)[0])

        # Compute accuracy for this focus
        for name, preds in [("Majority", majority_preds),
                            ("LogReg", logreg_preds),
                            ("DecTree", dtree_preds)]:
            correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
            acc = correct / len(true_labels)
            results.setdefault(name, []).append(acc)

    # Average across focuses
    avg_results = {}
    for name, accs in results.items():
        avg_results[name] = float(np.mean(accs))

    return avg_results, results


# ── Stress prediction: full ranking ──────────────────────────────────────


def run_stress_prediction_full(gt, split):
    """Run stress-prediction classifiers producing full rankings (not just top-1).

    Uses class probabilities to generate ranked lists, then evaluates with
    the standard metrics (Top-1, Top-3, Spearman, NDCG).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler

    datasets = sorted({v["dataset"] for v in gt.values()})
    features = {}
    for ds in datasets:
        features[ds] = extract_meta_features(find_csv(ds))

    classifier_predictions = {
        "Majority": {},
        "LogReg": {},
        "DecTree": {},
    }

    for focus in FOCUS_NAMES:
        labels = {}
        for ds in datasets:
            key = f"{ds}_{focus}"
            if key in gt:
                labels[ds] = gt[key]["best_model"]

        ds_with_labels = [ds for ds in datasets if ds in labels]

        for test_ds in ds_with_labels:
            train_ds = [ds for ds in ds_with_labels if ds != test_ds]
            X_train = np.array([features[ds] for ds in train_ds])
            y_train = [labels[ds] for ds in train_ds]
            X_test = features[test_ds].reshape(1, -1)
            key = f"{test_ds}_{focus}"

            # Majority: rank by frequency in training set
            counter = Counter(y_train)
            freq_ranking = [m for m, _ in counter.most_common()]
            remaining = [m for m in OVERLAP_MODELS if m not in freq_ranking]
            classifier_predictions["Majority"][key] = freq_ranking + remaining

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Logistic regression with probability ranking
            unique_labels = set(y_train)
            if len(unique_labels) < 2:
                # Only one class in training — predict that class
                only_class = list(unique_labels)[0]
                lr_ranking = [only_class] + [m for m in OVERLAP_MODELS if m != only_class]
                dt_ranking = lr_ranking
            else:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train_s, y_train)
                probs = lr.predict_proba(X_test_s)[0]
                classes = lr.classes_
                lr_ranking = [classes[i] for i in np.argsort(-probs)]
                lr_ranking = list(lr_ranking)
                remaining = [m for m in OVERLAP_MODELS if m not in lr_ranking]
                lr_ranking = lr_ranking + remaining

                # Decision tree
                dt = DecisionTreeClassifier(max_depth=3, random_state=42)
                dt.fit(X_train_s, y_train)
                probs = dt.predict_proba(X_test_s)[0]
                classes = dt.classes_
                dt_ranking = [classes[i] for i in np.argsort(-probs)]
                dt_ranking = list(dt_ranking)
                remaining = [m for m in OVERLAP_MODELS if m not in dt_ranking]
                dt_ranking = dt_ranking + remaining

            classifier_predictions["LogReg"][key] = lr_ranking
            classifier_predictions["DecTree"][key] = dt_ranking

    # Evaluate each classifier
    results = {}
    for name, preds in classifier_predictions.items():
        results[name] = evaluate_baseline(name, preds, gt, split)

    return results


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    gt, split = load_ground_truth()
    print(f"Ground truth: {len(gt)} pairs (train={len(split['train'])}, "
          f"test={len(split['test'])})\n")

    # Load optimized scale factors
    with open(OUTPUT_DIR / "best_scale_factors.json") as f:
        opt_sf = json.load(f)

    # Uniform scale factors
    uniform_sf = {
        focus: {cap: 1.0 for cap in opt_sf["privacy"]}
        for focus in FOCUS_NAMES
    }

    engine, profiles = _get_engine_and_profiles(gt)

    all_results = {}

    # ── Ablation variants ──────────────────────────────────────────────

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # Variant 1: Full Synthony (optimized)
    print("\n[1] Full Synthony (optimized)...")
    preds = _run_synthony_variant(engine, profiles, gt, opt_sf, zero_stress=False)
    res = evaluate_baseline("Full (optimized)", preds, gt, split)
    all_results["Full Synthony"] = res
    _print_result("Full Synthony", res)

    # Variant 2: No focus scaling (SF=1.0)
    print("\n[2] No focus scaling (SF=1.0)...")
    preds = _run_synthony_variant(engine, profiles, gt, uniform_sf, zero_stress=False)
    res = evaluate_baseline("No focus scaling", preds, gt, split)
    all_results["- Focus scaling"] = res
    _print_result("- Focus scaling", res)

    # Variant 3: No stress profiling (required_caps=0, optimized SF)
    print("\n[3] No stress profiling (optimized SF, no stress)...")
    preds = _run_synthony_variant(engine, profiles, gt, opt_sf, zero_stress=True)
    res = evaluate_baseline("No stress", preds, gt, split)
    all_results["- Stress profiling"] = res
    _print_result("- Stress profiling", res)

    # Variant 4: Neither stress nor focus (baseline capability scoring)
    print("\n[4] No stress + no focus (SF=1.0, no stress)...")
    preds = _run_synthony_variant(engine, profiles, gt, uniform_sf, zero_stress=True)
    res = evaluate_baseline("Bare scoring", preds, gt, split)
    all_results["- Stress - Focus"] = res
    _print_result("- Stress - Focus", res)

    # Reference: Vanilla LLM
    print("\n[Ref] Vanilla LLM (gpt-4o-mini)...")
    all_results["Vanilla LLM"] = {
        "train": {"top1": 0.429, "top3": 0.571, "spearman": 0.061, "ndcg": 0.882, "n": 14},
        "test": {"top1": 0.286, "top3": 0.429, "spearman": 0.004, "ndcg": 0.865, "n": 7},
    }
    _print_result("Vanilla LLM", all_results["Vanilla LLM"])

    # ── Ablation summary table ─────────────────────────────────────────

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    header = f"{'Variant':<25} | {'Tr Top-1':>8} | {'Tr NDCG':>8} | {'Te Top-1':>8} | {'Te NDCG':>8}"
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        tr = res["train"]
        te = res["test"]
        print(f"{name:<25} | {tr['top1']:.3f}    | {tr['ndcg']:.3f}    | "
              f"{te['top1']:.3f}    | {te['ndcg']:.3f}")

    # ── Stress prediction experiment ───────────────────────────────────

    print("\n" + "=" * 70)
    print("STRESS PREDICTION (classifiers on meta-features)")
    print("=" * 70)

    stress_results = run_stress_prediction_full(gt, split)

    # Add kNN reference from baselines
    from baselines.knn_selector import run_all as run_knn
    knn_preds = run_knn(gt, split, k=3)
    stress_results["kNN (k=3)"] = evaluate_baseline("kNN", knn_preds, gt, split)

    print(f"\n{'Predictor':<25} | {'Tr Top-1':>8} | {'Tr Top-3':>8} | "
          f"{'Te Top-1':>8} | {'Te Top-3':>8}")
    print("-" * 75)
    for name, res in stress_results.items():
        tr = res["train"]
        te = res["test"]
        print(f"{name:<25} | {tr['top1']:.3f}    | {tr['top3']:.3f}    | "
              f"{te['top1']:.3f}    | {te['top3']:.3f}")

    # ── Save all results ───────────────────────────────────────────────

    save_data = {
        "ablation": all_results,
        "stress_prediction": stress_results,
    }
    save_path = Path("ablation/results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nSaved: {save_path}")


def _print_result(name, res):
    tr = res["train"]
    te = res["test"]
    print(f"  Train: Top-1={tr['top1']:.3f}  Top-3={tr['top3']:.3f}  "
          f"Spearman={tr['spearman']:.3f}  NDCG={tr['ndcg']:.3f}")
    print(f"  Test:  Top-1={te['top1']:.3f}  Top-3={te['top3']:.3f}  "
          f"Spearman={te['spearman']:.3f}  NDCG={te['ndcg']:.3f}")


if __name__ == "__main__":
    main()
