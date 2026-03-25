#!/usr/bin/env python3
"""
Analyze benchmark results from trial4 and compare with model_capabilities.

This script:
1. Loads all benchmark results from ./output/benchmark/trial4/
2. Aggregates metrics by model
3. Compares empirical results with model_capabilities assumptions
4. Identifies gaps between expected and actual performance
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
import statistics


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model across all datasets."""
    model_name: str
    dataset_count: int
    avg_quality_score: float
    avg_fidelity: float
    avg_utility: float
    avg_privacy: float
    avg_kl_divergence: float
    avg_js_divergence: float
    # Stress factor preservation rates
    skew_preservation_rate: float  # % of datasets where skew was preserved
    cardinality_preservation_rate: float
    correlation_preservation_rate: float


def load_benchmark_results(benchmark_dir: Path) -> Dict[str, Dict[str, dict]]:
    """Load all benchmark results, organized by model then dataset."""
    results = defaultdict(dict)

    for json_file in benchmark_dir.glob("benchmark__*.json"):
        # Parse filename: benchmark__dataset__model.json
        parts = json_file.stem.split("__")
        if len(parts) >= 3:
            dataset = parts[1]
            model = parts[2]

            with open(json_file) as f:
                data = json.load(f)

            results[model][dataset] = data

    return dict(results)


def aggregate_model_metrics(results: Dict[str, Dict[str, dict]]) -> Dict[str, ModelMetrics]:
    """Aggregate metrics across datasets for each model."""
    aggregated = {}

    for model, datasets in results.items():
        quality_scores = []
        fidelity_scores = []
        utility_scores = []
        privacy_scores = []
        kl_divergences = []
        js_divergences = []
        skew_preserved = 0
        card_preserved = 0
        corr_preserved = 0
        total_with_profile = 0

        for dataset, data in datasets.items():
            quality_scores.append(data.get("overall_quality_score", 0))

            if "fidelity" in data:
                fidelity_scores.append(data["fidelity"].get("overall_fidelity", 0))
            if "utility" in data:
                utility_scores.append(data["utility"].get("overall_utility", 0))
            if "privacy" in data:
                privacy_scores.append(data["privacy"].get("privacy_score", 0))

            kl_divergences.append(data.get("avg_kl_divergence", 0))
            js_divergences.append(data.get("avg_js_divergence", 0))

            # Check stress factor preservation
            if "profile_comparison" in data:
                total_with_profile += 1
                sf = data["profile_comparison"].get("stress_factors", {})

                # Skew preservation: synthetic matches original
                if sf.get("severe_skew"):
                    orig_skew = sf["severe_skew"].get("original", False)
                    synth_skew = sf["severe_skew"].get("synthetic", False)
                    if orig_skew == synth_skew:
                        skew_preserved += 1

                # Cardinality preservation
                if sf.get("high_cardinality"):
                    orig_card = sf["high_cardinality"].get("original", False)
                    synth_card = sf["high_cardinality"].get("synthetic", False)
                    if orig_card == synth_card:
                        card_preserved += 1

                # Correlation preservation (use correlation metrics)
                corr_data = data["profile_comparison"].get("correlation", {})
                orig_r2 = corr_data.get("original", {}).get("mean_r_squared", 0)
                synth_r2 = corr_data.get("synthetic", {}).get("mean_r_squared", 0)
                # If R² is within 20% of original, consider it preserved
                if orig_r2 > 0 and abs(synth_r2 - orig_r2) / orig_r2 < 0.2:
                    corr_preserved += 1

        aggregated[model] = ModelMetrics(
            model_name=model,
            dataset_count=len(datasets),
            avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0,
            avg_fidelity=statistics.mean(fidelity_scores) if fidelity_scores else 0,
            avg_utility=statistics.mean(utility_scores) if utility_scores else 0,
            avg_privacy=statistics.mean(privacy_scores) if privacy_scores else 0,
            avg_kl_divergence=statistics.mean(kl_divergences) if kl_divergences else 0,
            avg_js_divergence=statistics.mean(js_divergences) if js_divergences else 0,
            skew_preservation_rate=skew_preserved / total_with_profile if total_with_profile > 0 else 0,
            cardinality_preservation_rate=card_preserved / total_with_profile if total_with_profile > 0 else 0,
            correlation_preservation_rate=corr_preserved / total_with_profile if total_with_profile > 0 else 0,
        )

    return aggregated


def load_model_capabilities(capabilities_path: Path) -> dict:
    """Load model capabilities JSON."""
    with open(capabilities_path) as f:
        return json.load(f)


def score_to_capability(score: float) -> int:
    """Convert a 0-1 score to a 0-4 capability rating."""
    if score >= 0.9:
        return 4
    elif score >= 0.75:
        return 3
    elif score >= 0.5:
        return 2
    elif score >= 0.25:
        return 1
    return 0


def compare_with_assumptions(
    metrics: Dict[str, ModelMetrics],
    capabilities: dict
) -> Dict[str, dict]:
    """Compare empirical metrics with model capability assumptions."""
    comparisons = {}

    for model_name, model_metrics in metrics.items():
        model_caps = capabilities.get("models", {}).get(model_name, {})
        caps = model_caps.get("capabilities", {})

        # Derive empirical capability scores
        empirical_skew = score_to_capability(model_metrics.skew_preservation_rate)
        empirical_card = score_to_capability(model_metrics.cardinality_preservation_rate)
        empirical_corr = score_to_capability(model_metrics.correlation_preservation_rate)
        empirical_quality = score_to_capability(model_metrics.avg_quality_score)

        # Get assumed scores
        assumed_skew = caps.get("skew_handling", 2)
        assumed_card = caps.get("cardinality_handling", 2)
        assumed_corr = caps.get("correlation_handling", 2)

        comparisons[model_name] = {
            "empirical": {
                "quality_score": model_metrics.avg_quality_score,
                "fidelity": model_metrics.avg_fidelity,
                "utility": model_metrics.avg_utility,
                "privacy": model_metrics.avg_privacy,
                "skew_preservation": model_metrics.skew_preservation_rate,
                "cardinality_preservation": model_metrics.cardinality_preservation_rate,
                "correlation_preservation": model_metrics.correlation_preservation_rate,
                "datasets_tested": model_metrics.dataset_count,
            },
            "assumed_capabilities": caps,
            "derived_capabilities": {
                "skew_handling": empirical_skew,
                "cardinality_handling": empirical_card,
                "correlation_handling": empirical_corr,
            },
            "gaps": {
                "skew_handling": empirical_skew - assumed_skew,
                "cardinality_handling": empirical_card - assumed_card,
                "correlation_handling": empirical_corr - assumed_corr,
            }
        }

    return comparisons


def print_analysis(metrics: Dict[str, ModelMetrics], comparisons: Dict[str, dict]):
    """Print formatted analysis results."""

    print("=" * 80)
    print("BENCHMARK ANALYSIS - Trial 4")
    print("=" * 80)
    print()

    # Sort models by quality score
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].avg_quality_score, reverse=True)

    print("MODEL RANKINGS BY QUALITY SCORE")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<20} {'Quality':<10} {'Fidelity':<10} {'Utility':<10} {'Privacy':<10} {'Datasets':<8}")
    print("-" * 80)

    for rank, (model, m) in enumerate(sorted_models, 1):
        print(f"{rank:<5} {model:<20} {m.avg_quality_score:.3f}     {m.avg_fidelity:.3f}     {m.avg_utility:.3f}     {m.avg_privacy:.3f}     {m.dataset_count:<8}")

    print()
    print("=" * 80)
    print("STRESS FACTOR PRESERVATION RATES")
    print("-" * 80)
    print(f"{'Model':<20} {'Skew %':<12} {'Card %':<12} {'Corr %':<12}")
    print("-" * 80)

    for model, m in sorted_models:
        print(f"{model:<20} {m.skew_preservation_rate*100:.1f}%        {m.cardinality_preservation_rate*100:.1f}%        {m.correlation_preservation_rate*100:.1f}%")

    print()
    print("=" * 80)
    print("COMPARISON WITH MODEL CAPABILITIES ASSUMPTIONS")
    print("-" * 80)

    for model, comp in comparisons.items():
        gaps = comp["gaps"]
        # Only show if there are significant gaps
        has_gap = any(abs(g) >= 1 for g in gaps.values())

        assumed = comp["assumed_capabilities"]
        derived = comp["derived_capabilities"]

        print(f"\n{model}:")
        print(f"  Datasets tested: {comp['empirical']['datasets_tested']}")
        print(f"  Avg Quality Score: {comp['empirical']['quality_score']:.3f}")
        print(f"  Capability Comparison (Assumed → Empirical):")

        for cap in ["skew_handling", "cardinality_handling", "correlation_handling"]:
            a = assumed.get(cap, 2)
            e = derived.get(cap, 2)
            gap = gaps.get(cap, 0)
            indicator = ""
            if gap > 0:
                indicator = " [↑ BETTER than expected]"
            elif gap < 0:
                indicator = " [↓ WORSE than expected]"
            print(f"    {cap}: {a} → {e} (gap: {gap:+d}){indicator}")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Find top performers
    top_quality = sorted_models[0]
    top_fidelity = max(metrics.items(), key=lambda x: x[1].avg_fidelity)
    top_utility = max(metrics.items(), key=lambda x: x[1].avg_utility)

    print(f"\n1. BEST OVERALL QUALITY: {top_quality[0]} (score: {top_quality[1].avg_quality_score:.3f})")
    print(f"2. BEST FIDELITY: {top_fidelity[0]} (score: {top_fidelity[1].avg_fidelity:.3f})")
    print(f"3. BEST UTILITY: {top_utility[0]} (score: {top_utility[1].avg_utility:.3f})")

    # Find models that underperformed expectations
    print("\n4. MODELS UNDERPERFORMING EXPECTATIONS:")
    for model, comp in comparisons.items():
        gaps = comp["gaps"]
        underperforming = [cap for cap, g in gaps.items() if g <= -2]
        if underperforming:
            print(f"   - {model}: {', '.join(underperforming)}")

    # Find models that exceeded expectations
    print("\n5. MODELS EXCEEDING EXPECTATIONS:")
    for model, comp in comparisons.items():
        gaps = comp["gaps"]
        exceeding = [cap for cap, g in gaps.items() if g >= 2]
        if exceeding:
            print(f"   - {model}: {', '.join(exceeding)}")

    print()


def main():
    benchmark_dir = Path("./output/benchmark/trial4")
    capabilities_path = Path("./src/synthony/recommender/model_capabilities_v5.json")

    print(f"Loading benchmark results from: {benchmark_dir}")
    results = load_benchmark_results(benchmark_dir)
    print(f"Found {len(results)} models with benchmark data")

    print("Aggregating metrics...")
    metrics = aggregate_model_metrics(results)

    print(f"Loading model capabilities from: {capabilities_path}")
    capabilities = load_model_capabilities(capabilities_path)

    print("Comparing with assumptions...")
    comparisons = compare_with_assumptions(metrics, capabilities)

    print_analysis(metrics, comparisons)

    # Save detailed comparison to JSON
    output_path = benchmark_dir / "analysis_comparison.json"
    with open(output_path, "w") as f:
        # Convert ModelMetrics to dict for JSON serialization
        output_data = {
            "metrics": {
                model: {
                    "model_name": m.model_name,
                    "dataset_count": m.dataset_count,
                    "avg_quality_score": m.avg_quality_score,
                    "avg_fidelity": m.avg_fidelity,
                    "avg_utility": m.avg_utility,
                    "avg_privacy": m.avg_privacy,
                    "avg_kl_divergence": m.avg_kl_divergence,
                    "avg_js_divergence": m.avg_js_divergence,
                    "skew_preservation_rate": m.skew_preservation_rate,
                    "cardinality_preservation_rate": m.cardinality_preservation_rate,
                    "correlation_preservation_rate": m.correlation_preservation_rate,
                }
                for model, m in metrics.items()
            },
            "comparisons": comparisons,
        }
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
