#!/usr/bin/env python3
"""
Generate model_capabilities.json from benchmark results.

This script scans benchmark JSON files and calculates capability scores
based on the methodology defined in docs/scoring_methodology.md.

Usage:
    python scripts/generate_model_capabilities.py [--benchmark-dir DIR] [--output FILE]
"""

import json
import argparse
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np


# Model metadata (static properties not derived from benchmarks)
MODEL_METADATA = {
    "Identity": {
        "full_name": "Identity Baseline",
        "type": "Baseline",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 1,
        "max_rows": 1000000,
        "best_for": ["Testing", "Baselines"],
    },
    "TVAE": {
        "full_name": "Tabular Variational Autoencoder",
        "type": "VAE",
        "package": "table-synthesizers",
        "requires_gpu": True,
        "privacy_dp": 0,
        "min_rows": 200,
        "max_rows": 100000,
        "best_for": ["Mixed data types", "General use"],
    },
    "SMOTE": {
        "full_name": "Synthetic Minority Oversampling Technique",
        "type": "Statistical",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 10,
        "max_rows": 100000,
        "best_for": ["Imbalanced datasets"],
    },
    "CART": {
        "full_name": "Classification and Regression Trees",
        "type": "Tree-based",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 50,
        "max_rows": 100000,
        "best_for": ["Interpretable models"],
    },
    "DPCART": {
        "full_name": "Differentially Private CART",
        "type": "Tree-based + DP",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 3,
        "min_rows": 500,
        "max_rows": 100000,
        "best_for": ["Privacy + interpretability"],
    },
    "AIM": {
        "full_name": "Adaptive and Iterative Mechanism",
        "type": "Statistical + DP",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 4,
        "min_rows": 1000,
        "max_rows": 100000,
        "best_for": ["Privacy-sensitive applications"],
    },
    "TabDDPM": {
        "full_name": "Tabular Denoising Diffusion Probabilistic Model",
        "type": "Diffusion",
        "package": "table-synthesizers",
        "requires_gpu": True,
        "privacy_dp": 0,
        "min_rows": 1000,
        "max_rows": 100000,
        "best_for": ["High-quality generation"],
    },
    "AutoDiff": {
        "full_name": "VAE + Diffusion Hybrid",
        "type": "Diffusion",
        "package": "table-synthesizers",
        "requires_gpu": True,
        "privacy_dp": 0,
        "min_rows": 1000,
        "max_rows": 100000,
        "best_for": ["Complex data distributions"],
    },
    "TabSyn": {
        "full_name": "Advanced Tabular Synthesis",
        "type": "Diffusion",
        "package": "table-synthesizers",
        "requires_gpu": True,
        "privacy_dp": 0,
        "min_rows": 1000,
        "max_rows": 100000,
        "best_for": ["Fast generation"],
    },
    "CTGAN": {
        "full_name": "Conditional Tabular GAN",
        "type": "GAN",
        "package": "table-synthesizers",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 500,
        "max_rows": 100000,
        "best_for": ["Large datasets"],
    },
    "PATECTGAN": {
        "full_name": "Privacy-Aware CTGAN with PATE",
        "type": "GAN + DP",
        "package": "table-synthesizers",
        "requires_gpu": True,
        "privacy_dp": 4,
        "min_rows": 1000,
        "max_rows": 100000,
        "best_for": ["Privacy-sensitive applications"],
    },
    "ARF": {
        "full_name": "Adversarial Random Forest",
        "type": "Tree-based",
        "package": "synthcity",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 50,
        "max_rows": 100000,
        "best_for": ["Tree-based synthesis"],
    },
    "NFlow": {
        "full_name": "Normalizing Flows",
        "type": "Flow",
        "package": "synthcity",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 200,
        "max_rows": 100000,
        "best_for": ["Probabilistic modeling"],
    },
    "BayesianNetwork": {
        "full_name": "Bayesian Network Synthesis",
        "type": "Statistical",
        "package": "synthcity",
        "requires_gpu": False,
        "privacy_dp": 0,
        "min_rows": 100,
        "max_rows": 50000,
        "best_for": ["Causal relationships"],
    },
    "GReaT": {
        "full_name": "Generation of Realistic Tabular data",
        "type": "LLM-based",
        "package": "synthcity",
        "requires_gpu": True,
        "privacy_dp": 0,
        "min_rows": 100,
        "max_rows": 10000,
        "best_for": ["Large language model approach"],
    },
}


def metric_to_score(value: float) -> int:
    """Convert 0-1 metric to 0-4 score."""
    if value >= 0.90:
        return 4
    elif value >= 0.75:
        return 3
    elif value >= 0.50:
        return 2
    elif value >= 0.25:
        return 1
    else:
        return 0


def calculate_skew_handling(benchmark: dict) -> Optional[int]:
    """Calculate skew handling capability from benchmark."""
    profile = benchmark.get("profile_comparison", {})
    original_skew = profile.get("skewness", {}).get("original", {})
    synthetic_skew = profile.get("skewness", {}).get("synthetic", {})
    
    if not original_skew or not synthetic_skew:
        return None
    
    # Only evaluate columns with |original_skew| > 2.0
    skew_scores = []
    for col, orig_val in original_skew.items():
        if abs(orig_val) > 2.0:
            synth_val = synthetic_skew.get(col, 0)
            # Preservation ratio (closer to original is better)
            preservation = 1 - abs(orig_val - synth_val) / max(abs(orig_val), 0.01)
            skew_scores.append(max(0, min(1, preservation)))
    
    if not skew_scores:
        return None
    
    return metric_to_score(sum(skew_scores) / len(skew_scores))


def calculate_cardinality_handling(benchmark: dict) -> Optional[int]:
    """Calculate cardinality handling capability from benchmark.

    Uses cardinality density preservation (unique/rows ratio) to avoid
    bias when synthetic dataset has fewer rows than original.
    """
    profile = benchmark.get("profile_comparison", {})
    original_card = profile.get("cardinality", {}).get("original", {})
    synthetic_card = profile.get("cardinality", {}).get("synthetic", {})

    if not original_card or not synthetic_card:
        return None

    orig_rows = benchmark.get("original_rows", 1)
    synth_rows = benchmark.get("synthetic_rows", 1)

    # Only evaluate columns with original cardinality > 500
    card_scores = []
    for col, orig_val in original_card.items():
        if orig_val > 500:
            synth_val = synthetic_card.get(col, 0)
            # Normalize by row count to remove sampling bias
            orig_density = orig_val / max(orig_rows, 1)
            synth_density = synth_val / max(synth_rows, 1)
            ratio = min(synth_density / max(orig_density, 1e-10), 1.0)
            card_scores.append(ratio)

    if not card_scores:
        return None

    return metric_to_score(sum(card_scores) / len(card_scores))


def calculate_correlation_handling(benchmark: dict) -> Optional[int]:
    """Calculate correlation handling capability from benchmark."""
    profile = benchmark.get("profile_comparison", {})
    orig_corr = profile.get("correlation", {}).get("original", {})
    synth_corr = profile.get("correlation", {}).get("synthetic", {})
    
    orig_r2 = orig_corr.get("mean_r_squared", 0)
    synth_r2 = synth_corr.get("mean_r_squared", 0)
    
    if orig_r2 == 0:
        return None
    
    preservation = synth_r2 / orig_r2
    
    # Also use fidelity.correlation_preservation if available
    fidelity_corr = benchmark.get("fidelity", {}).get("correlation_preservation")
    if fidelity_corr:
        preservation = (preservation + fidelity_corr) / 2
    
    return metric_to_score(min(preservation, 1.0))


def calculate_small_data_score(benchmark: dict) -> Optional[int]:
    """Calculate small data handling capability from benchmark."""
    rows = benchmark.get("original_rows", 0)
    
    if rows >= 1000:
        return None  # Not a small data benchmark
    
    quality = benchmark.get("overall_quality_score", 0)
    return metric_to_score(quality)


def calculate_overall_quality(benchmark: dict) -> Optional[int]:
    """Calculate overall quality score."""
    quality = benchmark.get("overall_quality_score", 0)
    return metric_to_score(quality)


def extract_model_name(filename: str) -> Optional[str]:
    """Extract model name from benchmark filename."""
    # Pattern: benchmark__dataset_ModelName.json or benchmark__dataset__ModelName.json
    name = filename.replace(".json", "")
    
    # Special case for identity (lowercase in filename)
    if "identity" in name.lower():
        return "Identity"
    
    # Split by underscores and check each part
    parts = name.split("_")
    
    # Find model name (usually last part)
    for part in reversed(parts):
        if part and part in MODEL_METADATA:
            return part
    
    return None


def aggregate_scores(scores: list[Optional[int]]) -> int:
    """Aggregate scores from multiple benchmarks."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return 2  # Default to moderate if no data
    return round(sum(valid) / len(valid))


def calculate_empirical_stats(benchmarks: list[dict]) -> dict:
    """Calculate empirical statistics from benchmarks for a model."""

    quality_scores = [b.get("overall_quality_score", 0) for b in benchmarks]
    fidelity_scores = [b.get("fidelity", {}).get("overall_fidelity", 0) for b in benchmarks]
    utility_scores = [b.get("utility", {}).get("overall_utility", 0) for b in benchmarks]

    # Skew preservation raw rates
    skew_rates = []
    for b in benchmarks:
        profile = b.get("profile_comparison", {})
        orig_skew = profile.get("skewness", {}).get("original", {})
        synth_skew = profile.get("skewness", {}).get("synthetic", {})
        for col, orig_val in orig_skew.items():
            if abs(orig_val) > 2.0:
                synth_val = synth_skew.get(col, 0)
                pres = 1 - abs(orig_val - synth_val) / max(abs(orig_val), 0.01)
                skew_rates.append(max(0, min(1, pres)))

    # Cardinality density preservation rates
    card_rates = []
    for b in benchmarks:
        profile = b.get("profile_comparison", {})
        orig_card = profile.get("cardinality", {}).get("original", {})
        synth_card = profile.get("cardinality", {}).get("synthetic", {})
        orig_rows = b.get("original_rows", 1)
        synth_rows = b.get("synthetic_rows", 1)
        for col, orig_val in orig_card.items():
            if orig_val > 500:
                synth_val = synth_card.get(col, 0)
                orig_density = orig_val / max(orig_rows, 1)
                synth_density = synth_val / max(synth_rows, 1)
                ratio = min(synth_density / max(orig_density, 1e-10), 1.0)
                card_rates.append(ratio)

    # Correlation preservation rates
    corr_rates = []
    for b in benchmarks:
        profile = b.get("profile_comparison", {})
        orig_r2 = profile.get("correlation", {}).get("original", {}).get("mean_r_squared", 0)
        synth_r2 = profile.get("correlation", {}).get("synthetic", {}).get("mean_r_squared", 0)
        if orig_r2 > 0:
            pres = synth_r2 / orig_r2
            fid_corr = b.get("fidelity", {}).get("correlation_preservation")
            if fid_corr:
                pres = (pres + fid_corr) / 2
            corr_rates.append(min(pres, 1.0))

    return {
        "avg_quality_score": round(float(np.mean(quality_scores)), 3) if quality_scores else 0.0,
        "avg_fidelity": round(float(np.mean(fidelity_scores)), 3) if fidelity_scores else 0.0,
        "avg_utility": round(float(np.mean(utility_scores)), 3) if utility_scores else 0.0,
        "skew_preservation": round(float(np.mean(skew_rates)), 3) if skew_rates else 0.0,
        "cardinality_preservation": round(float(np.mean(card_rates)), 3) if card_rates else 0.0,
        "correlation_preservation": round(float(np.mean(corr_rates)), 3) if corr_rates else 0.0,
        "datasets_tested": len(benchmarks),
    }


def generate_capabilities(benchmark_dir: Path, source_label: str = "spark") -> dict:
    """Generate model capabilities from benchmark files."""

    # Collect benchmarks by model
    model_benchmarks: dict[str, list[dict]] = {}

    for file in benchmark_dir.glob("benchmark__*.json"):
        if file.name.startswith("._"):
            continue  # Skip macOS metadata files

        model_name = extract_model_name(file.name)
        if not model_name:
            print(f"Warning: Could not extract model name from {file.name}")
            continue

        try:
            with open(file) as f:
                benchmark = json.load(f)

            if model_name not in model_benchmarks:
                model_benchmarks[model_name] = []
            model_benchmarks[model_name].append(benchmark)
            print(f"Loaded benchmark for {model_name}: {file.name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Load existing v6 config to preserve metadata fields
    v6_config_path = Path("config/model_capabilities.json")
    v6_models = {}
    if v6_config_path.exists():
        with open(v6_config_path) as f:
            v6_data = json.load(f)
        v6_models = v6_data.get("models", {})

    # Calculate capabilities for each model
    models = {}

    for model_name, benchmarks in model_benchmarks.items():
        if model_name not in MODEL_METADATA:
            print(f"Warning: No metadata for model {model_name}")
            continue

        metadata = MODEL_METADATA[model_name]

        # Calculate each capability from all benchmarks
        skew_scores = [calculate_skew_handling(b) for b in benchmarks]
        card_scores = [calculate_cardinality_handling(b) for b in benchmarks]
        corr_scores = [calculate_correlation_handling(b) for b in benchmarks]
        small_scores = [calculate_small_data_score(b) for b in benchmarks]
        quality_scores = [calculate_overall_quality(b) for b in benchmarks]

        # Aggregate scores
        capabilities = {
            "skew_handling": aggregate_scores(skew_scores),
            "cardinality_handling": aggregate_scores(card_scores),
            "zipfian_handling": 2,  # Default - needs specific benchmark
            "small_data": aggregate_scores(small_scores) if any(s is not None for s in small_scores) else 2,
            "correlation_handling": aggregate_scores(corr_scores),
            "privacy_dp": metadata["privacy_dp"],
        }

        # Preserve zipfian_handling from v6 if available
        v6_entry = v6_models.get(model_name, {})
        v6_caps = v6_entry.get("capabilities", {})
        if "zipfian_handling" in v6_caps:
            capabilities["zipfian_handling"] = v6_caps["zipfian_handling"]

        # Calculate empirical statistics
        empirical = calculate_empirical_stats(benchmarks)

        # Build model entry, preserving v6 metadata where available
        model_entry = {
            "name": model_name,
            "full_name": metadata["full_name"],
            "type": metadata["type"],
            "package": metadata["package"],
            "capabilities": capabilities,
            "capabilities_source": source_label,
            f"{source_label}_empirical": empirical,
            "constraints": {
                "requires_gpu": metadata["requires_gpu"],
                "cpu_only_compatible": not metadata["requires_gpu"],
                "max_recommended_rows": metadata["max_rows"],
                "min_rows": metadata["min_rows"],
            },
            "performance": v6_entry.get("performance", {
                "training_speed": "moderate",
                "inference_speed": "moderate",
                "memory_usage": "moderate",
            }),
            "best_for": metadata["best_for"],
            "benchmark_count": len(benchmarks),
        }

        # Preserve v6 descriptive fields or set defaults
        if v6_entry:
            if "class_path" in v6_entry:
                model_entry["class_path"] = v6_entry["class_path"]
            if "description" in v6_entry:
                model_entry["description"] = v6_entry["description"]
            if "strengths" in v6_entry:
                model_entry["strengths"] = v6_entry["strengths"]
            if "limitations" in v6_entry:
                model_entry["limitations"] = v6_entry["limitations"]
            if "exclude" in v6_entry:
                model_entry["exclude"] = v6_entry["exclude"]
            # Preserve trial4 empirical if it exists
            if "trial4_empirical" in v6_entry:
                model_entry["trial4_empirical"] = v6_entry["trial4_empirical"]

        # Ensure required fields always exist
        if "strengths" not in model_entry:
            model_entry["strengths"] = [f"{metadata['type']} model", f"Tested on {len(benchmarks)} datasets"]
        if "limitations" not in model_entry:
            model_entry["limitations"] = [f"Quality score: {empirical['avg_quality_score']:.3f}"]
        if "description" not in model_entry:
            model_entry["description"] = f"{metadata['full_name']} ({metadata['type']}). Benchmarked on {len(benchmarks)} datasets."
        if "exclude" not in model_entry:
            model_entry["exclude"] = False

        models[model_name] = model_entry

    # Add models without benchmarks (preserve from v6 or use defaults)
    for model_name, metadata in MODEL_METADATA.items():
        if model_name not in models:
            print(f"No benchmarks found for {model_name}, using v6 config or defaults")
            v6_entry = v6_models.get(model_name, {})
            if v6_entry:
                models[model_name] = v6_entry
                models[model_name]["capabilities_source"] = v6_entry.get("capabilities_source", "literature")
            else:
                models[model_name] = {
                    "name": model_name,
                    "full_name": metadata["full_name"],
                    "type": metadata["type"],
                    "package": metadata["package"],
                    "capabilities": {
                        "skew_handling": 2,
                        "cardinality_handling": 2,
                        "zipfian_handling": 2,
                        "small_data": 2,
                        "correlation_handling": 2,
                        "privacy_dp": metadata["privacy_dp"],
                    },
                    "capabilities_source": "default",
                    "constraints": {
                        "requires_gpu": metadata["requires_gpu"],
                        "cpu_only_compatible": not metadata["requires_gpu"],
                        "max_recommended_rows": metadata["max_rows"],
                        "min_rows": metadata["min_rows"],
                    },
                    "performance": {
                        "training_speed": "moderate",
                        "inference_speed": "moderate",
                        "memory_usage": "moderate",
                    },
                    "best_for": metadata["best_for"],
                    "benchmark_count": 0,
                }

    return {
        "metadata": {
            "version": "7.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "source": f"{source_label} benchmarks (10 datasets, 14 models) + v6.0.0 metadata. Cardinality uses density-normalized formula.",
            "description": "Model capabilities derived from empirical benchmark results with row-count-normalized cardinality scoring.",
            "compatible_models_count": len(models),
        },
        "models": models,
        "model_ranking_by_capability": generate_rankings(models),
        "tie_breaking_priority": {
            "small_data_priority": ["ARF", "CART", "BayesianNetwork", "SMOTE"],
            "speed_priority": ["CART", "ARF", "SMOTE", "TVAE", "DPCART"],
            "quality_priority": ["CART", "SMOTE", "BayesianNetwork", "ARF", "NFlow"],
        },
    }


def generate_rankings(models: dict) -> dict:
    """Generate model rankings by capability."""
    capabilities = ["skew_handling", "cardinality_handling", "zipfian_handling", 
                    "small_data", "correlation_handling", "privacy_dp"]
    
    rankings = {}
    for cap in capabilities:
        ranked = sorted(
            [(name, m["capabilities"].get(cap, 0)) for name, m in models.items()],
            key=lambda x: x[1],
            reverse=True
        )
        rankings[cap] = [name for name, score in ranked if score >= 3][:5]
    
    return rankings


def main():
    parser = argparse.ArgumentParser(description="Generate model capabilities from benchmarks")
    parser.add_argument(
        "--benchmark-dir", "-b",
        type=Path,
        default=Path("output/benchmark/spark"),
        help="Directory containing benchmark JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("config/model_capabilities.json"),
        help="Output file path"
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="spark",
        help="Label for the benchmark source (e.g., trial4, spark)"
    )
    
    args = parser.parse_args()
    
    if not args.benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {args.benchmark_dir}")
        return 1
    
    print(f"Scanning benchmarks in: {args.benchmark_dir}")
    capabilities = generate_capabilities(args.benchmark_dir, source_label=args.source_label)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(capabilities, f, indent=2)
    
    print(f"\n✓ Generated capabilities for {len(capabilities['models'])} models")
    print(f"✓ Output written to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
