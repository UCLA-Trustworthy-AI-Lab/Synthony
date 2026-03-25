"""
Accuracy regression tests for the recommendation engine.

Validates recommendations against benchmark ground truth from
evaluation_results_clean.csv (spark benchmarks, Column_Shape_Avg metric).

These tests ensure the engine produces reasonable recommendations —
the benchmark winner should appear in top-N, scores should differentiate
models, and constraints should be respected.
"""

import pytest
from pathlib import Path

from synthony import StochasticDataAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine

# Ground truth from spark benchmarks (Column_Shape_Avg metric)
GROUND_TRUTH = {
    "abalone": {"winner": "CART", "score": 0.916},
    "Bean": {"winner": "TVAE", "score": 0.852},
    "IndianLiverPatient": {"winner": "SMOTE", "score": 0.884},
    "Obesity": {"winner": "CART", "score": 0.874},
    "faults": {"winner": "SMOTE", "score": 0.949},
    "insurance": {"winner": "ARF", "score": 0.901},
    "wilt": {"winner": "CART", "score": 0.894},
}

DATA_DIR = Path("dataset/input_data")


def _get_available_datasets():
    """Return ground truth datasets that exist on disk."""
    available = {}
    for name, info in GROUND_TRUTH.items():
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            available[name] = info
    return available


AVAILABLE = _get_available_datasets()


@pytest.fixture(scope="module")
def analyzer():
    return StochasticDataAnalyzer()


@pytest.fixture(scope="module")
def engine():
    return ModelRecommendationEngine()


@pytest.fixture(scope="module")
def all_recommendations(analyzer, engine):
    """Run recommendations for all available datasets once."""
    results = {}
    for name in AVAILABLE:
        path = str(DATA_DIR / f"{name}.csv")
        profile = analyzer.analyze(path)
        result = engine.recommend(profile, method="rule_based", top_n=5)
        results[name] = result
    return results


@pytest.mark.skipif(len(AVAILABLE) == 0, reason="No benchmark datasets found")
class TestGroundTruthAccuracy:
    """Validate that benchmark winners appear in top-N recommendations."""

    def test_ground_truth_winner_in_top5(self, all_recommendations):
        """Benchmark winner should appear in top-5 for majority of datasets."""
        hits = 0
        total = len(AVAILABLE)
        details = []

        for name, info in AVAILABLE.items():
            result = all_recommendations[name]
            top5 = [result.recommended_model.model_name] + [
                a.model_name for a in result.alternative_models[:4]
            ]
            winner = info["winner"]
            found = winner in top5
            if found:
                hits += 1
            details.append(f"  {name}: gt={winner}, top5={top5}, hit={found}")

        hit_rate = hits / total if total > 0 else 0
        detail_str = "\n".join(details)
        assert hit_rate >= 0.5, (
            f"Ground truth hit rate {hits}/{total} ({hit_rate:.0%}) < 50%.\n"
            f"Details:\n{detail_str}"
        )

    def test_primary_recommendation_is_valid_model(self, all_recommendations, engine):
        """Primary recommendation must be a model in the registry."""
        for name, result in all_recommendations.items():
            model_name = result.recommended_model.model_name
            assert model_name in engine.models, (
                f"{name}: recommended '{model_name}' not in registry"
            )


@pytest.mark.skipif(len(AVAILABLE) == 0, reason="No benchmark datasets found")
class TestScoreDifferentiation:
    """Ensure scoring produces meaningful differentiation (no collapse)."""

    def test_no_score_collapse(self, analyzer, engine):
        """Models should NOT all score identically on any dataset."""
        for name in AVAILABLE:
            path = str(DATA_DIR / f"{name}.csv")
            profile = analyzer.analyze(path)

            eligible, _ = engine._apply_hard_filters(
                {"dataset_rows": profile.row_count}
            )
            required = engine._calculate_required_capabilities(profile, None)
            scored = engine._score_models(eligible, required)

            scores = [m["total_score"] for m in scored]
            unique_scores = len(set(round(s, 6) for s in scores))
            assert unique_scores > 1, (
                f"{name}: all {len(scores)} models scored identically "
                f"(score={scores[0]:.4f}) — scoring collapse detected"
            )

    def test_confidence_scores_are_reasonable(self, all_recommendations):
        """Confidence scores should be > 0 and <= 1.0."""
        for name, result in all_recommendations.items():
            conf = result.recommended_model.confidence_score
            assert 0.0 < conf <= 1.0, (
                f"{name}: confidence {conf} out of range"
            )

    def test_reasoning_is_non_empty(self, all_recommendations):
        """Every recommendation should include reasoning."""
        for name, result in all_recommendations.items():
            assert len(result.recommended_model.reasoning) > 0, (
                f"{name}: no reasoning provided"
            )


@pytest.mark.skipif(len(AVAILABLE) == 0, reason="No benchmark datasets found")
class TestConstraintAccuracy:
    """Validate constraint handling against ground truth datasets."""

    def test_cpu_only_excludes_gpu_models(self, analyzer, engine):
        """CPU-only constraint should never recommend GPU-requiring models."""
        gpu_models = {
            name for name, info in engine.models.items()
            if not info["constraints"].get("cpu_only_compatible", True)
        }

        for ds_name in list(AVAILABLE.keys())[:3]:  # Test on first 3
            path = str(DATA_DIR / f"{ds_name}.csv")
            profile = analyzer.analyze(path)
            result = engine.recommend(
                profile,
                method="rule_based",
                constraints={"cpu_only": True},
                top_n=5,
            )
            primary = result.recommended_model.model_name
            assert primary not in gpu_models, (
                f"{ds_name}: CPU-only recommended GPU model '{primary}'"
            )
            for alt in result.alternative_models:
                assert alt.model_name not in gpu_models, (
                    f"{ds_name}: CPU-only alternative '{alt.model_name}' requires GPU"
                )

    def test_strict_dp_only_recommends_dp_models(self, analyzer, engine):
        """Strict DP constraint should only recommend DP-capable models."""
        dp_threshold = engine.config.dp_min_score
        dp_models = {
            name for name, info in engine.models.items()
            if info["capabilities"]["privacy_dp"] >= dp_threshold
        }

        for ds_name in list(AVAILABLE.keys())[:3]:
            path = str(DATA_DIR / f"{ds_name}.csv")
            profile = analyzer.analyze(path)
            result = engine.recommend(
                profile,
                method="rule_based",
                constraints={"strict_dp": True},
                top_n=3,
            )
            primary = result.recommended_model.model_name
            assert primary in dp_models, (
                f"{ds_name}: strict_dp recommended non-DP model '{primary}'"
            )


@pytest.mark.skipif(len(AVAILABLE) == 0, reason="No benchmark datasets found")
class TestConfigFromRegistry:
    """Verify engine config is loaded from registry, not hardcoded defaults."""

    def test_dp_threshold_from_registry(self, engine):
        """DP threshold should match registry metadata."""
        registry_dp = engine.registry.get("metadata", {}).get("dp_threshold")
        if registry_dp is not None:
            assert engine.config.dp_min_score == int(registry_dp)

    def test_capability_thresholds_from_registry(self, engine):
        """Capability thresholds should match registry metadata."""
        cap_thresholds = engine.registry.get("metadata", {}).get(
            "capability_thresholds", {}
        )
        for key, value in cap_thresholds.items():
            if hasattr(engine.config, key):
                assert getattr(engine.config, key) == type(
                    getattr(engine.config, key)
                )(value), f"Config {key} doesn't match registry"

    def test_score_decay_from_registry(self, engine):
        """Score decay curve should match registry metadata."""
        decay = engine.registry.get("metadata", {}).get("score_decay", {})
        for key, value in decay.items():
            attr = f"score_decay_{key}"
            if hasattr(engine.config, attr):
                assert getattr(engine.config, attr) == float(value), (
                    f"Config {attr} doesn't match registry"
                )

    def test_hard_problem_confidence_from_registry(self, engine):
        """Hard problem confidence scores should match registry metadata."""
        hp_conf = engine.registry.get("metadata", {}).get(
            "hard_problem_confidence", {}
        )
        for key, value in hp_conf.items():
            attr = f"hard_problem_confidence_{key}"
            if hasattr(engine.config, attr):
                assert getattr(engine.config, attr) == float(value), (
                    f"Config {attr} doesn't match registry"
                )

    def test_hard_problem_routing_from_registry(self, engine):
        """Hard problem routing should match registry."""
        hp_routing = engine.registry.get("hard_problem_routing", {})
        if hp_routing:
            if "primary" in hp_routing:
                assert engine.config.hard_problem_primary == hp_routing["primary"]
            if "large_data_fallback" in hp_routing:
                assert (
                    engine.config.hard_problem_large_data_fallback
                    == hp_routing["large_data_fallback"]
                )
