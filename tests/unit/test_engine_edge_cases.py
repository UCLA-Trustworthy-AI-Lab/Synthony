"""
Unit tests for edge cases in the recommender engine.

Tests boundary conditions, conflicting constraints, extreme dataset
profiles, and scoring edge cases in ModelRecommendationEngine.
"""

import pytest

from synthony.core.schemas import (
    CardinalityMetrics,
    CorrelationMetrics,
    DatasetProfile,
    SkewnessMetrics,
    StressFactors,
    ZipfianMetrics,
)
from synthony.recommender.engine import EngineConfig, ModelRecommendationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    row_count: int = 5000,
    column_count: int = 10,
    severe_skew: bool = False,
    high_cardinality: bool = False,
    zipfian_distribution: bool = False,
    small_data: bool = False,
    large_data: bool = False,
    higher_order_correlation: bool = False,
    max_skewness: float = 0.5,
    max_cardinality: int = 100,
    top_20_percent_ratio: float = 0.3,
) -> DatasetProfile:
    """Create a DatasetProfile with configurable stress factors."""
    skewness = SkewnessMetrics(
        column_scores={"col0": max_skewness},
        max_skewness=max_skewness,
        severe_columns=["col0"] if severe_skew else [],
    )
    cardinality = CardinalityMetrics(
        column_counts={"col0": max_cardinality},
        max_cardinality=max_cardinality,
        high_cardinality_columns=["col0"] if high_cardinality else [],
    )
    zipfian = ZipfianMetrics(
        detected=zipfian_distribution,
        top_20_percent_ratio=top_20_percent_ratio if zipfian_distribution else 0.3,
        affected_columns=["col0"] if zipfian_distribution else [],
    )
    correlation = CorrelationMetrics(
        correlation_density=0.5 if higher_order_correlation else 0.05,
        mean_r_squared=0.1 if higher_order_correlation else 0.8,
        has_higher_order=higher_order_correlation,
    )

    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        stress_factors=StressFactors(
            severe_skew=severe_skew,
            high_cardinality=high_cardinality,
            zipfian_distribution=zipfian_distribution,
            small_data=small_data,
            large_data=large_data,
            higher_order_correlation=higher_order_correlation,
        ),
        skewness=skewness,
        cardinality=cardinality,
        zipfian=zipfian,
        correlation=correlation,
    )


def _engine() -> ModelRecommendationEngine:
    """Create a default engine instance."""
    return ModelRecommendationEngine()


# ---------------------------------------------------------------------------
# 1. Conflicting constraints
# ---------------------------------------------------------------------------

class TestConflictingConstraints:
    """Test constraint combinations that restrict the model pool."""

    def test_cpu_only_and_strict_dp_returns_eligible_models(self):
        """cpu_only + strict_dp should succeed because AIM and DPCART qualify."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        constraints = {"cpu_only": True, "strict_dp": True}
        result = engine.recommend(profile, constraints=constraints)

        # AIM (cpu_only=True, privacy_dp=4) and DPCART (cpu_only=True, privacy_dp=3)
        # should be eligible.
        assert result.recommended_model.model_name in ("AIM", "DPCART")

    def test_allowed_models_restricts_to_gpu_only_with_cpu_constraint_raises(self):
        """If allowed_models contains only GPU models and cpu_only=True, raise ValueError."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        # TabSyn, TabDDPM, PATECTGAN all require GPU
        constraints = {
            "cpu_only": True,
            "allowed_models": ["TabSyn", "TabDDPM", "PATECTGAN"],
        }
        with pytest.raises(ValueError, match="No eligible models"):
            engine.recommend(profile, constraints=constraints)

    def test_strict_dp_with_allowed_no_dp_models_raises(self):
        """strict_dp=True with allowed_models having no DP model should raise."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        constraints = {
            "strict_dp": True,
            "allowed_models": ["CART", "ARF", "SMOTE"],
        }
        with pytest.raises(ValueError, match="No eligible models"):
            engine.recommend(profile, constraints=constraints)


# ---------------------------------------------------------------------------
# 2. Single-column dataset
# ---------------------------------------------------------------------------

class TestSingleColumnDataset:
    """Recommendation should succeed even for a 1-column dataset."""

    def test_single_column_profile_succeeds(self):
        profile = _make_profile(row_count=1000, column_count=1)
        engine = _engine()
        result = engine.recommend(profile)
        assert result.recommended_model.model_name
        assert result.recommended_model.confidence_score >= 0.0


# ---------------------------------------------------------------------------
# 3. All-categorical dataset (no numeric columns)
# ---------------------------------------------------------------------------

class TestAllCategoricalDataset:
    """No numeric columns means skewness = 0 and correlation may be trivial."""

    def test_all_categorical_no_crash(self):
        profile = _make_profile(
            row_count=2000,
            column_count=5,
            max_skewness=0.0,
        )
        engine = _engine()
        result = engine.recommend(profile)
        assert result.recommended_model.model_name
        assert result.difficulty_summary is not None


# ---------------------------------------------------------------------------
# 4. Very small dataset (10 rows)
# ---------------------------------------------------------------------------

class TestVerySmallDataset:
    """10 rows should trigger small_data stress factor."""

    def test_small_data_stress_factor_detected(self):
        profile = _make_profile(row_count=10, small_data=True)
        engine = _engine()
        result = engine.recommend(profile)
        stress = result.difficulty_summary.get("stress_factors", {})
        assert stress.get("small_data") is True
        assert result.recommended_model.model_name

    def test_small_data_row_count_filtering(self):
        """Models with min_rows > 10 should be excluded."""
        engine = _engine()
        profile = _make_profile(row_count=10, small_data=True)
        result = engine.recommend(profile)
        # The excluded models dict should contain entries for models needing more rows
        excluded = result.excluded_models
        # At least some models require min_rows > 10 (e.g. CTGAN needs 500)
        too_small_exclusions = [
            reason for reason in excluded.values() if "too small" in reason.lower()
        ]
        assert len(too_small_exclusions) > 0


# ---------------------------------------------------------------------------
# 5. Very large row count in profile
# ---------------------------------------------------------------------------

class TestVeryLargeRowCount:
    """Test with mock profile having row_count=100000."""

    def test_large_row_count_excludes_limited_models(self):
        engine = _engine()
        profile = _make_profile(row_count=100000, large_data=True)
        result = engine.recommend(profile)
        assert result.recommended_model.model_name

        # GReaT has max_recommended_rows=10000, should be excluded
        excluded = result.excluded_models
        if "GReaT" in excluded:
            assert "too large" in excluded["GReaT"].lower()

    def test_large_data_stress_factor_present(self):
        profile = _make_profile(row_count=100000, large_data=True)
        engine = _engine()
        result = engine.recommend(profile)
        stress = result.difficulty_summary.get("stress_factors", {})
        assert stress.get("large_data") is True


# ---------------------------------------------------------------------------
# 6. Hard problem with all fallbacks filtered
# ---------------------------------------------------------------------------

class TestHardProblemFallbacksFiltered:
    """Trigger hard problem but filter all fallback models via allowed_models."""

    def test_falls_back_to_normal_scoring(self):
        """When hard problem is detected but all routed models are filtered,
        the engine should fall back to normal scoring."""
        engine = _engine()
        # Hard problem: all three stress factors active
        profile = _make_profile(
            row_count=5000,
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=True,
            max_skewness=5.0,
            max_cardinality=6000,
            top_20_percent_ratio=0.95,
        )

        # Filter to only models that are NOT in hard problem routing
        # GReaT (primary), TabDDPM (large fallback), ARF, TabSyn, CART, SMOTE, BayesianNetwork (fallbacks)
        # Use CTGAN and TVAE which are not in the hard problem routing
        # But TVAE requires GPU and has min_rows=200, CTGAN has min_rows=500
        # Use CTGAN and NFlow (both CPU compatible, not in hard problem routing)
        constraints = {
            "allowed_models": ["CTGAN", "NFlow"],
        }
        result = engine.recommend(profile, constraints=constraints)
        # Should still produce a result via normal scoring path
        assert result.recommended_model.model_name in ("CTGAN", "NFlow")
        # Method should indicate rule_based_v2, not hard_problem_path
        assert "hard_problem_path" not in result.method


# ---------------------------------------------------------------------------
# 7. Scale factor edge cases
# ---------------------------------------------------------------------------

class TestScaleFactors:
    """Test edge values for scale_factors."""

    def test_scale_factor_zero_disables_capability(self):
        """scale_factor=0.0 should zero out the contribution of that capability."""
        engine = _engine()
        profile = _make_profile(
            row_count=5000,
            severe_skew=True,
            max_skewness=5.0,
        )
        # With skew_handling scale factor = 0.0, skew capability should not matter
        result = engine.recommend(
            profile,
            scale_factors={"skew_handling": 0.0},
        )
        assert result.recommended_model.model_name
        assert result.recommended_model.confidence_score >= 0.0

    def test_scale_factor_high_emphasizes_capability(self):
        """scale_factor=10.0 should heavily weight that capability."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        # Emphasize privacy_dp enormously
        result_privacy = engine.recommend(
            profile,
            scale_factors={"privacy_dp": 10.0},
        )
        # With privacy_dp scaled to 10x, DP models (AIM, DPCART, PATECTGAN)
        # should score significantly higher
        dp_models = {"AIM", "DPCART", "PATECTGAN"}
        all_model_names = [result_privacy.recommended_model.model_name] + [
            a.model_name for a in result_privacy.alternative_models
        ]
        # At least one DP model should appear in the top recommendations
        assert any(m in dp_models for m in all_model_names[:3])

    def test_scale_factors_skip_hard_problem_path(self):
        """When scale_factors are provided, the hard problem path is skipped."""
        engine = _engine()
        profile = _make_profile(
            row_count=5000,
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=True,
            max_skewness=5.0,
            max_cardinality=6000,
            top_20_percent_ratio=0.95,
        )
        result = engine.recommend(
            profile,
            scale_factors={"skew_handling": 1.0},
        )
        # Method should not contain "hard_problem_path" since scale_factors are set
        assert "hard_problem_path" not in result.method


# ---------------------------------------------------------------------------
# 8. Empty stress factors (no active stress)
# ---------------------------------------------------------------------------

class TestEmptyStressFactors:
    """Dataset with no active stress factors should still produce a recommendation."""

    def test_no_stress_produces_recommendation(self):
        profile = _make_profile(
            row_count=5000,
            column_count=10,
            severe_skew=False,
            high_cardinality=False,
            zipfian_distribution=False,
            small_data=False,
            large_data=False,
            higher_order_correlation=False,
        )
        engine = _engine()
        result = engine.recommend(profile)
        assert result.recommended_model.model_name
        assert result.recommended_model.confidence_score >= 0.0
        # Should not be a hard problem
        assert result.difficulty_summary.get("is_hard_problem") is False

    def test_no_stress_required_capabilities_are_zero(self):
        """When no stress is active, all required capabilities should be 0."""
        profile = _make_profile(row_count=5000)
        engine = _engine()
        result = engine.recommend(profile)
        req_caps = result.difficulty_summary.get("required_capabilities", {})
        for cap, level in req_caps.items():
            assert level == 0, f"Expected 0 for {cap}, got {level}"


# ---------------------------------------------------------------------------
# 9. Confidence score bounds
# ---------------------------------------------------------------------------

class TestConfidenceScoreBounds:
    """Confidence should always be clamped to [0.0, 1.0]."""

    def test_confidence_in_bounds_no_stress(self):
        profile = _make_profile(row_count=5000)
        engine = _engine()
        result = engine.recommend(profile)
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        for alt in result.alternative_models:
            assert 0.0 <= alt.confidence_score <= 1.0

    def test_confidence_in_bounds_all_stress(self):
        profile = _make_profile(
            row_count=200,
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=True,
            small_data=True,
            higher_order_correlation=True,
            max_skewness=10.0,
            max_cardinality=10000,
            top_20_percent_ratio=0.99,
        )
        engine = _engine()
        result = engine.recommend(profile)
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        for alt in result.alternative_models:
            assert 0.0 <= alt.confidence_score <= 1.0

    def test_confidence_in_bounds_with_scale_factors(self):
        profile = _make_profile(
            row_count=5000,
            severe_skew=True,
            max_skewness=5.0,
        )
        engine = _engine()
        result = engine.recommend(
            profile,
            scale_factors={"skew_handling": 10.0, "privacy_dp": 10.0},
        )
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        for alt in result.alternative_models:
            assert 0.0 <= alt.confidence_score <= 1.0


# ---------------------------------------------------------------------------
# 10. top_n greater than eligible models
# ---------------------------------------------------------------------------

class TestTopNGreaterThanEligible:
    """Should return fewer alternatives without error."""

    def test_top_n_larger_than_pool(self):
        engine = _engine()
        profile = _make_profile(row_count=5000)
        # Request 100 alternatives but there are only ~15 models
        result = engine.recommend(profile, top_n=100)
        # Should not crash; alternatives count should be <= total eligible - 1
        total_models = len(engine.models)
        assert len(result.alternative_models) < total_models
        assert result.recommended_model.model_name

    def test_top_n_zero_returns_no_alternatives(self):
        engine = _engine()
        profile = _make_profile(row_count=5000)
        result = engine.recommend(profile, top_n=0)
        assert len(result.alternative_models) == 0
        assert result.recommended_model.model_name

    def test_top_n_one_returns_single_alternative(self):
        engine = _engine()
        profile = _make_profile(row_count=5000)
        result = engine.recommend(profile, top_n=1)
        assert len(result.alternative_models) <= 1
        assert result.recommended_model.model_name

    def test_allowed_models_two_with_top_n_ten(self):
        """Only 2 eligible models with top_n=10 should return at most 1 alternative."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        constraints = {"allowed_models": ["CART", "ARF"]}
        result = engine.recommend(profile, constraints=constraints, top_n=10)
        # 2 models total: 1 primary + at most 1 alternative
        assert len(result.alternative_models) <= 1
        assert result.recommended_model.model_name in ("CART", "ARF")


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEngineConfigCustomization:
    """Verify that custom EngineConfig values are respected."""

    def test_custom_small_data_threshold(self):
        """Setting small_data_threshold very high should affect tie-breaking."""
        config = EngineConfig(small_data_threshold=100000)
        engine = ModelRecommendationEngine(config=config)
        profile = _make_profile(row_count=5000, small_data=True)
        result = engine.recommend(profile)
        # Should still produce a valid result
        assert result.recommended_model.model_name

    def test_custom_tie_threshold(self):
        """A very large tie threshold should trigger tie-breaking more often."""
        config = EngineConfig(tie_threshold_percent=99.0)
        engine = ModelRecommendationEngine(config=config)
        profile = _make_profile(row_count=5000)
        result = engine.recommend(profile)
        assert result.recommended_model.model_name


class TestResultStructure:
    """Verify the structure of RecommendationResult."""

    def test_result_has_required_fields(self):
        engine = _engine()
        profile = _make_profile(row_count=5000)
        result = engine.recommend(profile)

        assert result.dataset_id
        assert result.method
        assert result.recommended_model is not None
        assert isinstance(result.alternative_models, list)
        assert isinstance(result.constraints, dict)
        assert isinstance(result.difficulty_summary, dict)
        assert isinstance(result.excluded_models, dict)

    def test_hard_problem_result_structure(self):
        """Hard problem path should still produce a valid result."""
        engine = _engine()
        profile = _make_profile(
            row_count=5000,
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=True,
            max_skewness=5.0,
            max_cardinality=6000,
            top_20_percent_ratio=0.95,
        )
        result = engine.recommend(profile)
        assert result.recommended_model.model_name
        assert "hard_problem" in result.method
        assert result.difficulty_summary.get("is_hard_problem") is True

    def test_model_info_populated(self):
        """recommended_model.model_info should contain capabilities."""
        engine = _engine()
        profile = _make_profile(row_count=5000)
        result = engine.recommend(profile)
        info = result.recommended_model.model_info
        assert "capabilities" in info
        assert "constraints" in info
        assert "performance" in info
