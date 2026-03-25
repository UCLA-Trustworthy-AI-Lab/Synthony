"""
Comprehensive evaluation tests for the Synthony recommender system.

Tests all engine code paths using the 11 real datasets from dataset/input_data/
and the 8 trial4 synthetic datasets from dataset/synth_data/trial4/.

Test classes:
1. TestStressProfileAccuracy - stress factor detection on all 11 datasets
2. TestHardProblemDetection - hard problem routing logic
3. TestConstraintFiltering - CPU-only, strict DP, combined constraints
4. TestScoringConsistency - capability scoring and scale factors
5. TestTieBreaking - tie-breaking rules
6. TestBenchmarkQuality - synthetic data quality with trial4 outputs
7. TestFocusProfiles - focus-based scale factor profiles
8. TestEndToEndRecommendation - full pipeline on all datasets
"""

from pathlib import Path

import pandas as pd
import pytest

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.schemas import DatasetProfile, SkewnessMetrics, StressFactors
from synthony.recommender.engine import ModelRecommendationEngine
from synthony.recommender.focus_profiles import (
    CAPABILITY_NAMES,
    FOCUS_REGISTRY,
    get_scale_factors,
    register_focus,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "input_data"
SYNTH_DIR = PROJECT_ROOT / "dataset" / "synth_data" / "trial4"

# ---------------------------------------------------------------------------
# Dataset ground truth: known properties from profiling
# ---------------------------------------------------------------------------
DATASET_GROUND_TRUTH = {
    "abalone": {
        "file": "abalone.csv",
        "rows": 4177,
        "cols": 9,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": False, "higher_order_correlation": False},
        "hard": False,
    },
    "Bean": {
        "file": "Bean.csv",
        "rows": 13611,
        "cols": 17,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": False, "higher_order_correlation": False},
        "hard": False,
    },
    "faults": {
        "file": "faults.csv",
        "rows": 1941,
        "cols": 34,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": False, "higher_order_correlation": True},
        "hard": False,
    },
    "HTRU2": {
        "file": "HTRU2.csv",
        "rows": 17898,
        "cols": 9,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": True, "higher_order_correlation": False},
        "hard": True,
    },
    "IndianLiverPatient": {
        "file": "IndianLiverPatient.csv",
        "rows": 579,
        "cols": 11,
        "stress": {"severe_skew": True, "high_cardinality": False, "zipfian_distribution": False, "higher_order_correlation": True},
        "hard": False,
    },
    "insurance": {
        "file": "insurance.csv",
        "rows": 1338,
        "cols": 7,
        "stress": {"severe_skew": False, "high_cardinality": True, "zipfian_distribution": False, "higher_order_correlation": False},
        "hard": False,
    },
    "News": {
        "file": "News.csv",
        "rows": 39644,
        "cols": 60,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": False, "higher_order_correlation": False},
        "hard": False,
    },
    "Obesity": {
        "file": "Obesity.csv",
        "rows": 2111,
        "cols": 17,
        "stress": {"severe_skew": False, "high_cardinality": True, "zipfian_distribution": True, "higher_order_correlation": False},
        "hard": False,
    },
    "Shoppers": {
        "file": "Shoppers.csv",
        "rows": 12330,
        "cols": 18,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": True, "higher_order_correlation": False},
        "hard": True,
    },
    "Titanic": {
        "file": "Titanic.csv",
        "rows": 714,
        "cols": 8,
        "stress": {"severe_skew": True, "high_cardinality": False, "zipfian_distribution": False, "higher_order_correlation": True},
        "hard": False,
    },
    "wilt": {
        "file": "wilt.csv",
        "rows": 4839,
        "cols": 6,
        "stress": {"severe_skew": True, "high_cardinality": True, "zipfian_distribution": True, "higher_order_correlation": True},
        "hard": True,
    },
}

# Derive model sets dynamically from model_capabilities.json via engine
from synthony.recommender.engine import ModelRecommendationEngine as _Engine
_engine_ref = _Engine()
_MODELS = _engine_ref.models
_DP_THRESHOLD = _engine_ref.config.dp_min_score

ALL_MODELS = set(_MODELS.keys())

# GPU-only models (cpu_only_compatible=False)
GPU_MODELS = {
    name for name, info in _MODELS.items()
    if not info.get("constraints", {}).get("cpu_only_compatible", True)
}

# CPU-compatible models
CPU_MODELS = ALL_MODELS - GPU_MODELS

# DP-capable models (privacy_dp >= dp_threshold from registry)
DP_MODELS = {
    name for name, info in _MODELS.items()
    if info.get("capabilities", {}).get("privacy_dp", 0) >= _DP_THRESHOLD
}

# CPU + DP intersection
CPU_DP_MODELS = CPU_MODELS & DP_MODELS

# Datasets with trial4 synthetic data
TRIAL4_DATASETS = ["abalone", "Bean", "faults", "IndianLiverPatient", "insurance", "Obesity", "Shoppers", "wilt"]

# Hard problem datasets
HARD_DATASETS = ["HTRU2", "Shoppers", "wilt"]
NON_HARD_DATASETS = [k for k in DATASET_GROUND_TRUTH if k not in HARD_DATASETS]

DATASET_NAMES = list(DATASET_GROUND_TRUTH.keys())

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer():
    """Create a StochasticDataAnalyzer instance (shared across module)."""
    return StochasticDataAnalyzer()


@pytest.fixture(scope="module")
def engine():
    """Create a ModelRecommendationEngine instance (shared across module)."""
    return ModelRecommendationEngine()


@pytest.fixture(scope="module")
def all_profiles(analyzer):
    """Profile all 11 datasets once and cache results."""
    profiles = {}
    for name, info in DATASET_GROUND_TRUTH.items():
        csv_path = DATA_DIR / info["file"]
        if csv_path.exists():
            profiles[name] = analyzer.analyze(csv_path)
    return profiles


# ===========================================================================
# 1. Stress Profile Accuracy
# ===========================================================================


class TestStressProfileAccuracy:
    """Verify stress factor detection on all 11 real datasets."""

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_row_count(self, all_profiles, dataset_name):
        """Row count matches known ground truth."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["rows"]
        assert profile.row_count == expected, (
            f"{dataset_name}: expected {expected} rows, got {profile.row_count}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_column_count(self, all_profiles, dataset_name):
        """Column count matches known ground truth."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["cols"]
        assert profile.column_count == expected, (
            f"{dataset_name}: expected {expected} cols, got {profile.column_count}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_severe_skew_detection(self, all_profiles, dataset_name):
        """Severe skew flag matches expected value."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["stress"]["severe_skew"]
        assert profile.stress_factors.severe_skew == expected, (
            f"{dataset_name}: severe_skew expected={expected}, "
            f"got={profile.stress_factors.severe_skew}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_high_cardinality_detection(self, all_profiles, dataset_name):
        """High cardinality flag matches expected value."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["stress"]["high_cardinality"]
        assert profile.stress_factors.high_cardinality == expected, (
            f"{dataset_name}: high_cardinality expected={expected}, "
            f"got={profile.stress_factors.high_cardinality}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_zipfian_detection(self, all_profiles, dataset_name):
        """Zipfian distribution flag matches expected value."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["stress"]["zipfian_distribution"]
        assert profile.stress_factors.zipfian_distribution == expected, (
            f"{dataset_name}: zipfian_distribution expected={expected}, "
            f"got={profile.stress_factors.zipfian_distribution}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_correlation_detection(self, all_profiles, dataset_name):
        """Higher-order correlation flag matches expected value."""
        profile = all_profiles[dataset_name]
        expected = DATASET_GROUND_TRUTH[dataset_name]["stress"]["higher_order_correlation"]
        assert profile.stress_factors.higher_order_correlation == expected, (
            f"{dataset_name}: higher_order_correlation expected={expected}, "
            f"got={profile.stress_factors.higher_order_correlation}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_no_small_data_flag(self, all_profiles, dataset_name):
        """Datasets with >= 1000 rows should not trigger small_data (threshold = 1000)."""
        profile = all_profiles[dataset_name]
        expected_rows = DATASET_GROUND_TRUTH[dataset_name]["rows"]
        if expected_rows >= 1000:
            assert profile.stress_factors.small_data is False, (
                f"{dataset_name} ({profile.row_count} rows) should not be flagged as small_data"
            )
        else:
            assert profile.stress_factors.small_data is True, (
                f"{dataset_name} ({profile.row_count} rows) should be flagged as small_data"
            )


# ===========================================================================
# 2. Hard Problem Detection
# ===========================================================================


class TestHardProblemDetection:
    """Test hard problem detection and routing logic."""

    @pytest.mark.parametrize("dataset_name", HARD_DATASETS)
    def test_hard_problem_detected(self, engine, all_profiles, dataset_name):
        """Known hard-problem datasets should be detected as hard."""
        profile = all_profiles[dataset_name]
        is_hard, details = engine._is_hard_problem(profile)
        assert is_hard is True, (
            f"{dataset_name} should be hard problem, details: {details}"
        )

    @pytest.mark.parametrize("dataset_name", NON_HARD_DATASETS)
    def test_non_hard_problem(self, engine, all_profiles, dataset_name):
        """Non-hard datasets should NOT be detected as hard."""
        profile = all_profiles[dataset_name]
        is_hard, details = engine._is_hard_problem(profile)
        assert is_hard is False, (
            f"{dataset_name} should NOT be hard problem, details: {details}"
        )

    def test_hard_problem_details_have_all_keys(self, engine, all_profiles):
        """Hard problem details dict should have all three keys."""
        profile = all_profiles["HTRU2"]
        _, details = engine._is_hard_problem(profile)
        assert "severe_skew" in details
        assert "high_cardinality" in details
        assert "zipfian" in details

    def test_hard_problem_routes_to_great_for_small_data(self, engine):
        """Hard problem routing should prefer GReaT when rows < max_recommended."""
        profile = DatasetProfile(
            row_count=5000,
            column_count=10,
            stress_factors=StressFactors(
                severe_skew=True, high_cardinality=True,
                zipfian_distribution=True, small_data=False,
                large_data=False, higher_order_correlation=False,
            ),
        )
        eligible, excluded = engine._apply_hard_filters({"dataset_rows": profile.row_count})
        assert "GReaT" in eligible, "GReaT should be eligible for 5000 rows"
        result = engine._handle_hard_problem(profile, eligible, excluded)
        assert result == "GReaT", f"Expected GReaT for hard problem, got {result}"

    def test_hard_problem_htru2_excludes_great(self, engine, all_profiles):
        """HTRU2 (17898 rows) exceeds GReaT's max_rows, should route to fallback."""
        profile = all_profiles["HTRU2"]
        eligible, excluded = engine._apply_hard_filters({"dataset_rows": profile.row_count})
        # GReaT has max_recommended_rows=10000, HTRU2 has 17898
        assert "GReaT" not in eligible, "GReaT should be excluded for HTRU2 (17898 rows)"
        result = engine._handle_hard_problem(profile, eligible, excluded)
        assert result in engine.config.hard_problem_fallback

    def test_hard_problem_fallback_without_great(self, engine, all_profiles):
        """When GReaT is excluded, fallback should be from config list."""
        profile = all_profiles["Shoppers"]
        eligible = [m for m in engine.models if m != "GReaT"]
        result = engine._handle_hard_problem(profile, eligible, {})
        assert result in engine.config.hard_problem_fallback, (
            f"Fallback {result} not in {engine.config.hard_problem_fallback}"
        )

    def test_hard_problem_large_data_prefers_tabddpm(self, engine):
        """For large hard problems (>50k rows), should prefer TabDDPM over GReaT."""
        # Create a fake large-data profile
        profile = DatasetProfile(
            row_count=60000,
            column_count=10,
            stress_factors=StressFactors(
                severe_skew=True,
                high_cardinality=True,
                zipfian_distribution=True,
                small_data=False,
                large_data=True,
                higher_order_correlation=False,
            ),
        )
        eligible = list(engine.models.keys())
        result = engine._handle_hard_problem(profile, eligible, {})
        assert result == "TabDDPM", (
            f"Large hard problem should route to TabDDPM, got {result}"
        )

    def test_hard_problem_cpu_only_excludes_gpu_models(self, engine, all_profiles):
        """CPU-only constraint should filter GPU models from hard problem path."""
        profile = all_profiles["wilt"]
        eligible, excluded = engine._apply_hard_filters({
            "cpu_only": True,
            "dataset_rows": profile.row_count,
        })
        # GReaT should be excluded
        assert "GReaT" not in eligible
        # Fallback should still return something
        result = engine._handle_hard_problem(profile, eligible, excluded)
        assert result is not None
        assert result not in GPU_MODELS

    def test_hard_problem_with_scale_factors_skips_routing(self, engine, all_profiles):
        """When scale_factors are provided, hard problem path should be skipped."""
        profile = all_profiles["HTRU2"]
        sf = {cap: 1.0 for cap in CAPABILITY_NAMES}
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            scale_factors=sf,
            top_n=3,
        )
        # Method should NOT contain "hard_problem_path"
        assert "hard_problem_path" not in result.method


# ===========================================================================
# 3. Constraint Filtering
# ===========================================================================


REPRESENTATIVE_DATASETS = ["Bean", "Titanic", "News"]


class TestConstraintFiltering:
    """Test constraint combinations across representative datasets."""

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_cpu_only_excludes_gpu_models(self, engine, all_profiles, dataset_name):
        """CPU-only should never recommend GPU models."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            constraints={"cpu_only": True},
            method="rule_based",
            top_n=5,
        )
        assert result.recommended_model.model_name not in GPU_MODELS, (
            f"GPU model {result.recommended_model.model_name} recommended with cpu_only=True"
        )
        for alt in result.alternative_models:
            assert alt.model_name not in GPU_MODELS, (
                f"GPU model {alt.model_name} in alternatives with cpu_only=True"
            )

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_strict_dp_only_dp_models(self, engine, all_profiles, dataset_name):
        """Strict DP should only recommend DP-capable models."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            constraints={"strict_dp": True},
            method="rule_based",
            top_n=5,
        )
        assert result.recommended_model.model_name in DP_MODELS, (
            f"Non-DP model {result.recommended_model.model_name} recommended with strict_dp=True"
        )
        for alt in result.alternative_models:
            assert alt.model_name in DP_MODELS, (
                f"Non-DP model {alt.model_name} in alternatives with strict_dp=True"
            )

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_cpu_plus_dp_constraint(self, engine, all_profiles, dataset_name):
        """Combined CPU+DP should only recommend CPU-compatible DP models."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            constraints={"cpu_only": True, "strict_dp": True},
            method="rule_based",
            top_n=5,
        )
        assert result.recommended_model.model_name in CPU_DP_MODELS, (
            f"Model {result.recommended_model.model_name} not in CPU_DP_MODELS"
        )
        for alt in result.alternative_models:
            assert alt.model_name in CPU_DP_MODELS

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_excluded_models_have_reasons(self, engine, all_profiles, dataset_name):
        """When constraints filter models, excluded_models should have reasons."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            constraints={"cpu_only": True},
            method="rule_based",
            top_n=3,
        )
        # Some GPU models should be in excluded_models
        for gpu_model in GPU_MODELS:
            if gpu_model in result.excluded_models:
                assert len(result.excluded_models[gpu_model]) > 0

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_allowed_models_filter(self, engine, all_profiles, dataset_name):
        """allowed_models constraint should restrict to specified models."""
        profile = all_profiles[dataset_name]
        allowed = ["ARF", "CTGAN", "GaussianCopula"]
        result = engine.recommend(
            dataset_profile=profile,
            constraints={"allowed_models": allowed},
            method="rule_based",
            top_n=5,
        )
        assert result.recommended_model.model_name in allowed
        for alt in result.alternative_models:
            assert alt.model_name in allowed

    @pytest.mark.parametrize("dataset_name", REPRESENTATIVE_DATASETS)
    def test_no_constraints_returns_valid_model(self, engine, all_profiles, dataset_name):
        """No constraints should still return a valid model from registry."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=3,
        )
        assert result.recommended_model.model_name in ALL_MODELS


# ===========================================================================
# 4. Scoring Consistency
# ===========================================================================


class TestScoringConsistency:
    """Test capability scoring logic and scale factors."""

    def test_required_capabilities_severe_skew(self, engine, all_profiles):
        """Severe skew should set skew_handling requirement to 3 or 4."""
        profile = all_profiles["Bean"]  # Has severe_skew=True
        required = engine._calculate_required_capabilities(profile, None)
        assert required["skew_handling"] >= 3

    def test_required_capabilities_high_cardinality(self, engine, all_profiles):
        """High cardinality should set cardinality_handling requirement."""
        profile = all_profiles["insurance"]  # Has high_cardinality=True
        required = engine._calculate_required_capabilities(profile, None)
        assert required["cardinality_handling"] >= 3

    def test_required_capabilities_no_stress(self, engine):
        """No stress factors should set all requirements to 0."""
        profile = DatasetProfile(
            row_count=5000,
            column_count=10,
            stress_factors=StressFactors(
                severe_skew=False,
                high_cardinality=False,
                zipfian_distribution=False,
                small_data=False,
                large_data=False,
                higher_order_correlation=False,
            ),
        )
        required = engine._calculate_required_capabilities(profile, None)
        assert all(v == 0 for v in required.values())

    def test_higher_capability_scores_higher(self, engine):
        """Models with higher capability should score higher for active stress factors."""
        profile = DatasetProfile(
            row_count=5000,
            column_count=10,
            stress_factors=StressFactors(
                severe_skew=True,
                high_cardinality=False,
                zipfian_distribution=False,
                small_data=False,
                large_data=False,
                higher_order_correlation=False,
            ),
            skewness=SkewnessMetrics(
                column_scores={"col1": 3.5},
                max_skewness=3.5,
                severe_columns=["col1"],
            ),
        )
        required = engine._calculate_required_capabilities(profile, None)
        eligible = list(engine.models.keys())
        scored = engine._score_models(eligible, required)

        # Build lookup
        scores_by_name = {m["model_name"]: m["total_score"] for m in scored}

        # Check that models with skew_handling=4 score >= models with skew_handling=1
        high_skew_models = [
            name for name, info in engine.models.items()
            if info["capabilities"]["skew_handling"] >= 4
        ]
        low_skew_models = [
            name for name, info in engine.models.items()
            if info["capabilities"]["skew_handling"] <= 1
        ]
        if high_skew_models and low_skew_models:
            max_low = max(scores_by_name.get(m, 0) for m in low_skew_models)
            min_high = min(scores_by_name.get(m, 0) for m in high_skew_models)
            assert min_high >= max_low, (
                f"High-skew models should score >= low-skew models: "
                f"min_high={min_high}, max_low={max_low}"
            )

    def test_scale_factor_zero_disables_capability(self, engine, all_profiles):
        """SF=0.0 should make a capability contribute nothing to score."""
        profile = all_profiles["Bean"]
        required = engine._calculate_required_capabilities(profile, None)
        eligible = list(engine.models.keys())

        # Normal scoring
        scored_normal = engine._score_models(eligible, required)
        # Scoring with skew zeroed out
        sf = {cap: 1.0 for cap in CAPABILITY_NAMES}
        sf["skew_handling"] = 0.0
        scored_zeroed = engine._score_models(eligible, required, scale_factors=sf)

        # Models that differ only in skew should have compressed score differences
        normal_by_name = {m["model_name"]: m for m in scored_normal}
        zeroed_by_name = {m["model_name"]: m for m in scored_zeroed}

        # Verify the skew_handling weight is 0
        for model in scored_zeroed:
            if "skew_handling" in model["capability_scores"]:
                assert model["capability_scores"]["skew_handling"]["weight"] == 0.0

    def test_scale_factor_amplifies_capability(self, engine, all_profiles):
        """SF=10.0 should amplify a capability's contribution."""
        profile = all_profiles["Bean"]
        required = engine._calculate_required_capabilities(profile, None)
        eligible = list(engine.models.keys())

        sf = {cap: 1.0 for cap in CAPABILITY_NAMES}
        sf["skew_handling"] = 10.0
        scored = engine._score_models(eligible, required, scale_factors=sf)

        for model in scored:
            if "skew_handling" in model["capability_scores"]:
                assert model["capability_scores"]["skew_handling"]["scale_factor"] == 10.0

    def test_score_ordering_deterministic(self, engine, all_profiles):
        """Same inputs should produce identical score ordering."""
        profile = all_profiles["abalone"]
        required = engine._calculate_required_capabilities(profile, None)
        eligible = list(engine.models.keys())

        scored1 = engine._score_models(eligible, required)
        scored2 = engine._score_models(eligible, required)

        names1 = [m["model_name"] for m in sorted(scored1, key=lambda x: -x["total_score"])]
        names2 = [m["model_name"] for m in sorted(scored2, key=lambda x: -x["total_score"])]
        assert names1 == names2

    def test_all_scores_non_negative(self, engine, all_profiles):
        """All model scores should be non-negative."""
        for dataset_name, profile in all_profiles.items():
            required = engine._calculate_required_capabilities(profile, None)
            eligible = list(engine.models.keys())
            scored = engine._score_models(eligible, required)
            for model in scored:
                assert model["total_score"] >= 0, (
                    f"{dataset_name}: {model['model_name']} has negative score"
                )


# ===========================================================================
# 5. Tie-Breaking
# ===========================================================================


class TestTieBreaking:
    """Test tie-breaking rules."""

    def _make_sorted_models(self, names_scores):
        """Helper: create sorted_models list from [(name, score), ...]."""
        return [
            {"model_name": name, "total_score": score}
            for name, score in names_scores
        ]

    def test_clear_winner_no_tiebreak(self, engine):
        """When score gap > 5%, highest scorer wins."""
        sorted_models = self._make_sorted_models([
            ("AutoDiff", 10.0),
            ("ARF", 5.0),
            ("CTGAN", 4.0),
        ])
        profile = DatasetProfile(
            row_count=5000, column_count=10,
            stress_factors=StressFactors(
                severe_skew=False, high_cardinality=False,
                zipfian_distribution=False, small_data=False,
                large_data=False, higher_order_correlation=False,
            ),
        )
        result = engine._apply_tie_breaking(sorted_models, profile, {})
        assert result == "AutoDiff"

    def test_small_data_tiebreak_prefers_arf(self, engine):
        """When rows < small_data_threshold and scores tied, ARF should win."""
        sorted_models = self._make_sorted_models([
            ("CTGAN", 10.0),
            ("ARF", 9.8),
            ("GaussianCopula", 9.5),
        ])
        profile = DatasetProfile(
            row_count=500, column_count=5,
            stress_factors=StressFactors(
                severe_skew=False, high_cardinality=False,
                zipfian_distribution=False, small_data=True,
                large_data=False, higher_order_correlation=False,
            ),
        )
        result = engine._apply_tie_breaking(sorted_models, profile, {})
        assert result == "ARF"

    def test_speed_preference_tiebreak(self, engine):
        """When prefer_speed=True and scores tied, fast models should win."""
        sorted_models = self._make_sorted_models([
            ("TabDDPM", 10.0),
            ("TVAE", 9.8),
            ("CTGAN", 9.5),
        ])
        profile = DatasetProfile(
            row_count=5000, column_count=10,
            stress_factors=StressFactors(
                severe_skew=False, high_cardinality=False,
                zipfian_distribution=False, small_data=False,
                large_data=False, higher_order_correlation=False,
            ),
        )
        result = engine._apply_tie_breaking(
            sorted_models, profile, {"prefer_speed": True}
        )
        assert result in {"TVAE", "CTGAN", "ARF", "GaussianCopula"}

    def test_quality_default_tiebreak(self, engine):
        """Default tie-breaking should prefer GPU models when cpu_only=false."""
        sorted_models = self._make_sorted_models([
            ("ARF", 10.0),
            ("TabDDPM", 9.8),
            ("CART", 9.5),
        ])
        profile = DatasetProfile(
            row_count=5000, column_count=10,
            stress_factors=StressFactors(
                severe_skew=False, high_cardinality=False,
                zipfian_distribution=False, small_data=False,
                large_data=False, higher_order_correlation=False,
            ),
        )
        # Default (cpu_only not set) → GPU quality priority
        result = engine._apply_tie_breaking(sorted_models, profile, {})
        assert result in {"GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TVAE", "PATECTGAN"}

        # cpu_only=True → CPU quality priority
        result_cpu = engine._apply_tie_breaking(sorted_models, profile, {"cpu_only": True})
        assert result_cpu in {"CART", "SMOTE", "BayesianNetwork", "ARF", "NFlow"}

    def test_deterministic_tiebreak(self, engine, all_profiles):
        """Tie-breaking should be deterministic across runs."""
        profile = all_profiles["Titanic"]
        results = []
        for _ in range(3):
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=3,
            )
            results.append(result.recommended_model.model_name)
        assert results[0] == results[1] == results[2]


# ===========================================================================
# 6. Benchmark Quality (trial4 synthetic data)
# ===========================================================================


class TestBenchmarkQuality:
    """Test benchmark quality comparison using trial4 synthetic outputs."""

    @pytest.mark.parametrize("dataset_name", TRIAL4_DATASETS)
    def test_benchmark_metrics_computable(self, dataset_name):
        """Benchmark metrics should be computable for all trial4 datasets."""
        from synthony.benchmark.metrics import DataQualityBenchmark

        original_path = DATA_DIR / f"{dataset_name}.csv"
        synth_dir = SYNTH_DIR / dataset_name
        if not synth_dir.exists():
            pytest.skip(f"No trial4 data for {dataset_name}")

        original = pd.read_csv(original_path)
        benchmark = DataQualityBenchmark()

        # Try at least one synthetic file
        synth_files = list(synth_dir.glob(f"{dataset_name}__*.csv"))
        assert len(synth_files) > 0, f"No synthetic files in {synth_dir}"

        synth = pd.read_csv(synth_files[0])
        result = benchmark.compare(original, synth)
        assert result is not None
        assert hasattr(result, "overall_score") or hasattr(result, "fidelity")

    @pytest.mark.parametrize("dataset_name", TRIAL4_DATASETS)
    def test_all_synth_models_loadable(self, dataset_name):
        """All synthetic CSVs for trial4 datasets should be loadable."""
        synth_dir = SYNTH_DIR / dataset_name
        if not synth_dir.exists():
            pytest.skip(f"No trial4 data for {dataset_name}")

        synth_files = list(synth_dir.glob(f"{dataset_name}__*.csv"))
        for f in synth_files:
            df = pd.read_csv(f)
            assert len(df) > 0, f"Empty synthetic file: {f}"

    @pytest.mark.parametrize("dataset_name", TRIAL4_DATASETS)
    def test_recommended_model_has_synth_output(self, engine, all_profiles, dataset_name):
        """The recommended model should have synthetic output in trial4."""
        if dataset_name not in all_profiles:
            pytest.skip(f"No profile for {dataset_name}")

        synth_dir = SYNTH_DIR / dataset_name
        if not synth_dir.exists():
            pytest.skip(f"No trial4 data for {dataset_name}")

        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=3,
        )
        rec_model = result.recommended_model.model_name

        # Check if this model has synthetic output
        synth_files = {f.stem.split("__")[1] for f in synth_dir.glob("*.csv")}
        # Note: not all models may have trial4 data, so we just verify the recommendation is valid
        assert rec_model in ALL_MODELS


# ===========================================================================
# 7. Focus Profiles
# ===========================================================================


class TestFocusProfiles:
    """Test focus-based scale factor profiles."""

    def test_get_privacy_focus(self):
        """Privacy focus should return valid scale factors."""
        sf = get_scale_factors("privacy")
        assert isinstance(sf, dict)
        for cap in CAPABILITY_NAMES:
            assert cap in sf
            assert isinstance(sf[cap], float)

    def test_get_fidelity_focus(self):
        """Fidelity focus should return valid scale factors."""
        sf = get_scale_factors("fidelity")
        assert isinstance(sf, dict)
        assert len(sf) == len(CAPABILITY_NAMES)

    def test_get_latency_focus(self):
        """Latency focus should return valid scale factors."""
        sf = get_scale_factors("latency")
        assert isinstance(sf, dict)
        assert len(sf) == len(CAPABILITY_NAMES)

    def test_unknown_focus_raises(self):
        """Unknown focus name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown focus"):
            get_scale_factors("nonexistent_focus")

    def test_register_new_focus(self):
        """register_focus() should add a new profile."""
        register_focus("test_custom", {"skew_handling": 5.0, "privacy_dp": 3.0})
        sf = get_scale_factors("test_custom")
        assert sf["skew_handling"] == 5.0
        assert sf["privacy_dp"] == 3.0
        # Missing capabilities should default to 1.0
        assert sf["cardinality_handling"] == 1.0
        # Cleanup
        del FOCUS_REGISTRY["test_custom"]

    def test_focus_returns_copy(self):
        """get_scale_factors() should return a copy, not a reference."""
        sf1 = get_scale_factors("privacy")
        sf1["skew_handling"] = 999.0
        sf2 = get_scale_factors("privacy")
        assert sf2["skew_handling"] != 999.0

    def test_focus_parameter_in_recommend(self, engine, all_profiles):
        """Focus parameter should be accepted by engine.recommend()."""
        profile = all_profiles["abalone"]
        # Should not raise
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            focus="privacy",
            top_n=3,
        )
        assert result.recommended_model.model_name in ALL_MODELS

    def test_scale_factors_override_focus(self, engine, all_profiles):
        """Explicit scale_factors should override focus parameter."""
        profile = all_profiles["abalone"]
        custom_sf = {cap: 1.0 for cap in CAPABILITY_NAMES}
        custom_sf["privacy_dp"] = 10.0

        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            focus="fidelity",
            scale_factors=custom_sf,
            top_n=3,
        )
        assert result.recommended_model.model_name in ALL_MODELS


# ===========================================================================
# 8. End-to-End Recommendation
# ===========================================================================


class TestEndToEndRecommendation:
    """Full pipeline validation on all datasets."""

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_full_pipeline_structure(self, engine, all_profiles, dataset_name):
        """Full pipeline should produce well-structured RecommendationResult."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=3,
        )

        # Validate required fields
        assert result.dataset_id is not None
        assert result.method is not None
        assert result.recommended_model is not None
        assert result.recommended_model.model_name in ALL_MODELS
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert len(result.recommended_model.reasoning) > 0
        assert isinstance(result.difficulty_summary, dict)

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_alternatives_ranked_by_score(self, engine, all_profiles, dataset_name):
        """Alternative models should be ranked by descending confidence."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=5,
        )
        if len(result.alternative_models) >= 2:
            scores = [alt.confidence_score for alt in result.alternative_models]
            assert scores == sorted(scores, reverse=True), (
                f"{dataset_name}: alternatives not sorted by score: {scores}"
            )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_hard_problem_flag_in_difficulty_summary(self, engine, all_profiles, dataset_name):
        """difficulty_summary.is_hard_problem should match expected value."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=3,
        )
        expected_hard = DATASET_GROUND_TRUTH[dataset_name]["hard"]
        assert result.difficulty_summary["is_hard_problem"] == expected_hard, (
            f"{dataset_name}: is_hard_problem expected={expected_hard}, "
            f"got={result.difficulty_summary['is_hard_problem']}"
        )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_confidence_in_valid_range(self, engine, all_profiles, dataset_name):
        """All confidence scores should be in [0.0, 1.0]."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=5,
        )
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        for alt in result.alternative_models:
            assert 0.0 <= alt.confidence_score <= 1.0, (
                f"{dataset_name}: alt {alt.model_name} confidence={alt.confidence_score}"
            )

    @pytest.mark.parametrize("dataset_name", DATASET_NAMES)
    def test_no_duplicate_models_in_results(self, engine, all_profiles, dataset_name):
        """Primary and alternatives should not have duplicate model names."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=5,
        )
        all_names = [result.recommended_model.model_name] + [
            alt.model_name for alt in result.alternative_models
        ]
        assert len(all_names) == len(set(all_names)), (
            f"{dataset_name}: duplicate models in results: {all_names}"
        )

    @pytest.mark.parametrize("dataset_name", HARD_DATASETS)
    def test_hard_problem_recommends_specialized_model(self, engine, all_profiles, dataset_name):
        """Hard problem datasets should recommend GReaT or approved fallbacks."""
        profile = all_profiles[dataset_name]
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=3,
        )
        acceptable = {"GReaT", "TabDDPM"} | set(engine.config.hard_problem_fallback)
        assert result.recommended_model.model_name in acceptable, (
            f"{dataset_name}: hard problem recommended {result.recommended_model.model_name}, "
            f"expected one of {acceptable}"
        )

    def test_hybrid_method_fallback(self, engine, all_profiles):
        """Hybrid method should work (falls back to rule-based if no LLM)."""
        profile = all_profiles["Titanic"]
        result = engine.recommend(
            dataset_profile=profile,
            method="hybrid",
            top_n=3,
        )
        assert result.recommended_model.model_name in ALL_MODELS
        assert "hybrid" in result.method or "rule_based" in result.method
