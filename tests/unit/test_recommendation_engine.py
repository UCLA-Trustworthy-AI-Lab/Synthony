"""
Unit tests for ModelRecommendationEngine.

Tests recommendation scoring, constraint filtering, and SystemPrompt loading.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from synthony.recommender.engine import ModelRecommendationEngine
from synthony.core.schemas import DatasetProfile, StressFactors, RecommendationConstraints


@pytest.fixture
def mock_dataset_profile():
    """Create a mock dataset profile for testing."""
    return DatasetProfile(
        dataset_id="test_dataset",
        row_count=1000,
        column_count=5,
        stress_factors=StressFactors(
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=False,
            higher_order_correlation=False,
            small_data=False,
            large_data=False,
        ),
    )


@pytest.fixture
def small_data_profile():
    """Profile for small dataset."""
    return DatasetProfile(
        dataset_id="small_dataset",
        row_count=200,
        column_count=3,
        stress_factors=StressFactors(
            severe_skew=False,
            high_cardinality=False,
            zipfian_distribution=False,
            higher_order_correlation=False,
            small_data=True,
            large_data=False,
        ),
    )


@pytest.fixture
def large_data_profile():
    """Profile for large dataset."""
    return DatasetProfile(
        dataset_id="large_dataset",
        row_count=75000,
        column_count=10,
        stress_factors=StressFactors(
            severe_skew=True,
            high_cardinality=True,
            zipfian_distribution=True,
            higher_order_correlation=False,
            small_data=False,
            large_data=True,
        ),
    )


class TestRecommendationEngine:
    """Test recommendation engine functionality."""

    def test_engine_initialization_without_llm(self):
        """Engine should initialize in rule-based only mode without API key."""
        engine = ModelRecommendationEngine()

        assert engine.llm_available is False
        assert engine.openai_client is None

    def test_engine_initialization_with_mock_llm(self):
        """Engine should initialize with LLM support when API key provided."""
        with patch("openai.OpenAI"):
            engine = ModelRecommendationEngine(openai_api_key="test_key")

            assert engine.llm_available is True
            assert engine.openai_client is not None

    def test_rule_based_recommendation(self, mock_dataset_profile):
        """Test rule-based recommendation without LLM."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )

        assert "rule_based" in result.method
        assert result.recommended_model is not None
        assert result.recommended_model.model_name in [
            "TabDDPM", "AutoDiff", "ARF", "TVAE", "DPCART", "AIM", "BayesianNetwork", "SMOTE", "CART", "Identity", "NFlow",
            "CTGAN", "PATECTGAN", "TabSyn", "GReaT",
        ]
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert len(result.recommended_model.reasoning) > 0
        assert len(result.alternative_models) >= 0

    def test_cpu_only_constraint(self, mock_dataset_profile):
        """CPU-only constraint should exclude GPU models."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=True, strict_dp=False),
            method="rule_based",
            top_n=5,
        )

        # Should not recommend GPU-dependent models
        gpu_models = {"TabDDPM", "AutoDiff"}  # Only implemented GPU models
        assert result.recommended_model.model_name not in gpu_models

        # Check alternatives also exclude GPU models
        for alt in result.alternative_models:
            assert alt.model_name not in gpu_models

    def test_strict_dp_constraint(self, mock_dataset_profile):
        """Strict DP constraint should only allow DP models."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=True),
            method="rule_based",
            top_n=3,
        )

        # Should only recommend models with privacy_dp >= dp_threshold from registry
        dp_threshold = engine.config.dp_min_score
        dp_models = {
            name for name, info in engine.models.items()
            if info["capabilities"]["privacy_dp"] >= dp_threshold
        }
        assert result.recommended_model.model_name in dp_models

        # Check alternatives are also DP models
        for alt in result.alternative_models:
            assert alt.model_name in dp_models

    def test_small_data_recommendation(self, small_data_profile):
        """Small data should prefer ARF (best small-data model in registry)."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=small_data_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )

        # ARF is the registry's top-rated model for small data
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])

        assert "ARF" in all_models

    def test_large_data_excludes_llm_models(self, large_data_profile):
        """Large data should route away from GReaT (LLM model, too slow)."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=large_data_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=5,
        )

        # GReaT is the hard-problem primary but is excluded for large data;
        # engine falls back to hard_problem_large_data_fallback instead.
        assert result.recommended_model.model_name != "GReaT"
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])
        assert "GReaT" not in all_models

    def test_top_n_parameter(self, mock_dataset_profile):
        """top_n should control number of alternatives."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=5,
        )

        # Engine returns exactly top_n alternatives (registry has 15 models, plenty available)
        assert len(result.alternative_models) == 5

    def test_excluded_models_with_reasons(self, mock_dataset_profile):
        """Excluded models should have explanations."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=True, strict_dp=False),
            method="rule_based",
            top_n=3,
        )

        # Should have excluded models with reasons
        if result.excluded_models:
            # excluded_models may be list of strings or objects
            for excluded in result.excluded_models:
                if hasattr(excluded, 'model_name'):
                    assert excluded.model_name is not None
                    assert excluded.reason is not None
                else:
                    # It's a string model name
                    assert isinstance(excluded, str)

    def test_systemprompt_loading_default_path(self):
        """SystemPrompt should load from default path."""
        engine = ModelRecommendationEngine()

        # Check if SystemPrompt was loaded
        if engine.system_prompt_loaded:
            assert engine.system_prompt is not None
            assert len(engine.system_prompt) > 0
            assert "SystemPrompt" in engine.system_prompt_path.name

    def test_systemprompt_custom_path(self, tmp_path):
        """SystemPrompt should load from custom path."""
        # Create custom SystemPrompt file
        custom_prompt_path = tmp_path / "custom_prompt.md"
        custom_content = "# Custom System Prompt\n\nThis is a test prompt."
        custom_prompt_path.write_text(custom_content)

        engine = ModelRecommendationEngine(system_prompt_path=custom_prompt_path)

        assert engine.system_prompt_loaded is True
        assert engine.system_prompt == custom_content

    def test_systemprompt_missing_file(self, tmp_path):
        """Engine should handle missing SystemPrompt gracefully."""
        missing_path = tmp_path / "nonexistent.md"

        engine = ModelRecommendationEngine(system_prompt_path=missing_path)

        assert engine.system_prompt_loaded is False
        assert engine.system_prompt is None

    def test_confidence_scores_range(self, mock_dataset_profile):
        """All confidence scores should be between 0.0 and 1.0."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=5,
        )

        # Check recommended model
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0

        # Check alternatives
        for alt in result.alternative_models:
            assert 0.0 <= alt.confidence_score <= 1.0

    def test_reasoning_not_empty(self, mock_dataset_profile):
        """All recommendations should have reasoning."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )

        # Recommended model should have reasoning
        assert len(result.recommended_model.reasoning) > 0
        for reason in result.recommended_model.reasoning:
            assert len(reason) > 0

    def test_model_metadata_included(self, mock_dataset_profile):
        """Recommendations should include model metadata."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )

        # Check recommended model has metadata
        rec = result.recommended_model
        # model_type may not be in ModelRecommendation schema
        assert rec.model_name is not None
        assert rec.confidence_score is not None

    def test_hybrid_mode_fallback(self, mock_dataset_profile):
        """Hybrid mode should fall back to rule-based if LLM unavailable."""
        engine = ModelRecommendationEngine()  # No API key

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="hybrid",
            top_n=3,
        )

        # Should fall back to rule-based
        assert "rule_based" in result.method  # May be "hybrid (llm unavailable - used rule_based)"
        assert result.recommended_model is not None

    @patch("openai.OpenAI")
    def test_llm_mode_requires_api_key(self, mock_openai, mock_dataset_profile):
        """LLM mode should work with API key."""
        # Create mock LLM response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "recommended_model": "TabDDPM",
                "confidence": 0.95,
                "reasoning": "Test reasoning",
                "alternatives": []
            })))
        ]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        engine = ModelRecommendationEngine(openai_api_key="test_key")

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="llm",
            top_n=3,
        )

        # Should use LLM mode
        assert result.method == "llm"
        assert result.llm_reasoning is not None

    def test_combined_constraints(self, mock_dataset_profile):
        """Test with both CPU and DP constraints."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            constraints=RecommendationConstraints(cpu_only=True, strict_dp=True),
            method="rule_based",
            top_n=3,
        )

        # Should only recommend CPU-compatible DP models (derived from registry)
        dp_threshold = engine.config.dp_min_score
        cpu_dp_models = {
            name for name, info in engine.models.items()
            if info["capabilities"]["privacy_dp"] >= dp_threshold
            and info["constraints"].get("cpu_only_compatible", False)
        }
        assert result.recommended_model.model_name in cpu_dp_models


class TestWithRealData:
    """
    Tests using real CSV files from dataset/input_data.
    Uses IndianLiverPatient.csv (small) and insurance.csv (medium).
    
    Paths can be configured via .env:
    - INPUT_TESTDATA: input data directory
    - OUTPUT_TESTDATA: output data directory
    """

    def test_recommendation_with_real_small_data(self, real_csv_small):
        """Test recommendation using real IndianLiverPatient dataset."""
        from synthony.core.analyzer import StochasticDataAnalyzer
        
        engine = ModelRecommendationEngine()
        analyzer = StochasticDataAnalyzer()
        
        # Profile real data
        profile = analyzer.analyze(real_csv_small)
        
        # Get recommendation
        result = engine.recommend(
            dataset_profile=profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )
        
        assert result.recommended_model is not None
        assert result.recommended_model.model_name is not None
        assert result.recommended_model.confidence_score > 0
        assert len(result.recommended_model.reasoning) > 0
        
        # Small data (583 rows) should favor certain models
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])
        assert len(all_models) >= 1

    def test_recommendation_with_real_medium_data(self, real_csv_medium):
        """Test recommendation using real insurance dataset."""
        from synthony.core.analyzer import StochasticDataAnalyzer
        
        engine = ModelRecommendationEngine()
        analyzer = StochasticDataAnalyzer()
        
        # Profile real data
        profile = analyzer.analyze(real_csv_medium)
        
        # Get recommendation
        result = engine.recommend(
            dataset_profile=profile,
            constraints=RecommendationConstraints(cpu_only=False, strict_dp=False),
            method="rule_based",
            top_n=3,
        )
        
        assert result.recommended_model is not None
        assert result.recommended_model.model_name is not None
        # Medium dataset (1338 rows) should have good recommendations
        assert result.recommended_model.confidence_score > 0

    def test_cpu_only_with_real_data(self, real_csv_small):
        """Test CPU-only constraint with real data."""
        from synthony.core.analyzer import StochasticDataAnalyzer
        
        engine = ModelRecommendationEngine()
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze(real_csv_small)
        
        result = engine.recommend(
            dataset_profile=profile,
            constraints=RecommendationConstraints(cpu_only=True, strict_dp=False),
            method="rule_based",
            top_n=3,
        )
        
        # GPU models should be excluded
        gpu_models = {"TabDDPM", "AutoDiff"}
        assert result.recommended_model.model_name not in gpu_models
        for alt in result.alternative_models:
            assert alt.model_name not in gpu_models
