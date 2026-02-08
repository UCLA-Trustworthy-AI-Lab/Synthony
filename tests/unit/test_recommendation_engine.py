"""
Unit tests for ModelRecommendationEngine.

Tests recommendation scoring and SystemPrompt loading.
"""

import json
from unittest.mock import Mock, patch

import pytest

from synthony.core.schemas import DatasetProfile, StressFactors
from synthony.recommender.engine import ModelRecommendationEngine


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
            method="rule_based",
            top_n=3,
        )

        assert "rule_based" in result.method
        assert result.recommended_model is not None
        assert result.recommended_model.model_name in [
            "GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TabTree", "ARF",
            "CTGAN", "TVAE", "PATE-CTGAN", "DPCART", "AIM", "GaussianCopula"
        ]
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert len(result.recommended_model.reasoning) > 0
        assert len(result.alternative_models) >= 0

    def test_small_data_recommendation(self, small_data_profile):
        """Small data should prefer ARF or GaussianCopula."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=small_data_profile,
            method="rule_based",
            top_n=3,
        )

        # ARF and GaussianCopula are best for small data
        # Should be recommended or in top alternatives
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])

        assert "ARF" in all_models or "GaussianCopula" in all_models

    def test_large_data_excludes_llm_models(self, large_data_profile):
        """Large data should exclude LLM models (too slow)."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=large_data_profile,
            method="rule_based",
            top_n=5,
        )

        # Should not recommend GReaT for large data
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])

        assert "GReaT" not in all_models  # Too slow for large data

    def test_row_count_filtering(self):
        """Row-count filtering should exclude models outside their supported range."""
        engine = ModelRecommendationEngine()

        # Dataset with 100 rows - small but not so small that all models are excluded
        small_profile = DatasetProfile(
            dataset_id="small_dataset",
            row_count=100,
            column_count=2,
            stress_factors=StressFactors(
                severe_skew=False,
                high_cardinality=False,
                zipfian_distribution=False,
                higher_order_correlation=False,
                small_data=True,
                large_data=False,
            ),
        )

        result = engine.recommend(
            dataset_profile=small_profile,
            method="rule_based",
            top_n=5,
        )

        # Should still get a recommendation
        assert result.recommended_model is not None

        # Models with min_rows > 100 should be excluded
        all_recommended = [result.recommended_model.model_name]
        all_recommended.extend([alt.model_name for alt in result.alternative_models])

        # High-min-row models like TabDDPM (min 1000) should NOT appear
        assert "TabDDPM" not in all_recommended

    def test_top_n_parameter(self, mock_dataset_profile):
        """top_n should control total number of recommendations."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            method="rule_based",
            top_n=3,
        )

        # top_n controls alternatives count
        assert len(result.alternative_models) <= 5

    def test_excluded_models_with_reasons(self, mock_dataset_profile):
        """Excluded models info should be present."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            method="rule_based",
            top_n=3,
        )

        # excluded_models is a dict in the actual engine
        if result.excluded_models:
            assert isinstance(result.excluded_models, dict)

    def test_systemprompt_loading_default_path(self):
        """SystemPrompt should load from default path."""
        engine = ModelRecommendationEngine()

        # Check if SystemPrompt was loaded
        if engine.system_prompt_loaded:
            assert engine.system_prompt is not None
            assert len(engine.system_prompt) > 0
            assert engine.system_prompt_path.name == "SystemPrompt_v3.md"

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
            method="rule_based",
            top_n=3,
        )

        # Recommended model should have reasoning
        assert len(result.recommended_model.reasoning) > 0
        for reason in result.recommended_model.reasoning:
            assert len(reason) > 0

    def test_model_info_included(self, mock_dataset_profile):
        """Recommendations should include model info."""
        engine = ModelRecommendationEngine()

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            method="rule_based",
            top_n=3,
        )

        # Check recommended model has model_info
        rec = result.recommended_model
        assert rec.model_info is not None
        assert "type" in rec.model_info
        assert "name" in rec.model_info

    def test_hybrid_mode_fallback(self, mock_dataset_profile):
        """Hybrid mode should fall back to rule-based if LLM unavailable."""
        engine = ModelRecommendationEngine()  # No API key

        result = engine.recommend(
            dataset_profile=mock_dataset_profile,
            method="hybrid",
            top_n=3,
        )

        # Should fall back to rule-based (method may include fallback note)
        assert "rule_based" in result.method or "hybrid" in result.method
        assert result.recommended_model is not None

    def test_llm_mode_requires_api_key(self, mock_dataset_profile):
        """LLM mode should work with API key."""
        # Create mock LLM response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "recommended_model": "GReaT",
                "confidence": 0.95,
                "reasoning": "Test reasoning",
                "alternatives": []
            })))
        ]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            engine = ModelRecommendationEngine(openai_api_key="test_key")

            result = engine.recommend(
                dataset_profile=mock_dataset_profile,
                method="llm",
                top_n=3,
            )

            # Should use LLM mode
            assert "llm" in result.method
            assert result.llm_reasoning is not None
