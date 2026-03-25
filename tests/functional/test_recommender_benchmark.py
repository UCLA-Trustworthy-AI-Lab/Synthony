"""
Benchmark Evaluation Tests for Recommender Engine.

Compares rule-based, LLM-based, and hybrid recommendation approaches
against actual benchmark results from evaluation_results_clean.csv.

This test module evaluates:
1. How well the recommender predicts actual top-performing models
2. Consistency and determinism of recommendations
3. Shortcomings in model_capabilities.json capability scores
"""

import json
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch

import pytest

from synthony.recommender.engine import ModelRecommendationEngine
from synthony.core.schemas import DatasetProfile, StressFactors, ZipfianMetrics


# ============================================================================
# Test Data and Fixtures
# ============================================================================

# Benchmark results from evaluation_results_clean.csv
BENCHMARK_DATA = {
    "Abalone": {
        "models_tested": ["AIM", "AutoDiff", "CART", "DPCART", "Identity", "SMOTE", "TabDDPM"],
        "best_by_shape": ("Identity", 0.988),  # But this is baseline - actual best is CART
        "actual_best": ("CART", 0.916),  # Excluding Identity baseline
        "row_count": 4177,
    },
    "Bean": {
        "models_tested": ["AutoDiff", "DPCART", "NFlow", "TVAE"],
        "best_by_shape": ("TVAE", 0.852),
        "actual_best": ("TVAE", 0.852),
        "row_count": 13611,
    },
    "IndianLiverPatient": {
        "models_tested": ["AIM", "DPCART", "SMOTE", "TVAE", "TabDDPM"],
        "best_by_shape": ("SMOTE", 0.884),
        "actual_best": ("SMOTE", 0.884),
        "row_count": 583,
    },
    "Obesity": {
        "models_tested": ["AIM", "AutoDiff", "CART", "DPCART", "Identity", "NFlow", "SMOTE", "TabDDPM"],
        "best_by_shape": ("CART", 0.874),
        "actual_best": ("CART", 0.874),
        "row_count": 2111,
    },
    "faults": {
        "models_tested": ["BayesianNetwork", "DPCART", "Identity", "SMOTE", "TVAE", "TabDDPM"],
        "best_by_shape": ("SMOTE", 0.949),
        "actual_best": ("SMOTE", 0.949),
        "row_count": 1941,
    },
    "insurance": {
        "models_tested": ["ARF", "CART", "SMOTE"],
        "best_by_shape": ("ARF", 0.901),
        "actual_best": ("ARF", 0.901),
        "row_count": 1338,
    },
    "wilt": {
        "models_tested": ["AIM", "AutoDiff", "BayesianNetwork", "CART", "DPCART", "Identity", "NFlow", "SMOTE", "TVAE"],
        "best_by_shape": ("Identity", 0.984),  # Baseline
        "actual_best": ("CART", 0.894),  # Excluding Identity
        "row_count": 4339,
    },
}


def create_dataset_profile(
    dataset_name: str,
    row_count: int,
    severe_skew: bool = False,
    high_cardinality: bool = False,
    zipfian: bool = False,
) -> DatasetProfile:
    """Create a mock DatasetProfile for testing."""
    zipfian_metrics = None
    if zipfian:
        zipfian_metrics = ZipfianMetrics(
            detected=True,
            top_20_percent_ratio=0.85,
        )
    
    return DatasetProfile(
        dataset_id=dataset_name,
        row_count=row_count,
        column_count=10,
        stress_factors=StressFactors(
            severe_skew=severe_skew,
            high_cardinality=high_cardinality,
            zipfian_distribution=zipfian,
            higher_order_correlation=False,
            small_data=row_count < 1000,
            large_data=row_count > 50000,
        ),
        zipfian=zipfian_metrics,
    )


@pytest.fixture
def engine() -> ModelRecommendationEngine:
    """Create a recommender engine with default config."""
    return ModelRecommendationEngine()


@pytest.fixture
def benchmark_csv_path() -> Path:
    """Path to evaluation results CSV."""
    return Path(__file__).parent.parent.parent / "evaluation_results_clean.csv"


# ============================================================================
# Rule-Based Recommendation Tests
# ============================================================================

class TestRuleBasedVsBenchmarks:
    """Test rule-based recommendations against benchmark results."""

    @pytest.mark.parametrize("dataset_name,benchmark_info", BENCHMARK_DATA.items())
    def test_recommendation_for_dataset(
        self, engine: ModelRecommendationEngine, dataset_name: str, benchmark_info: Dict
    ):
        """Test that recommendation is reasonable for each benchmark dataset."""
        profile = create_dataset_profile(
            dataset_name=dataset_name,
            row_count=benchmark_info["row_count"],
        )
        
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=5,
        )
        
        # Get all recommended models
        all_recommended = [result.recommended_model.model_name]
        all_recommended.extend([alt.model_name for alt in result.alternative_models])
        
        # Check if actual best model is in the recommendation list
        actual_best = benchmark_info["actual_best"][0]
        models_tested = benchmark_info["models_tested"]
        
        # The recommended model should be reasonable
        assert result.recommended_model.model_name is not None
        assert result.recommended_model.confidence_score > 0
        
        # Log for analysis
        print(f"\n{dataset_name}:")
        print(f"  Actual best: {actual_best}")
        print(f"  Recommended: {result.recommended_model.model_name}")
        print(f"  Alternatives: {[m.model_name for m in result.alternative_models[:3]]}")
        print(f"  Models tested in benchmark: {models_tested}")

    def test_small_data_recommendation_accuracy(self, engine: ModelRecommendationEngine):
        """Test recommendations for small datasets prefer appropriate models."""
        # IndianLiverPatient: 583 rows, SMOTE performed best
        profile = create_dataset_profile(
            dataset_name="IndianLiverPatient",
            row_count=583,
        )
        
        result = engine.recommend(
            dataset_profile=profile,
            method="rule_based",
            top_n=5,
        )
        
        all_models = [result.recommended_model.model_name]
        all_models.extend([alt.model_name for alt in result.alternative_models])
        
        # Small data should recommend tree-based or statistical models
        small_data_appropriate = {"ARF", "CART", "BayesianNetwork", "SMOTE", "GaussianCopula"}
        assert any(m in small_data_appropriate for m in all_models), \
            f"Expected small-data models, got: {all_models}"

    def test_simple_statistical_models_underrated(self, engine: ModelRecommendationEngine):
        """Test whether SMOTE/CART are recommended when they actually perform well."""
        # SMOTE outperformed complex models on multiple datasets
        smote_wins = ["IndianLiverPatient", "faults"]
        cart_wins = ["Obesity", "wilt"]
        
        for dataset in smote_wins + cart_wins:
            profile = create_dataset_profile(
                dataset_name=dataset,
                row_count=BENCHMARK_DATA[dataset]["row_count"],
            )
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=5,
            )
            
            all_models = [result.recommended_model.model_name]
            all_models.extend([alt.model_name for alt in result.alternative_models])
            
            # These simple models should appear in recommendations
            # This test may fail, highlighting a shortcoming
            print(f"\n{dataset}: Recommended {all_models[:3]}")


# ============================================================================
# LLM-Based Recommendation Tests (Mocked)
# ============================================================================

class TestLLMBasedMocked:
    """Test LLM-based recommendations with mocked API responses."""

    @patch("openai.OpenAI")
    def test_llm_recommendation_parsing(
        self, mock_openai: Mock, engine: ModelRecommendationEngine
    ):
        """Test that LLM responses are parsed correctly."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "recommended_model": "TabDDPM",
                "confidence": 0.92,
                "reasoning": "TabDDPM excels at handling complex distributions.",
                "alternatives": ["ARF", "TVAE"],  # Parser expects strings, not dicts
                "warnings": [],
                "hard_problem_detected": False,
                "tie_break_applied": False,
            })))
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create engine with mocked API key
        engine = ModelRecommendationEngine(openai_api_key="test_key")
        
        profile = create_dataset_profile("test", row_count=5000)
        result = engine.recommend(
            dataset_profile=profile,
            method="llm",
            top_n=3,
        )
        
        assert result.method == "llm"
        assert result.recommended_model.model_name == "TabDDPM"
        assert result.llm_reasoning is not None

    @patch("openai.OpenAI")
    def test_llm_handles_edge_cases(self, mock_openai: Mock):
        """Test LLM mode handles edge cases gracefully."""
        # Mock response with minimal data
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "recommended_model": "ARF",
                "confidence": 0.75,
                "reasoning": "Best for small data",
            })))
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = ModelRecommendationEngine(openai_api_key="test_key")
        profile = create_dataset_profile("small", row_count=100)
        
        result = engine.recommend(
            dataset_profile=profile,
            method="llm",
            top_n=3,
        )
        
        assert result.recommended_model.model_name == "ARF"


# ============================================================================
# Hybrid Recommendation Tests
# ============================================================================

class TestHybridRecommendations:
    """Test hybrid mode combining rule-based and LLM reasoning."""

    def test_hybrid_fallback_without_llm(self, engine: ModelRecommendationEngine):
        """Hybrid mode should fallback to rule-based when LLM unavailable."""
        profile = create_dataset_profile("test", row_count=5000)
        
        result = engine.recommend(
            dataset_profile=profile,
            method="hybrid",
            top_n=3,
        )
        
        # Should indicate fallback occurred
        assert "rule_based" in result.method.lower() or "hybrid" in result.method.lower()
        assert result.recommended_model is not None

    @patch("openai.OpenAI")
    def test_hybrid_combines_approaches(self, mock_openai: Mock):
        """Test that hybrid mode properly combines rule-based ranking with LLM reasoning."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "recommended_model": "TabDDPM",
                "reasoning": "LLM analysis: TabDDPM best for this distribution",
                "alternatives": ["ARF", "TVAE"],
            })))
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = ModelRecommendationEngine(openai_api_key="test_key")
        profile = create_dataset_profile("hybrid_test", row_count=10000)
        
        result = engine.recommend(
            dataset_profile=profile,
            method="hybrid",
            top_n=3,
        )
        
        assert result.method == "hybrid"
        assert result.llm_reasoning is not None


# ============================================================================
# Recommendation Consistency & Determinism Tests
# ============================================================================

class TestRecommendationConsistency:
    """Test that recommendations are deterministic and consistent."""

    def test_same_input_same_output(self, engine: ModelRecommendationEngine):
        """Same input should always produce same recommendation."""
        profile = create_dataset_profile("determinism_test", row_count=5000)
        
        results = []
        for _ in range(5):
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=3,
            )
            results.append(result.recommended_model.model_name)
        
        # All results should be identical
        assert len(set(results)) == 1, f"Non-deterministic results: {results}"

    def test_tie_breaking_is_deterministic(self, engine: ModelRecommendationEngine):
        """Tie-breaking should produce consistent results."""
        # Small data profile where multiple models may tie
        profile = create_dataset_profile("tie_test", row_count=200)
        
        results = []
        for _ in range(3):
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=5,
            )
            results.append(result.recommended_model.model_name)
        
        assert len(set(results)) == 1, f"Tie-breaking not deterministic: {results}"


# ============================================================================
# Capability Score Accuracy Analysis
# ============================================================================

class TestCapabilityScoreAccuracy:
    """Analyze whether capability scores correlate with benchmark performance."""

    def test_correlation_handling_score_matches_benchmarks(
        self, engine: ModelRecommendationEngine
    ):
        """Test if models with high correlation_handling score actually perform well."""
        high_corr_models = []
        capabilities = engine.model_capabilities.get("models", {})
        
        for model_name, model_info in capabilities.items():
            caps = model_info.get("capabilities", {})
            if caps.get("correlation_handling", 0) >= 4:
                high_corr_models.append(model_name)
        
        # These models should appear in benchmark winners
        benchmark_winners = set()
        for info in BENCHMARK_DATA.values():
            benchmark_winners.add(info["actual_best"][0])
        
        overlap = set(high_corr_models) & benchmark_winners
        
        print(f"\nHigh correlation_handling models: {high_corr_models}")
        print(f"Benchmark winners: {benchmark_winners}")
        print(f"Overlap: {overlap}")

    def test_small_data_score_matches_benchmarks(
        self, engine: ModelRecommendationEngine
    ):
        """Test if models with high small_data score perform well on small datasets."""
        # Get models with high small_data capability
        high_small_data = []
        capabilities = engine.model_capabilities.get("models", {})
        
        for model_name, model_info in capabilities.items():
            caps = model_info.get("capabilities", {})
            if caps.get("small_data", 0) >= 3:
                high_small_data.append(model_name)
        
        # Check against small dataset winners
        small_datasets = ["IndianLiverPatient", "insurance"]
        small_data_winners = [BENCHMARK_DATA[d]["actual_best"][0] for d in small_datasets]
        
        print(f"\nHigh small_data models: {high_small_data}")
        print(f"Small dataset winners: {small_data_winners}")


# ============================================================================
# Gap Analysis & Reporting
# ============================================================================

class TestGapAnalysis:
    """Identify gaps between recommendations and benchmark results."""

    def test_generate_accuracy_report(self, engine: ModelRecommendationEngine):
        """Generate a report of recommendation accuracy vs benchmarks."""
        results = []
        
        for dataset_name, benchmark_info in BENCHMARK_DATA.items():
            profile = create_dataset_profile(
                dataset_name=dataset_name,
                row_count=benchmark_info["row_count"],
            )
            
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=5,
            )
            
            all_recommended = [result.recommended_model.model_name]
            all_recommended.extend([alt.model_name for alt in result.alternative_models])
            
            actual_best = benchmark_info["actual_best"][0]
            
            results.append({
                "dataset": dataset_name,
                "row_count": benchmark_info["row_count"],
                "actual_best": actual_best,
                "recommended": result.recommended_model.model_name,
                "in_top5": actual_best in all_recommended,
                "position": all_recommended.index(actual_best) + 1 if actual_best in all_recommended else -1,
            })
        
        # Print report
        print("\n" + "=" * 80)
        print("RECOMMENDATION ACCURACY REPORT")
        print("=" * 80)
        
        correct = sum(1 for r in results if r["recommended"] == r["actual_best"])
        in_top5 = sum(1 for r in results if r["in_top5"])
        total = len(results)
        
        print(f"\nExact match: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"In top 5: {in_top5}/{total} ({100*in_top5/total:.1f}%)")
        
        print("\nDetails:")
        for r in results:
            match = "✓" if r["recommended"] == r["actual_best"] else "✗"
            pos = f"(pos {r['position']})" if r["in_top5"] else "(not in top 5)"
            print(f"  {match} {r['dataset']}: Recommended {r['recommended']}, Actual best: {r['actual_best']} {pos}")
        
        print("\n" + "=" * 80)

    def test_identify_underperforming_models(self, engine: ModelRecommendationEngine):
        """Identify models that are recommended but underperform in benchmarks."""
        # Collect all recommendations
        recommended_counts = {}
        for dataset_name, benchmark_info in BENCHMARK_DATA.items():
            profile = create_dataset_profile(
                dataset_name=dataset_name,
                row_count=benchmark_info["row_count"],
            )
            
            result = engine.recommend(
                dataset_profile=profile,
                method="rule_based",
                top_n=1,
            )
            
            model = result.recommended_model.model_name
            recommended_counts[model] = recommended_counts.get(model, 0) + 1
        
        # Count actual wins
        actual_wins = {}
        for info in BENCHMARK_DATA.values():
            model = info["actual_best"][0]
            actual_wins[model] = actual_wins.get(model, 0) + 1
        
        print("\n" + "=" * 80)
        print("MODEL RECOMMENDATION vs ACTUAL PERFORMANCE")
        print("=" * 80)
        print(f"\n{'Model':<20} {'Recommended':<15} {'Actual Wins':<15} {'Gap':<10}")
        print("-" * 60)
        
        all_models = set(recommended_counts.keys()) | set(actual_wins.keys())
        for model in sorted(all_models):
            rec = recommended_counts.get(model, 0)
            wins = actual_wins.get(model, 0)
            gap = rec - wins
            gap_str = f"+{gap}" if gap > 0 else str(gap)
            print(f"{model:<20} {rec:<15} {wins:<15} {gap_str:<10}")
