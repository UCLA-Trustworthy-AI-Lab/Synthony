"""
Functional tests for recommendation methods.

Tests rule-based, LLM, and hybrid recommendation approaches.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthony.api.server import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def titanic_like_csv():
    """Create Titanic-like dataset for testing."""
    np.random.seed(42)
    from scipy.stats import lognorm

    df = pd.DataFrame({
        "PassengerId": range(1, 715),
        "Survived": np.random.choice([0, 1], 714),
        "Pclass": np.random.choice([1, 2, 3], 714),
        "Age": np.random.normal(30, 15, 714).clip(0, 80),
        "SibSp": np.random.choice([0, 1, 2, 3], 714),
        "Parch": np.random.choice([0, 1, 2], 714),
        "Fare": lognorm.rvs(s=0.95, scale=np.exp(3), size=714, random_state=42),
        "Embarked": np.random.choice(["C", "Q", "S"], 714),
    })

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return ("titanic.csv", csv_buffer, "text/csv")


class TestRuleBasedMethod:
    """Test rule-based recommendation method."""

    def test_rule_based_deterministic(self, client, titanic_like_csv):
        """Rule-based should be deterministic (same input → same output)."""
        # First call
        response1 = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "test_deterministic",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": titanic_like_csv},
        )

        # Recreate file (BytesIO was consumed)
        np.random.seed(42)
        from scipy.stats import lognorm
        df = pd.DataFrame({
            "PassengerId": range(1, 715),
            "Survived": np.random.choice([0, 1], 714),
            "Pclass": np.random.choice([1, 2, 3], 714),
            "Age": np.random.normal(30, 15, 714).clip(0, 80),
            "SibSp": np.random.choice([0, 1, 2, 3], 714),
            "Parch": np.random.choice([0, 1, 2], 714),
            "Fare": lognorm.rvs(s=0.95, scale=np.exp(3), size=714, random_state=42),
            "Embarked": np.random.choice(["C", "Q", "S"], 714),
        })
        csv_buffer2 = io.BytesIO()
        df.to_csv(csv_buffer2, index=False)
        csv_buffer2.seek(0)

        # Second call
        response2 = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "test_deterministic",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("titanic.csv", csv_buffer2, "text/csv")},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should recommend same model
        rec1 = data1["recommendation"]["recommended_model"]["model_name"]
        rec2 = data2["recommendation"]["recommended_model"]["model_name"]
        assert rec1 == rec2

        # Confidence should be identical
        conf1 = data1["recommendation"]["recommended_model"]["confidence_score"]
        conf2 = data2["recommendation"]["recommended_model"]["confidence_score"]
        assert conf1 == conf2

    def test_rule_based_fast_response(self, client, titanic_like_csv):
        """Rule-based should respond quickly (no LLM API call)."""
        import time

        start_time = time.time()

        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": titanic_like_csv},
        )

        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        # Rule-based should be fast (< 5 seconds)
        assert elapsed_time < 5.0

    def test_rule_based_complete_reasoning(self, client, titanic_like_csv):
        """Rule-based should provide complete reasoning."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": titanic_like_csv},
        )

        assert response.status_code == 200
        data = response.json()

        rec = data["recommendation"]["recommended_model"]

        # Should have reasoning
        assert "reasoning" in rec
        assert len(rec["reasoning"]) > 0

        # Each reasoning point should be descriptive
        for reason in rec["reasoning"]:
            assert len(reason) > 10  # Not just empty strings


class TestLLMMethod:
    """Test LLM recommendation method."""

    def test_llm_mode_availability(self, client):
        """Check if LLM mode is available."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Check LLM availability
        llm_available = data["llm_available"]

        if llm_available:
            pytest.skip("LLM mode is available, full test would require API key")
        else:
            # LLM not available, test should gracefully handle

            # Create test data
            np.random.seed(42)
            df = pd.DataFrame({"value": np.random.randn(100)})
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # Try LLM mode
            response = client.post(
                "/analyze-and-recommend",
                params={"method": "llm", "top_n": 3},
                files={"file": ("test.csv", csv_buffer, "text/csv")},
            )

            # Should either return error or fall back to rule-based
            assert response.status_code in [200, 400, 503]

    def test_llm_fallback_to_rule_based(self, client, titanic_like_csv):
        """LLM mode should fall back to rule-based if unavailable."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "llm",
                "top_n": 3,
            },
            files={"file": titanic_like_csv},
        )

        # Should complete successfully (may fall back)
        assert response.status_code in [200, 400, 503]

        if response.status_code == 200:
            data = response.json()

            # Method should be either "llm" or "rule_based" (fallback)
            assert data["recommendation"]["method"] in ["llm", "rule_based"]


class TestHybridMethod:
    """Test hybrid recommendation method."""

    def test_hybrid_fallback_without_llm(self, client, titanic_like_csv):
        """Hybrid should fall back to rule-based if LLM unavailable."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "hybrid",
                "top_n": 3,
            },
            files={"file": titanic_like_csv},
        )

        assert response.status_code == 200
        data = response.json()

        # Should complete successfully
        assert "recommendation" in data
        assert "recommended_model" in data["recommendation"]

        # Method should be either "hybrid" or "rule_based" (fallback)
        assert data["recommendation"]["method"] in ["hybrid", "rule_based"]

    def test_hybrid_includes_alternatives(self, client, titanic_like_csv):
        """Hybrid mode should include alternative models."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "hybrid",
                "top_n": 5,
            },
            files={"file": titanic_like_csv},
        )

        assert response.status_code == 200
        data = response.json()

        rec = data["recommendation"]

        # Should have alternatives
        if "alternative_models" in rec:
            assert len(rec["alternative_models"]) > 0

            # Alternatives should have confidence scores
            for alt in rec["alternative_models"]:
                assert "model_name" in alt
                assert "confidence_score" in alt
                assert 0.0 <= alt["confidence_score"] <= 1.0


class TestMethodComparison:
    """Compare different recommendation methods."""

    def test_all_methods_with_same_dataset(self, client, titanic_like_csv):
        """Compare rule-based, llm, and hybrid on same dataset."""
        methods = ["rule_based", "hybrid"]  # Skip "llm" if not available

        results = {}

        for method in methods:
            # Recreate file for each request
            np.random.seed(42)
            from scipy.stats import lognorm
            df = pd.DataFrame({
                "PassengerId": range(1, 715),
                "Survived": np.random.choice([0, 1], 714),
                "Pclass": np.random.choice([1, 2, 3], 714),
                "Age": np.random.normal(30, 15, 714).clip(0, 80),
                "SibSp": np.random.choice([0, 1, 2, 3], 714),
                "Parch": np.random.choice([0, 1, 2], 714),
                "Fare": lognorm.rvs(s=0.95, scale=np.exp(3), size=714, random_state=42),
                "Embarked": np.random.choice(["C", "Q", "S"], 714),
            })
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze-and-recommend",
                params={
                    "method": method,
                    "top_n": 3,
                },
                files={"file": ("titanic.csv", csv_buffer, "text/csv")},
            )

            if response.status_code == 200:
                data = response.json()
                results[method] = {
                    "model": data["recommendation"]["recommended_model"]["model_name"],
                    "confidence": data["recommendation"]["recommended_model"]["confidence_score"],
                    "method": data["recommendation"]["method"],
                }

        # All methods should complete successfully
        assert len(results) > 0

        # Rule-based should always work
        assert "rule_based" in results

        # If hybrid worked, check it has valid result
        if "hybrid" in results:
            assert results["hybrid"]["model"] is not None
            assert 0.0 <= results["hybrid"]["confidence"] <= 1.0

        # Different methods may recommend different models (this is expected)
        # But all should be valid recommendations
        for method, result in results.items():
            assert result["model"] in [
                "GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TabTree", "ARF",
                "CTGAN", "TVAE", "PATE-CTGAN", "DPCART", "AIM", "GaussianCopula"
            ]

    def test_method_performance_comparison(self, client, titanic_like_csv):
        """Compare response times of different methods."""
        import time

        times = {}

        # Test rule-based (should be fastest)
        np.random.seed(42)
        from scipy.stats import lognorm
        df = pd.DataFrame({
            "PassengerId": range(1, 715),
            "Survived": np.random.choice([0, 1], 714),
            "Pclass": np.random.choice([1, 2, 3], 714),
            "Age": np.random.normal(30, 15, 714).clip(0, 80),
            "SibSp": np.random.choice([0, 1, 2, 3], 714),
            "Parch": np.random.choice([0, 1, 2], 714),
            "Fare": lognorm.rvs(s=0.95, scale=np.exp(3), size=714, random_state=42),
            "Embarked": np.random.choice(["C", "Q", "S"], 714),
        })
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        start = time.time()
        response = client.post(
            "/analyze-and-recommend",
            params={"method": "rule_based", "top_n": 3},
            files={"file": ("titanic.csv", csv_buffer, "text/csv")},
        )
        times["rule_based"] = time.time() - start

        assert response.status_code == 200

        # Rule-based should be fast (< 5 seconds)
        assert times["rule_based"] < 5.0
