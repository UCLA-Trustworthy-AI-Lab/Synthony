"""
Functional tests for API workflows.

Tests complete user workflows: upload CSV → analyze → recommend → get results.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from synthony.api.server import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for upload."""
    df = pd.DataFrame({
        "id": range(1000),
        "value": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })

    # Convert to bytes for file upload
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return ("test.csv", csv_buffer, "text/csv")


@pytest.fixture
def skewed_csv_file():
    """Create skewed dataset CSV."""
    from scipy.stats import lognorm

    np.random.seed(42)
    df = pd.DataFrame({
        "skewed_col": lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42),
        "normal_col": np.random.randn(1000),
    })

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return ("skewed.csv", csv_buffer, "text/csv")


@pytest.fixture
def small_csv_file():
    """Create small dataset CSV (<500 rows)."""
    df = pd.DataFrame({
        "value": np.random.randn(200),
        "category": np.random.choice(["A", "B"], 200),
    })

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return ("small.csv", csv_buffer, "text/csv")


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return server status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "llm_available" in data
        assert "models_count" in data
        assert data["models_count"] > 0


class TestModelEndpoints:
    """Test model listing and details endpoints."""

    def test_list_all_models(self, client):
        """List all models endpoint."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert len(data["models"]) > 0

        # Check model structure
        for model in data["models"]:
            assert "name" in model
            assert "type" in model
            assert "supports_gpu" in model
            assert "supports_dp" in model

    def test_get_model_details(self, client):
        """Get details for a specific model."""
        response = client.get("/models/GReaT")

        assert response.status_code == 200
        data = response.json()

        assert data["model_name"] == "GReaT"
        assert "model_type" in data
        assert "supports_gpu" in data
        assert "supports_dp" in data
        assert "capabilities" in data

    def test_get_nonexistent_model(self, client):
        """Request for non-existent model should return 404."""
        response = client.get("/models/NonExistentModel")

        assert response.status_code == 404


class TestAnalyzeEndpoint:
    """Test dataset analysis endpoint."""

    def test_analyze_csv_file(self, client, sample_csv_file):
        """Analyze a CSV file."""
        response = client.post(
            "/analyze",
            files={"file": sample_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Check profile structure
        assert "dataset_profile" in data
        profile = data["dataset_profile"]

        assert profile["row_count"] == 1000
        assert profile["column_count"] == 3
        assert "stress_factors" in profile
        assert "dataset_id" in profile

    def test_analyze_with_dataset_id(self, client, sample_csv_file):
        """Analyze with custom dataset ID."""
        response = client.post(
            "/analyze?dataset_id=my_custom_id",
            files={"file": sample_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["dataset_profile"]["dataset_id"] == "my_custom_id"

    def test_analyze_skewed_data(self, client, skewed_csv_file):
        """Analyze skewed dataset."""
        response = client.post(
            "/analyze",
            files={"file": skewed_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect severe skew
        assert data["dataset_profile"]["stress_factors"]["severe_skew"] is True

    def test_analyze_small_data(self, client, small_csv_file):
        """Analyze small dataset."""
        response = client.post(
            "/analyze",
            files={"file": small_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect small data
        assert data["dataset_profile"]["stress_factors"]["small_data"] is True
        assert data["dataset_profile"]["row_count"] == 200

    def test_analyze_invalid_file(self, client):
        """Analyze with invalid file should return error."""
        # Empty file
        response = client.post(
            "/analyze",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )

        # Should return error (400 or 422)
        assert response.status_code in [400, 422, 500]


class TestRecommendEndpoint:
    """Test model recommendation endpoint."""

    def test_recommend_with_profile(self, client, sample_csv_file):
        """Two-step workflow: analyze then recommend."""
        # Step 1: Analyze
        analyze_response = client.post(
            "/analyze",
            files={"file": sample_csv_file},
        )

        assert analyze_response.status_code == 200
        analysis = analyze_response.json()

        # Step 2: Recommend
        recommend_response = client.post(
            "/recommend",
            json={
                "dataset_profile": analysis["dataset_profile"],
                "method": "rule_based",
                "top_n": 3,
            },
        )

        assert recommend_response.status_code == 200
        data = recommend_response.json()

        # Check recommendation structure
        assert "recommended_model" in data
        rec = data["recommended_model"]

        assert "model_name" in rec
        assert "confidence_score" in rec
        assert "reasoning" in rec
        assert len(rec["reasoning"]) > 0


class TestAnalyzeAndRecommendEndpoint:
    """Test one-shot analyze-and-recommend endpoint."""

    def test_one_shot_workflow(self, client, sample_csv_file):
        """One-shot workflow: upload → analyze → recommend."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": sample_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have both analysis and recommendation
        assert "analysis" in data
        assert "recommendation" in data

        # Check analysis
        assert "dataset_profile" in data["analysis"]
        profile = data["analysis"]["dataset_profile"]
        assert profile["row_count"] == 1000

        # Check recommendation
        assert "recommended_model" in data["recommendation"]
        rec = data["recommendation"]["recommended_model"]
        assert "model_name" in rec
        assert "confidence_score" in rec

    def test_one_shot_with_small_data(self, client, small_csv_file):
        """One-shot with small dataset."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": small_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect small data
        assert data["analysis"]["dataset_profile"]["stress_factors"]["small_data"] is True

        # Recommendation should consider small data
        # ARF and GaussianCopula are best for small data
        all_models = [data["recommendation"]["recommended_model"]["model_name"]]
        if "alternative_models" in data["recommendation"]:
            all_models.extend([
                alt["model_name"]
                for alt in data["recommendation"]["alternative_models"]
            ])

        # Should recommend or include ARF or GaussianCopula
        assert "ARF" in all_models or "GaussianCopula" in all_models

    def test_one_shot_with_skewed_data(self, client, skewed_csv_file):
        """One-shot with skewed dataset."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": skewed_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect severe skew
        assert data["analysis"]["dataset_profile"]["stress_factors"]["severe_skew"] is True

        # Should recommend models good at handling skew
        # GReaT, TabDDPM, TabSyn, AutoDiff, TabTree have good skew handling
        rec_model = data["recommendation"]["recommended_model"]["model_name"]
        skew_capable_models = {"GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TabTree", "ARF"}
        assert rec_model in skew_capable_models

    def test_one_shot_hybrid_mode_fallback(self, client, sample_csv_file):
        """Hybrid mode should fall back to rule-based if LLM unavailable."""
        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "hybrid",
                "top_n": 3,
            },
            files={"file": sample_csv_file},
        )

        assert response.status_code == 200
        data = response.json()

        # Should complete successfully (may fall back to rule-based)
        assert "recommendation" in data
        assert "recommended_model" in data["recommendation"]

        # Method should be either "hybrid" or "rule_based" (fallback)
        assert data["recommendation"]["method"] in ["hybrid", "rule_based"]
