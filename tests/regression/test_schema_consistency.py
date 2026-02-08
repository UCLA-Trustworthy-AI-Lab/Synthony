"""
Regression tests for API response schema consistency.

Ensures API responses maintain backward compatibility.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthony.api.server import app


@pytest.fixture
def client():
    """Create test client for API with startup events triggered."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_data():
    """Sample dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1000),
        "value": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })


class TestHealthEndpointSchema:
    """Test health endpoint response schema."""

    def test_health_response_structure(self, client):
        """Health endpoint should return expected fields."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "status" in data
        assert "llm_available" in data
        assert "models_count" in data

        # Field types
        assert isinstance(data["status"], str)
        assert isinstance(data["llm_available"], bool)
        assert isinstance(data["models_count"], int)

        # Valid values
        assert data["status"] in ["healthy", "unhealthy"]
        assert data["models_count"] > 0


class TestModelsEndpointSchema:
    """Test models endpoint response schema."""

    def test_list_models_schema(self, client):
        """Models list endpoint should return consistent schema."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        # Required top-level fields
        assert "models" in data

        # Check models dict (API returns dict, not list)
        assert isinstance(data["models"], dict)
        assert len(data["models"]) > 0

        # Check each model has required fields
        for model_name, model_info in data["models"].items():
            assert "name" in model_info
            assert "type" in model_info

            # Field types
            assert isinstance(model_info["name"], str)
            assert isinstance(model_info["type"], str)

    def test_get_model_details_schema(self, client):
        """Model details endpoint should return consistent schema."""
        response = client.get("/models/GReaT")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = [
            "model_name",
            "type",  # API returns 'type' not 'model_type'
            "capabilities",
        ]

        for field in required_fields:
            assert field in data

        # Check capabilities structure
        assert isinstance(data["capabilities"], dict)


class TestAnalyzeEndpointSchema:
    """Test analyze endpoint response schema."""

    def test_analyze_response_schema(self, client, sample_data):
        """Analyze endpoint should return consistent schema."""
        csv_buffer = io.BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Required top-level fields
        assert "dataset_profile" in data
        profile = data["dataset_profile"]

        # Required profile fields
        required_profile_fields = [
            "dataset_id",
            "row_count",
            "column_count",
            "stress_factors",
        ]

        for field in required_profile_fields:
            assert field in profile

        # Field types
        assert isinstance(profile["dataset_id"], str)
        assert isinstance(profile["row_count"], int)
        assert isinstance(profile["column_count"], int)
        assert isinstance(profile["stress_factors"], dict)

        # Stress factors structure
        required_stress_factors = [
            "severe_skew",
            "high_cardinality",
            "zipfian_distribution",
            "higher_order_correlation",
            "small_data",
            "large_data",
        ]

        for factor in required_stress_factors:
            assert factor in profile["stress_factors"]
            assert isinstance(profile["stress_factors"][factor], bool)


class TestRecommendEndpointSchema:
    """Test recommend endpoint response schema."""

    def test_recommend_response_schema(self, client, sample_data):
        """Recommend endpoint should return consistent schema."""
        # First analyze
        csv_buffer = io.BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        analyze_response = client.post(
            "/analyze",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        profile = analyze_response.json()["dataset_profile"]

        # Then recommend (include required dataset_id)
        recommend_response = client.post(
            "/recommend",
            json={
                "dataset_id": analyze_response.json().get("dataset_id", "test"),
                "dataset_profile": profile,
                "method": "rule_based",
                "top_n": 3,
            },
        )

        assert recommend_response.status_code == 200
        data = recommend_response.json()

        # Required top-level fields
        assert "method" in data
        assert "recommended_model" in data

        # Check recommended model structure
        rec = data["recommended_model"]
        required_rec_fields = [
            "model_name",
            "confidence_score",
            "reasoning",
        ]

        for field in required_rec_fields:
            assert field in rec

        # Field types
        assert isinstance(rec["model_name"], str)
        # model_type may or may not be present
        assert isinstance(rec["confidence_score"], (int, float))
        assert isinstance(rec["reasoning"], list)

        # Confidence score range
        assert 0.0 <= rec["confidence_score"] <= 1.0

        # Reasoning should be non-empty
        assert len(rec["reasoning"]) > 0
        for reason in rec["reasoning"]:
            assert isinstance(reason, str)
            assert len(reason) > 0

        # Check alternatives if present
        if "alternative_models" in data:
            assert isinstance(data["alternative_models"], list)

            for alt in data["alternative_models"]:
                assert "model_name" in alt
                assert "confidence_score" in alt
                assert 0.0 <= alt["confidence_score"] <= 1.0


class TestAnalyzeAndRecommendSchema:
    """Test one-shot analyze-and-recommend endpoint schema."""

    def test_one_shot_response_schema(self, client, sample_data):
        """One-shot endpoint should return consistent schema."""
        csv_buffer = io.BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Required top-level fields
        assert "analysis" in data
        assert "recommendation" in data

        # Analysis should have dataset_profile
        assert "dataset_profile" in data["analysis"]

        # Recommendation should have recommended_model
        assert "recommended_model" in data["recommendation"]
        assert "method" in data["recommendation"]

        # Check nested structures match schemas from other endpoints
        profile = data["analysis"]["dataset_profile"]
        assert "stress_factors" in profile
        assert "row_count" in profile
        assert "column_count" in profile

        rec = data["recommendation"]["recommended_model"]
        assert "model_name" in rec
        assert "confidence_score" in rec
        assert "reasoning" in rec


class TestBackwardCompatibility:
    """Test backward compatibility of API responses."""

    def test_no_removed_fields(self, client, sample_data):
        """Ensure no previously existing fields have been removed."""
        # This test documents the expected API schema
        # If this test fails, it means backward compatibility was broken

        csv_buffer = io.BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Expected schema (version 0.2.0 - constraints removed)
        expected_structure = {
            "analysis": {
                "dataset_profile": {
                    "dataset_id": str,
                    "row_count": int,
                    "column_count": int,
                    "stress_factors": {
                        "severe_skew": bool,
                        "high_cardinality": bool,
                        "zipfian_distribution": bool,
                        "higher_order_correlation": bool,
                        "small_data": bool,
                        "large_data": bool,
                    },
                },
            },
            "recommendation": {
                "method": str,
                "recommended_model": {
                    "model_name": str,
                    "confidence_score": (int, float),
                    "reasoning": list,
                },
            },
        }

        def check_structure(actual, expected, path=""):
            """Recursively check structure matches."""
            for key, expected_type in expected.items():
                assert key in actual, f"Missing field: {path}.{key}"

                actual_value = actual[key]

                if isinstance(expected_type, dict):
                    # Nested structure
                    assert isinstance(actual_value, dict), \
                        f"Field {path}.{key} should be dict, got {type(actual_value)}"
                    check_structure(actual_value, expected_type, f"{path}.{key}")
                elif isinstance(expected_type, tuple):
                    # Multiple allowed types
                    assert isinstance(actual_value, expected_type), \
                        f"Field {path}.{key} has wrong type: expected {expected_type}, got {type(actual_value)}"
                else:
                    # Single type
                    assert isinstance(actual_value, expected_type), \
                        f"Field {path}.{key} has wrong type: expected {expected_type}, got {type(actual_value)}"

        check_structure(data, expected_structure)

    def test_additional_fields_allowed(self, client, sample_data):
        """New fields can be added without breaking compatibility."""
        csv_buffer = io.BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # It's okay to have additional fields beyond the minimum required
        # This test just documents that additional fields are acceptable

        # Check we at least have the core required fields
        assert "analysis" in data
        assert "recommendation" in data


class TestErrorResponseConsistency:
    """Test error responses have consistent structure."""

    def test_404_error_structure(self, client):
        """404 errors should have consistent structure."""
        response = client.get("/models/NonExistentModel")

        assert response.status_code == 404
        data = response.json()

        # Should have error details
        assert "detail" in data

    def test_422_validation_error_structure(self, client):
        """422 validation errors should have consistent structure."""
        # Send invalid request (missing required fields)
        response = client.post(
            "/recommend",
            json={
                # Missing required 'dataset_profile' field
            },
        )

        assert response.status_code == 422
        data = response.json()

        # Should have validation error details
        assert "detail" in data
