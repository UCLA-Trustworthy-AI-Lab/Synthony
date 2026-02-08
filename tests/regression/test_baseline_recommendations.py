"""
Regression tests for baseline recommendations.

Ensures that known datasets always produce consistent recommendations.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats import lognorm

from synthony.api.server import app
from synthony.benchmark.generators import BenchmarkDatasetGenerator


@pytest.fixture
def client():
    """Create test client for API with startup events triggered."""
    with TestClient(app) as client:
        yield client


# Baseline expectations for benchmark datasets
BASELINE_EXPECTATIONS = {
    "long_tail": {
        "stress_factors": {
            "severe_skew": True,
            "small_data": False,
        },
        "expected_models": ["GReaT", "TabDDPM", "TabSyn", "AutoDiff", "NFlow"],
        "min_confidence": 0.7,
    },
    "needle_haystack": {
        "stress_factors": {
            "zipfian_distribution": True,
            "high_cardinality": True,
        },
        "expected_models": ["GReaT", "TabSyn", "ARF"],
        "min_confidence": 0.7,
    },
    "small_data": {
        "stress_factors": {
            "small_data": True,
        },
        "expected_models": ["ARF", "GaussianCopula"],
        "min_confidence": 0.8,
    },
}


class TestBenchmarkBaselineRecommendations:
    """Test that benchmark datasets produce consistent recommendations."""

    def test_long_tail_baseline(self, client):
        """Long Tail dataset should consistently recommend skew-capable models."""
        # Generate with fixed seed for reproducibility
        df = BenchmarkDatasetGenerator.generate_long_tail(n_rows=1000, seed=42)

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "baseline_long_tail",
                "method": "rule_based",
                "top_n": 5,
            },
            files={"file": ("long_tail.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check stress factors
        profile = data["analysis"]["dataset_profile"]
        expected = BASELINE_EXPECTATIONS["long_tail"]["stress_factors"]

        for factor, expected_value in expected.items():
            assert profile["stress_factors"][factor] == expected_value, \
                f"Stress factor {factor} mismatch: expected {expected_value}, got {profile['stress_factors'][factor]}"

        # Check recommended model is in expected list
        rec_model = data["recommendation"]["recommended_model"]["model_name"]
        expected_models = BASELINE_EXPECTATIONS["long_tail"]["expected_models"]

        assert rec_model in expected_models, \
            f"Unexpected recommendation: {rec_model} not in {expected_models}"

        # Check confidence
        confidence = data["recommendation"]["recommended_model"]["confidence_score"]
        min_conf = BASELINE_EXPECTATIONS["long_tail"]["min_confidence"]

        assert confidence >= min_conf, \
            f"Confidence {confidence} below minimum {min_conf}"

    def test_needle_haystack_baseline(self, client):
        """Needle in Haystack should consistently detect Zipfian and recommend appropriate models."""
        df = BenchmarkDatasetGenerator.generate_needle_in_haystack(n_rows=1000, seed=42)

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "baseline_needle",
                "method": "rule_based",
                "top_n": 5,
            },
            files={"file": ("needle.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check stress factors
        profile = data["analysis"]["dataset_profile"]
        expected = BASELINE_EXPECTATIONS["needle_haystack"]["stress_factors"]

        for factor, expected_value in expected.items():
            assert profile["stress_factors"][factor] == expected_value

        # Check recommendation
        rec_model = data["recommendation"]["recommended_model"]["model_name"]
        expected_models = BASELINE_EXPECTATIONS["needle_haystack"]["expected_models"]

        assert rec_model in expected_models

    def test_small_data_baseline(self, client):
        """Small Data Trap should consistently recommend ARF or GaussianCopula."""
        df = BenchmarkDatasetGenerator.generate_small_data_trap(n_rows=200, seed=42)

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "baseline_small",
                "method": "rule_based",
                "top_n": 5,
            },
            files={"file": ("small.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check stress factors
        profile = data["analysis"]["dataset_profile"]
        expected = BASELINE_EXPECTATIONS["small_data"]["stress_factors"]

        for factor, expected_value in expected.items():
            assert profile["stress_factors"][factor] == expected_value

        # For small data, check top recommendations include ARF or GaussianCopula
        rec_model = data["recommendation"]["recommended_model"]["model_name"]
        all_models = [rec_model]

        if "alternative_models" in data["recommendation"]:
            all_models.extend([
                alt["model_name"]
                for alt in data["recommendation"]["alternative_models"]
            ])

        expected_models = BASELINE_EXPECTATIONS["small_data"]["expected_models"]
        assert any(model in expected_models for model in all_models), \
            f"Expected one of {expected_models} in recommendations, got {all_models}"


class TestKnownDatasetConsistency:
    """Test that known datasets produce consistent results across runs."""

    @pytest.fixture
    def titanic_like_data(self):
        """Create Titanic-like dataset with fixed seed."""
        np.random.seed(42)
        return pd.DataFrame({
            "PassengerId": range(1, 715),
            "Survived": np.random.choice([0, 1], 714),
            "Pclass": np.random.choice([1, 2, 3], 714),
            "Age": np.random.normal(30, 15, 714).clip(0, 80),
            "Fare": lognorm.rvs(s=0.95, scale=np.exp(3), size=714, random_state=42),
            "Embarked": np.random.choice(["C", "Q", "S"], 714),
        })

    def test_titanic_deterministic_recommendation(self, client, titanic_like_data):
        """Titanic dataset should produce same recommendation every time."""
        results = []

        # Run 3 times
        for i in range(3):
            csv_buffer = io.BytesIO()
            titanic_like_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze-and-recommend",
                params={
                    "dataset_id": f"titanic_run_{i}",
                    "method": "rule_based",
                    "top_n": 3,
                },
                files={"file": ("titanic.csv", csv_buffer, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()

            results.append({
                "model": data["recommendation"]["recommended_model"]["model_name"],
                "confidence": data["recommendation"]["recommended_model"]["confidence_score"],
            })

        # All runs should produce identical results
        assert results[0]["model"] == results[1]["model"] == results[2]["model"]
        assert results[0]["confidence"] == results[1]["confidence"] == results[2]["confidence"]

    def test_stress_factors_consistency(self, client, titanic_like_data):
        """Stress factor detection should be consistent."""
        stress_factors_list = []

        for i in range(2):
            csv_buffer = io.BytesIO()
            titanic_like_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze",
                params={"dataset_id": f"test_{i}"},
                files={"file": ("titanic.csv", csv_buffer, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()

            stress_factors_list.append(data["dataset_profile"]["stress_factors"])

        # Stress factors should be identical
        assert stress_factors_list[0] == stress_factors_list[1]


class TestConstraintConsistency:
    """Test that constraints produce consistent filtering."""

    @pytest.fixture
    def test_data(self):
        """Create test dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            "col1": np.random.randn(1000),
            "col2": lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42),
            "col3": np.random.choice(["A", "B", "C"], 1000),
        })

    def test_cpu_constraint_consistency(self, client, test_data):
        """CPU constraint should consistently exclude GPU models."""
        # Run 3 times
        for i in range(3):
            csv_buffer = io.BytesIO()
            test_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze-and-recommend",
                params={
                    "dataset_id": f"cpu_test_{i}",
                    "method": "rule_based",
                    "cpu_only": True,
                    "top_n": 5,
                },
                files={"file": ("test.csv", csv_buffer, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()

            # GPU models should NEVER be recommended with cpu_only=True
            gpu_models = {"TabDDPM", "TabSyn", "GReaT"}

            rec_model = data["recommendation"]["recommended_model"]["model_name"]
            assert rec_model not in gpu_models

            # Check alternatives
            if "alternative_models" in data["recommendation"]:
                for alt in data["recommendation"]["alternative_models"]:
                    assert alt["model_name"] not in gpu_models

    def test_dp_constraint_consistency(self, client, test_data):
        """DP constraint should consistently include only DP models."""
        for i in range(3):
            csv_buffer = io.BytesIO()
            test_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze-and-recommend",
                params={
                    "dataset_id": f"dp_test_{i}",
                    "method": "rule_based",
                    "cpu_only": False,
                    "strict_dp": True,
                    "top_n": 3,
                },
                files={"file": ("test.csv", csv_buffer, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()

            # Only DP models should be recommended
            dp_models = {"PATE-CTGAN", "PATECTGAN", "AIM", "DPCART"}

            rec_model = data["recommendation"]["recommended_model"]["model_name"]
            assert rec_model in dp_models

            # Check alternatives
            if "alternative_models" in data["recommendation"]:
                for alt in data["recommendation"]["alternative_models"]:
                    assert alt["model_name"] in dp_models



class TestConfidenceScoreRegression:
    """Test that confidence scores remain in expected ranges."""

    def test_high_confidence_for_clear_cases(self, client):
        """Clear-cut cases should have high confidence."""
        # Very small dataset (clear recommendation: ARF or GaussianCopula)
        df = pd.DataFrame({
            "value": np.random.randn(100)
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("clear_case.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have high confidence for clear small data case
        confidence = data["recommendation"]["recommended_model"]["confidence_score"]
        assert confidence >= 0.7

    def test_confidence_bounds(self, client):
        """All confidence scores should be in valid range."""
        # Test with various datasets
        test_datasets = [
            pd.DataFrame({"x": np.random.randn(100)}),
            pd.DataFrame({"x": np.random.randn(5000)}),
            pd.DataFrame({"x": range(1000), "y": np.random.choice(["A", "B"], 1000)}),
        ]

        for idx, df in enumerate(test_datasets):
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = client.post(
                "/analyze-and-recommend",
                params={
                    "dataset_id": f"confidence_test_{idx}",
                    "method": "rule_based",
                    "top_n": 5,
                },
                files={"file": ("test.csv", csv_buffer, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()

            # Check all confidence scores
            rec_conf = data["recommendation"]["recommended_model"]["confidence_score"]
            assert 0.0 <= rec_conf <= 1.0

            if "alternative_models" in data["recommendation"]:
                for alt in data["recommendation"]["alternative_models"]:
                    assert 0.0 <= alt["confidence_score"] <= 1.0
