"""
Integration tests for end-to-end scenarios.

Tests complete workflows from file upload through analysis to final recommendation.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthony.api.server import app
from synthony.benchmark.generators import BenchmarkDatasetGenerator


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


class TestBenchmarkDatasetWorkflows:
    """Test complete workflows with benchmark datasets."""

    def test_long_tail_dataset_workflow(self, client):
        """Complete workflow with Long Tail benchmark dataset."""
        # Generate benchmark dataset
        df = BenchmarkDatasetGenerator.generate_long_tail(n_rows=1000, seed=42)

        # Convert to CSV
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Step 1: One-shot analyze and recommend
        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "benchmark_long_tail",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("long_tail.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify analysis detected severe skew
        assert data["analysis"]["dataset_profile"]["stress_factors"]["severe_skew"] is True

        # Verify recommendation considers skew
        rec = data["recommendation"]["recommended_model"]

        # Should recommend models good at handling skew
        # GReaT, TabDDPM, TabSyn, AutoDiff, TabTree have skew capability >= 3
        skew_capable_models = {"GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TabTree"}
        assert rec["model_name"] in skew_capable_models

        # Reasoning should mention skew
        reasoning_text = " ".join(rec["reasoning"]).lower()
        assert "skew" in reasoning_text or "tail" in reasoning_text

    def test_needle_in_haystack_workflow(self, client):
        """Complete workflow with Needle in Haystack benchmark dataset."""
        # Generate benchmark dataset
        df = BenchmarkDatasetGenerator.generate_needle_in_haystack(n_rows=1000, seed=42)

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "benchmark_needle",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("needle.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify Zipfian detection
        assert data["analysis"]["dataset_profile"]["stress_factors"]["zipfian_distribution"] is True

        # Should recommend models good at Zipfian handling
        rec = data["recommendation"]["recommended_model"]

        # GReaT, TabSyn, TabTree, ARF have good Zipfian capability
        zipfian_capable_models = {"GReaT", "TabSyn", "TabTree", "ARF"}
        assert rec["model_name"] in zipfian_capable_models

    def test_small_data_trap_workflow(self, client):
        """Complete workflow with Small Data Trap benchmark dataset."""
        # Generate benchmark dataset
        df = BenchmarkDatasetGenerator.generate_small_data_trap(n_rows=200, seed=42)

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "benchmark_small",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("small.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify small data detection
        assert data["analysis"]["dataset_profile"]["stress_factors"]["small_data"] is True
        assert data["analysis"]["dataset_profile"]["row_count"] == 200

        # Should recommend ARF or GaussianCopula (best for small data)
        rec = data["recommendation"]["recommended_model"]
        all_models = [rec["model_name"]]
        if "alternative_models" in data["recommendation"]:
            all_models.extend([
                alt["model_name"]
                for alt in data["recommendation"]["alternative_models"]
            ])

        assert "ARF" in all_models or "GaussianCopula" in all_models


class TestRealWorldScenarios:
    """Test realistic data science workflows."""

    def test_exploratory_analysis_workflow(self, client):
        """Simulate exploratory data analysis workflow."""
        # Create realistic dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "user_id": range(5000),
            "age": np.random.normal(35, 12, 5000).clip(18, 80),
            "income": np.random.lognormal(10, 1, 5000),
            "purchases": np.random.poisson(5, 5000),
            "region": np.random.choice(["North", "South", "East", "West"], 5000),
            "is_premium": np.random.choice([0, 1], 5000, p=[0.7, 0.3]),
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Step 1: Analyze dataset characteristics
        analyze_response = client.post(
            "/analyze",
            params={"dataset_id": "customer_data"},
            files={"file": ("customers.csv", csv_buffer, "text/csv")},
        )

        assert analyze_response.status_code == 200
        analysis = analyze_response.json()

        profile = analysis["dataset_profile"]
        assert profile["row_count"] == 5000
        assert profile["column_count"] == 6

        # Step 2: Get model recommendations
        recommend_response = client.post(
            "/recommend",
            json={
                "dataset_profile": profile,
                "method": "rule_based",
                "top_n": 5,
            },
        )

        assert recommend_response.status_code == 200
        recommendation = recommend_response.json()

        # Should have alternatives
        assert "alternative_models" in recommendation
        assert len(recommendation["alternative_models"]) > 0

    def test_production_deployment_workflow(self, client):
        """Simulate production deployment workflow."""
        # Production dataset (larger)
        np.random.seed(42)
        df = pd.DataFrame({
            "transaction_id": range(10000),
            "amount": np.random.lognormal(5, 2, 10000),
            "merchant_category": np.random.choice([f"cat_{i}" for i in range(50)], 10000),
            "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
            "is_fraud": np.random.choice([0, 1], 10000, p=[0.98, 0.02]),
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/analyze-and-recommend",
            params={
                "dataset_id": "prod_transactions",
                "method": "rule_based",
                "top_n": 3,
            },
            files={"file": ("transactions.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify production-ready recommendations
        rec = data["recommendation"]["recommended_model"]

        # Should have high confidence
        assert rec["confidence_score"] >= 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_dataset(self, client):
        """Test with extremely small dataset (< 50 rows)."""
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5] * 5  # 25 rows
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
            files={"file": ("tiny.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect as small data
        assert data["analysis"]["dataset_profile"]["stress_factors"]["small_data"] is True

        # Should recommend models suitable for small data
        rec = data["recommendation"]["recommended_model"]
        all_models = [rec["model_name"]]
        if "alternative_models" in data["recommendation"]:
            all_models.extend([alt["model_name"] for alt in data["recommendation"]["alternative_models"]])

        # ARF or GaussianCopula should be recommended
        assert "ARF" in all_models or "GaussianCopula" in all_models

    def test_single_column_dataset(self, client):
        """Test with single-column dataset."""
        df = pd.DataFrame({
            "value": np.random.randn(1000)
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
            files={"file": ("single_col.csv", csv_buffer, "text/csv")},
        )

        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()

        assert data["analysis"]["dataset_profile"]["column_count"] == 1
        assert "recommendation" in data

    def test_all_categorical_dataset(self, client):
        """Test with dataset containing only categorical columns."""
        df = pd.DataFrame({
            "cat1": np.random.choice(["A", "B", "C"], 1000),
            "cat2": np.random.choice(["X", "Y", "Z"], 1000),
            "cat3": np.random.choice(["1", "2", "3"], 1000),
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
            files={"file": ("all_cat.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should handle and provide recommendation
        assert "recommendation" in data
        assert data["recommendation"]["recommended_model"] is not None
