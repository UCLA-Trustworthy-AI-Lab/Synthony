#!/usr/bin/env python3
"""
Test script for Synthony API server.

Tests all API endpoints with sample data.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_health():
    """Test health check endpoint."""
    print_section("1. Testing Health Check")

    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"

    health = response.json()
    print(f"✓ Server status: {health['status']}")
    print(f"✓ Version: {health['version']}")
    print(f"✓ Analyzer available: {health['analyzer_available']}")
    print(f"✓ Recommender available: {health['recommender_available']}")
    print(f"✓ LLM available: {health['llm_available']}")
    print(f"✓ Models count: {health['models_count']}")

    return health


def test_list_models():
    """Test list models endpoint."""
    print_section("2. Testing List Models")

    # Test all models
    response = requests.get(f"{BASE_URL}/models")
    assert response.status_code == 200, f"List models failed: {response.status_code}"

    models = response.json()
    print(f"✓ Total models: {models['total_models']}")
    print(f"✓ Model names: {list(models['models'].keys())}")

    return models


def test_get_model_info():
    """Test get model info endpoint."""
    print_section("3. Testing Get Model Info")

    model_name = "GReaT"
    response = requests.get(f"{BASE_URL}/models/{model_name}")
    assert response.status_code == 200, f"Get model info failed: {response.status_code}"

    model_info = response.json()
    print(f"✓ Model: {model_info['model_name']}")
    print(f"✓ Type: {model_info['type']}")
    print(f"✓ Capabilities: {model_info['capabilities']}")
    print(f"✓ Strengths: {len(model_info['strengths'])} items")
    print(f"✓ Limitations: {len(model_info['limitations'])} items")

    return model_info


def create_test_dataset():
    """Create a test CSV dataset with stress factors."""
    print_section("4. Creating Test Dataset")

    # Create a dataset with known stress factors
    import numpy as np
    from scipy.stats import lognorm

    np.random.seed(42)

    # Generate data with severe skew (LogNormal)
    skewed_data = lognorm(s=0.95, scale=np.exp(5)).rvs(1000)

    # Generate high cardinality categorical
    categories = [f"cat_{i}" for i in range(600)]
    cat_data = np.random.choice(categories, size=1000)

    # Generate Zipfian distribution
    # Top 20% categories should have >80% of data
    zipf_categories = [f"zipf_{i}" for i in range(100)]
    zipf_probs = np.array([1 / (i + 1) ** 1.5 for i in range(100)])
    zipf_probs = zipf_probs / zipf_probs.sum()
    zipf_data = np.random.choice(zipf_categories, size=1000, p=zipf_probs)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "skewed_column": skewed_data,
            "high_cardinality": cat_data,
            "zipfian_column": zipf_data,
            "normal_column": np.random.randn(1000),
            "id": range(1000),
        }
    )

    # Save to temp file
    test_file = Path("test_data.csv")
    df.to_csv(test_file, index=False)

    print(f"✓ Created test dataset: {test_file}")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Skewness of skewed_column: {df['skewed_column'].skew():.2f}")
    print(f"✓ Cardinality of high_cardinality: {df['high_cardinality'].nunique()}")
    print(f"✓ Unique values in zipfian_column: {df['zipfian_column'].nunique()}")

    return test_file


def test_analyze(test_file: Path):
    """Test analyze endpoint."""
    print_section("5. Testing Analyze Endpoint")

    with open(test_file, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/analyze",
            params={"dataset_id": "test_dataset"},
            files={"file": f},
        )

    assert response.status_code == 200, f"Analyze failed: {response.status_code}"

    result = response.json()
    print(f"✓ Dataset ID: {result['dataset_id']}")
    print(f"✓ Rows: {result['dataset_profile']['row_count']}")
    print(f"✓ Columns: {result['dataset_profile']['column_count']}")

    stress_factors = result["dataset_profile"]["stress_factors"]
    active_stress = [k for k, v in stress_factors.items() if v]
    print(f"✓ Active stress factors: {active_stress}")

    print(f"✓ Max column difficulty: {result['column_analysis']['max_column_difficulty']}")
    print(f"✓ Difficult columns: {result['column_analysis']['difficult_columns']}")

    return result


def test_recommend(analysis_result: dict, method: str = "rule_based"):
    """Test recommend endpoint."""
    print_section(f"6. Testing Recommend Endpoint (method={method})")

    request_body = {
        "dataset_id": analysis_result["dataset_id"],
        "dataset_profile": analysis_result["dataset_profile"],
        "column_analysis": analysis_result["column_analysis"],
        "method": method,
        "top_n": 3,
    }

    response = requests.post(f"{BASE_URL}/recommend", json=request_body)

    assert response.status_code == 200, f"Recommend failed: {response.status_code}"

    result = response.json()
    print(f"✓ Method used: {result['method']}")
    print(f"✓ Recommended model: {result['recommended_model']['model_name']}")
    print(f"✓ Confidence: {result['recommended_model']['confidence_score']:.2f}")
    print(f"✓ Reasoning:")
    for reason in result["recommended_model"]["reasoning"][:3]:
        print(f"    {reason}")

    print(f"✓ Alternatives: {[m['model_name'] for m in result['alternative_models']]}")

    if result["recommended_model"]["warnings"]:
        print(f"✓ Warnings:")
        for warning in result["recommended_model"]["warnings"][:2]:
            print(f"    {warning}")

    excluded = result.get("excluded_models", {})
    if excluded:
        print(f"✓ Excluded models: {len(excluded)}")

    return result


def test_analyze_and_recommend(test_file: Path):
    """Test one-shot analyze-and-recommend endpoint."""
    print_section("7. Testing Analyze-and-Recommend (One-Shot)")

    with open(test_file, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/analyze-and-recommend",
            params={
                "dataset_id": "test_dataset_oneshot",
                "method": "hybrid",
                "top_n": 3,
            },
            files={"file": f},
        )

    assert (
        response.status_code == 200
    ), f"Analyze-and-recommend failed: {response.status_code}"

    result = response.json()
    print(f"✓ Dataset ID: {result['dataset_id']}")
    print(f"✓ Analysis completed")
    print(f"✓ Recommendation method: {result['recommendation']['method']}")
    print(f"✓ Recommended model: {result['recommendation']['recommended_model']['model_name']}")
    print(
        f"✓ Confidence: {result['recommendation']['recommended_model']['confidence_score']:.2f}"
    )

    return result


def save_results(results: dict, filename: str = "api_test_results.json"):
    """Save test results to JSON file."""
    print_section("Saving Results")

    output_path = Path(filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path.absolute()}")


def main():
    """Run all API tests."""
    print("\n" + "=" * 80)
    print(" Synthony API Test Suite")
    print("=" * 80)
    print(f"\nTesting server at: {BASE_URL}")

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"\n❌ Server is not responding properly: {response.status_code}")
            print("Please start the server with: synthony-api")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to server at {BASE_URL}")
        print("Please start the server with: synthony-api")
        sys.exit(1)

    results = {}

    try:
        # Run tests
        results["health"] = test_health()
        time.sleep(0.5)

        results["models"] = test_list_models()
        time.sleep(0.5)

        results["model_info"] = test_get_model_info()
        time.sleep(0.5)

        test_file = create_test_dataset()
        time.sleep(0.5)

        results["analysis"] = test_analyze(test_file)
        time.sleep(0.5)

        results["recommendation_rule_based"] = test_recommend(results["analysis"], method="rule_based")
        time.sleep(0.5)

        # Test hybrid mode (will fall back to rule_based if no OpenAI key)
        results["recommendation_hybrid"] = test_recommend(results["analysis"], method="hybrid")
        time.sleep(0.5)

        results["one_shot"] = test_analyze_and_recommend(test_file)
        time.sleep(0.5)

        # Clean up test file
        test_file.unlink()
        print(f"\n✓ Cleaned up test file: {test_file}")

        # Save results
        save_results(results)

        print_section("✅ All Tests Passed!")
        print("\nAPI is working correctly. You can now:")
        print("  1. View interactive docs: http://localhost:8000/docs")
        print("  2. Test with your own data using the examples in docs/API_USAGE.md")
        print("  3. Set OPENAI_API_KEY to enable LLM mode")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
