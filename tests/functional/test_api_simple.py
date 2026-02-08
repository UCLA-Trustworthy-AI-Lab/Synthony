#!/usr/bin/env python3
"""Simple, focused API test."""

import sys

import requests

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Test 1: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   LLM available: {data['llm_available']}")
    print(f"   Models: {data['models_count']}\n")

def test_list_models():
    """Test list models endpoint."""
    print("Test 2: List Models")
    response = requests.get(f"{BASE_URL}/models")
    data = response.json()
    print(f"   Total: {len(data['models'])} models")
    print(f"   Models: {', '.join([m['name'] for m in data['models'][:4]])}...\n")

def test_get_model():
    """Test get specific model endpoint."""
    print("Test 3: Get Model Info (TabTree)")
    response = requests.get(f"{BASE_URL}/models/TabTree")
    data = response.json()
    print(f"   Model: {data['model_name']}")
    print(f"   Type: {data['type']}")
    print(f"   Training speed: {data['performance']['training_speed']}\n")

def test_analyze_and_recommend():
    """Test one-shot analyze-and-recommend endpoint."""
    print("Test 4: Analyze & Recommend (Titanic)")

    with open("/Users/hochan.son/Project/Synthony/dataset/input_data/Titanic.csv", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/analyze-and-recommend",
            params={
                "dataset_id": "titanic_simple_test",
                "method": "rule_based",
                "top_n": 3
            },
            files={"file": f}
        )

    if response.status_code != 200:
        print(f"   Error {response.status_code}: {response.text}")
        return

    result = response.json()

    # Show analysis
    profile = result["analysis"]["dataset_profile"]
    col_analysis = result["analysis"]["column_analysis"]
    print(f"   Dataset: {profile['row_count']} rows x {profile['column_count']} columns")
    print(f"   Max difficulty: {col_analysis['max_column_difficulty']}/4")

    # Show stress factors
    active_stress = [k for k, v in profile["stress_factors"].items() if v]
    if active_stress:
        print(f"   Stress factors: {', '.join(active_stress)}")

    # Show recommendation
    rec = result["recommendation"]["recommended_model"]
    print(f"   Recommended: {rec['model_name']} ({rec['confidence_score']:.0%} confidence)")
    print(f"   Type: {rec['model_info']['type']}")
    print(f"   Alternatives: {', '.join([m['model_name'] for m in result['recommendation']['alternative_models']])}")
    print()

def main():
    print("=" * 80)
    print("Synthony API - Simple Live Tests")
    print("=" * 80)
    print()

    try:
        test_health()
        test_list_models()
        test_get_model()
        test_analyze_and_recommend()

        print("=" * 80)
        print("All tests passed!")
        print("=" * 80)
        print(f"\nFull API docs: http://localhost:8000/docs")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
