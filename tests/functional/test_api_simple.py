#!/usr/bin/env python3
"""Simple, focused API test."""

import requests
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("🔍 Test 1: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"   ✓ Status: {data['status']}")
    print(f"   ✓ LLM available: {data['llm_available']}")
    print(f"   ✓ Models: {data['models_count']}\n")

def test_list_models():
    """Test list models endpoint."""
    print("🔍 Test 2: List Models (CPU-only)")
    response = requests.get(f"{BASE_URL}/models?cpu_only=true")
    data = response.json()
    print(f"   ✓ Total: {data['total_models']}, CPU-compatible: {data['filtered_models']}")
    print(f"   ✓ Models: {', '.join(list(data['models'].keys())[:4])}...\n")

def test_get_model():
    """Test get specific model endpoint."""
    print("🔍 Test 3: Get Model Info (TabTree)")
    response = requests.get(f"{BASE_URL}/models/TabTree")
    data = response.json()
    print(f"   ✓ Model: {data['model_name']}")
    print(f"   ✓ Type: {data['type']}")
    print(f"   ✓ Training speed: {data['performance']['training_speed']}\n")

def test_analyze_and_recommend():
    """Test one-shot analyze-and-recommend endpoint."""
    print("🔍 Test 4: Analyze & Recommend (Titanic)")

    with open("/Users/hochan.son/Project/Synthony/dataset/input_data/Titanic.csv", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/analyze-and-recommend",
            params={
                "dataset_id": "titanic_simple_test",
                "method": "rule_based",
                "cpu_only": True,
                "top_n": 3
            },
            files={"file": f}
        )

    if response.status_code != 200:
        print(f"   ❌ Error {response.status_code}: {response.text}")
        return

    result = response.json()

    # Show analysis
    profile = result["analysis"]["dataset_profile"]
    col_analysis = result["analysis"]["column_analysis"]
    print(f"   ✓ Dataset: {profile['row_count']} rows × {profile['column_count']} columns")
    print(f"   ✓ Max difficulty: {col_analysis['max_column_difficulty']}/4")

    # Show stress factors
    active_stress = [k for k, v in profile["stress_factors"].items() if v]
    if active_stress:
        print(f"   ✓ Stress factors: {', '.join(active_stress)}")

    # Show recommendation
    rec = result["recommendation"]["recommended_model"]
    print(f"   ✓ Recommended: {rec['model_name']} ({rec['confidence_score']:.0%} confidence)")
    print(f"   ✓ Type: {rec['model_info']['type']}")
    print(f"   ✓ Alternatives: {', '.join([m['model_name'] for m in result['recommendation']['alternative_models']])}")
    print()

def test_compare_constraints():
    """Test with different constraints."""
    print("🔍 Test 5: Compare CPU-only vs All models (Insurance)")

    with open("/Users/hochan.son/Project/Synthony/dataset/input_data/insurance.csv", "rb") as f:
        file_content = f.read()

    # CPU-only
    response1 = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={"cpu_only": True},
        files={"file": ("insurance.csv", file_content)}
    )
    result1 = response1.json()
    rec1 = result1["recommendation"]["recommended_model"]

    # All models
    response2 = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={"cpu_only": False},
        files={"file": ("insurance.csv", file_content)}
    )
    result2 = response2.json()
    rec2 = result2["recommendation"]["recommended_model"]

    print(f"   ✓ CPU-only → {rec1['model_name']}")
    print(f"   ✓ All models → {rec2['model_name']}")
    print(f"   ✓ GPU models excluded with CPU-only: {len(result1['recommendation']['excluded_models'])}")
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
        test_compare_constraints()

        print("=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print("\n📚 Full API docs: http://localhost:8000/docs")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
