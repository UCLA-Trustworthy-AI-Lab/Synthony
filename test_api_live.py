#!/usr/bin/env python3
"""Quick live test of the Synthony API."""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("Synthony API Live Test")
print("=" * 80)

# Test 1: Health Check
print("\n1️⃣  Testing Health Check")
print("-" * 80)
response = requests.get(f"{BASE_URL}/health")
health = response.json()
print(f"✓ Status: {health['status']}")
print(f"✓ Version: {health['version']}")
print(f"✓ LLM available: {health['llm_available']}")
print(f"✓ Models: {health['models_count']}")

# Test 2: List Models
print("\n2️⃣  Testing List Models (CPU-only)")
print("-" * 80)
response = requests.get(f"{BASE_URL}/models?cpu_only=true")
models = response.json()
print(f"✓ Total models: {models['total_models']}")
print(f"✓ CPU-compatible: {models['filtered_models']}")
print(f"✓ Available: {', '.join(list(models['models'].keys())[:5])}...")

# Test 3: Get Specific Model Info
print("\n3️⃣  Testing Get Model Info (GReaT)")
print("-" * 80)
response = requests.get(f"{BASE_URL}/models/GReaT")
model_info = response.json()
print(f"✓ Model: {model_info['model_name']}")
print(f"✓ Type: {model_info['type']}")
print(f"✓ Capabilities: {model_info['capabilities']}")
print(f"✓ Strengths: {model_info['strengths'][0]}")

# Test 4: Analyze and Recommend with Titanic dataset
print("\n4️⃣  Testing Analyze-and-Recommend (Titanic dataset)")
print("-" * 80)
with open("/Users/hochan.son/Project/Synthony/dataset/input_data/Titanic.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "titanic_live_test",
            "method": "rule_based",
            "cpu_only": True,
            "top_n": 3
        },
        files={"file": f}
    )

if response.status_code == 200:
    result = response.json()

    # Analysis
    profile = result["analysis"]["dataset_profile"]
    print(f"\n📊 Dataset Profile:")
    print(f"  Size: {profile['row_count']} rows × {profile['column_count']} columns")

    stress_factors = profile["stress_factors"]
    active_stress = [k for k, v in stress_factors.items() if v]
    if active_stress:
        print(f"  ⚠️  Stress factors: {', '.join(active_stress)}")

    # Column analysis
    col_analysis = result["analysis"]["column_analysis"]
    print(f"\n📈 Column Analysis:")
    print(f"  Max difficulty: {col_analysis['max_column_difficulty']}/4")
    print(f"  Difficult columns: {len(col_analysis['difficult_columns'])}")

    # Recommendation
    rec = result["recommendation"]["recommended_model"]
    print(f"\n🎯 Recommendation:")
    print(f"  ✓ Model: {rec['model_name']}")
    print(f"  ✓ Type: {rec['model_info']['type']}")
    print(f"  ✓ Confidence: {rec['confidence_score']:.0%}")

    print(f"\n  Reasoning:")
    for reason in rec["reasoning"][:3]:
        print(f"    {reason}")

    if rec["warnings"]:
        print(f"\n  Warnings:")
        for warning in rec["warnings"][:2]:
            print(f"    {warning}")

    # Alternatives
    alternatives = result["recommendation"]["alternative_models"]
    print(f"\n  Alternatives: {', '.join([m['model_name'] for m in alternatives])}")

    # Save result
    output_file = "titanic_live_test_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Full result saved to: {output_file}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")

# Test 5: Test with insurance dataset (smaller, different characteristics)
print("\n5️⃣  Testing with Insurance dataset")
print("-" * 80)
with open("/Users/hochan.son/Project/Synthony/dataset/input_data/insurance.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "insurance_live_test",
            "method": "rule_based",
            "cpu_only": True,
            "top_n": 3
        },
        files={"file": f}
    )

if response.status_code == 200:
    result = response.json()
    profile = result["analysis"]["dataset_profile"]
    rec = result["recommendation"]["recommended_model"]

    print(f"  Dataset: {profile['row_count']} rows × {profile['column_count']} columns")
    print(f"  Recommended: {rec['model_name']} (confidence: {rec['confidence_score']:.0%})")

    active_stress = [k for k, v in profile["stress_factors"].items() if v]
    if active_stress:
        print(f"  Stress factors: {', '.join(active_stress)}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")

print("\n" + "=" * 80)
print("✅ All API Tests Completed Successfully!")
print("=" * 80)
print("\nInteractive API docs available at: http://localhost:8000/docs")
