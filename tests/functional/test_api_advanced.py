#!/usr/bin/env python3
"""Advanced API testing scenarios."""


import requests

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("Advanced API Testing Scenarios")
print("=" * 80)

# Scenario 1: Two-step workflow (analyze, then recommend)
print("\nScenario 1: Two-step workflow")
print("-" * 80)
print("Step 1: Analyze dataset...")

with open("/Users/hochan.son/Project/Synthony/dataset/input_data/abalone.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze",
        params={"dataset_id": "abalone"},
        files={"file": f}
    )

analysis = response.json()
print(f"Analyzed: {analysis['dataset_id']}")
print(f"  Rows: {analysis['dataset_profile']['row_count']}")
print(f"  Max difficulty: {analysis['column_analysis']['max_column_difficulty']}/4")

print("\nStep 2: Get recommendations with different methods...")

# Try rule-based
print("\n  a) Rule-based recommendation:")
rec_response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "dataset_id": "abalone",
        "dataset_profile": analysis["dataset_profile"],
        "column_analysis": analysis["column_analysis"],
        "method": "rule_based",
        "top_n": 2
    }
)
rec_result = rec_response.json()
print(f"     Recommended: {rec_result['recommended_model']['model_name']}")
print(f"     Confidence: {rec_result['recommended_model']['confidence_score']:.0%}")
print(f"     Alternatives: {[m['model_name'] for m in rec_result['alternative_models']]}")

# Try hybrid (will fallback to rule-based if LLM fails)
print("\n  b) Hybrid recommendation:")
rec_response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "dataset_id": "abalone",
        "dataset_profile": analysis["dataset_profile"],
        "column_analysis": analysis["column_analysis"],
        "method": "hybrid",
        "top_n": 3
    }
)
rec_result = rec_response.json()
print(f"     Recommended: {rec_result['recommended_model']['model_name']}")
print(f"     Method used: {rec_result['method']}")
print(f"     Confidence: {rec_result['recommended_model']['confidence_score']:.0%}")

# Scenario 2: Filtering models by type
print("\nScenario 2: Filtering models by type")
print("-" * 80)

types_to_test = ["GAN", "Diffusion", "Tree-based"]
for model_type in types_to_test:
    response = requests.get(f"{BASE_URL}/models", params={"model_type": model_type})
    models = response.json()
    model_names = [m['name'] for m in models['models']]
    print(f"  {model_type}: {', '.join(model_names)}")

# Scenario 3: Performance comparison
print("\nScenario 3: Model performance characteristics")
print("-" * 80)

performance_models = ["GReaT", "TabDDPM", "TabTree", "GaussianCopula"]
print(f"\n{'Model':<20} {'Training':<15} {'Inference':<15} {'Memory':<15}")
print("-" * 70)

for model_name in performance_models:
    response = requests.get(f"{BASE_URL}/models/{model_name}")
    if response.status_code == 200:
        model = response.json()
        perf = model['performance']
        print(f"{model_name:<20} {perf['training_speed']:<15} {perf['inference_speed']:<15} {perf['memory_usage']:<15}")

print("\n" + "=" * 80)
print("All advanced scenarios tested successfully!")
print("=" * 80)
print("\nKey Takeaways:")
print("  - Two-step workflow allows reusing analysis for different methods")
print("  - Performance characteristics help choose based on resources")
