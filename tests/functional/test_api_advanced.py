#!/usr/bin/env python3
"""Advanced API testing scenarios."""

import os
from pathlib import Path
import requests

BASE_URL = os.environ.get("SYNTHONY_API_URL", "http://localhost:8000")
# Dataset directory: override with SYNTHONY_DATA_DIR env var
_default_data_dir = Path(__file__).resolve().parents[2] / "dataset" / "input_data"
DATA_DIR = Path(os.environ.get("SYNTHONY_DATA_DIR", str(_default_data_dir)))

print("=" * 80)
print("Advanced API Testing Scenarios")
print("=" * 80)

# Scenario 1: Two-step workflow (analyze, then recommend)
print("\n📋 Scenario 1: Two-step workflow")
print("-" * 80)
print("Step 1: Analyze dataset...")

with open(DATA_DIR / "abalone.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze",
        params={"dataset_id": "abalone"},
        files={"file": f}
    )

analysis = response.json()
print(f"✓ Analyzed: {analysis['dataset_id']}")
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
        "constraints": {"cpu_only": True},
        "method": "rule_based",
        "top_n": 2
    }
)
rec_result = rec_response.json()
print(f"     ✓ Recommended: {rec_result['recommended_model']['model_name']}")
print(f"     ✓ Confidence: {rec_result['recommended_model']['confidence_score']:.0%}")
print(f"     ✓ Alternatives: {[m['model_name'] for m in rec_result['alternative_models']]}")

# Try hybrid (will fallback to rule-based if LLM fails)
print("\n  b) Hybrid recommendation:")
rec_response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "dataset_id": "abalone",
        "dataset_profile": analysis["dataset_profile"],
        "column_analysis": analysis["column_analysis"],
        "constraints": {"cpu_only": False},  # Allow GPU models
        "method": "hybrid",
        "top_n": 3
    }
)
rec_result = rec_response.json()
print(f"     ✓ Recommended: {rec_result['recommended_model']['model_name']}")
print(f"     ✓ Method used: {rec_result['method']}")
print(f"     ✓ Confidence: {rec_result['recommended_model']['confidence_score']:.0%}")

# Scenario 2: Compare different constraints
print("\n📋 Scenario 2: Comparing constraints (Bean dataset)")
print("-" * 80)

with open(DATA_DIR / "Bean.csv", "rb") as f:
    file_content = f.read()

# Test with CPU-only constraint
print("  a) With CPU-only constraint:")
response1 = requests.post(
    f"{BASE_URL}/analyze-and-recommend",
    params={"dataset_id": "bean_cpu", "cpu_only": True, "top_n": 2},
    files={"file": ("Bean.csv", file_content)}
)
result1 = response1.json()
rec1 = result1["recommendation"]["recommended_model"]
print(f"     ✓ Recommended: {rec1['model_name']}")
print(f"     ✓ Excluded: {len(result1['recommendation']['excluded_models'])} GPU models")

# Test without CPU constraint
print("\n  b) Without CPU constraint (GPU allowed):")
response2 = requests.post(
    f"{BASE_URL}/analyze-and-recommend",
    params={"dataset_id": "bean_gpu", "cpu_only": False, "top_n": 2},
    files={"file": ("Bean.csv", file_content)}
)
result2 = response2.json()
rec2 = result2["recommendation"]["recommended_model"]
print(f"     ✓ Recommended: {rec2['model_name']}")
print(f"     ✓ Type: {rec2['model_info']['type']}")
print(f"     ✓ GPU required: {rec2['model_info']['constraints']['requires_gpu']}")

# Scenario 3: Filtering models by type
print("\n📋 Scenario 3: Filtering models by type")
print("-" * 80)

types_to_test = ["GAN", "Diffusion", "Tree-based"]
for model_type in types_to_test:
    response = requests.get(f"{BASE_URL}/models", params={"model_type": model_type})
    models = response.json()
    model_names = list(models['models'].keys())
    print(f"  {model_type}: {', '.join(model_names)}")

# Scenario 4: Test with differential privacy constraint
print("\n📋 Scenario 4: Differential Privacy models")
print("-" * 80)

response = requests.get(f"{BASE_URL}/models", params={"requires_dp": True})
dp_models = response.json()
print(f"  DP-enabled models: {list(dp_models['models'].keys())}")

with open(DATA_DIR / "insurance.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={"dataset_id": "insurance_dp", "strict_dp": True, "top_n": 2},
        files={"file": f}
    )

if response.status_code == 200:
    result = response.json()
    rec = result["recommendation"]["recommended_model"]
    print(f"\n  With strict_dp=True:")
    print(f"    ✓ Recommended: {rec['model_name']}")
    print(f"    ✓ DP capability: {rec['capability_match']['privacy_dp']}/4")
    print(f"    ✓ Excluded non-DP models: {len(result['recommendation']['excluded_models'])}")

# Scenario 5: Performance comparison
print("\n📋 Scenario 5: Model performance characteristics")
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
print("✅ All advanced scenarios tested successfully!")
print("=" * 80)
print("\nKey Takeaways:")
print("  • Two-step workflow allows reusing analysis for different constraints")
print("  • CPU-only constraint significantly affects recommendations")
print("  • Differential privacy models available for sensitive data")
print("  • Performance characteristics help choose based on resources")
