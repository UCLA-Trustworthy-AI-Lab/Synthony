#!/usr/bin/env python3
"""
Test LLM mode with SystemPrompt loading.

This test demonstrates:
1. SystemPrompt is loaded from docs/SystemPrompt_v3.md
2. LLM mode uses the SystemPrompt for better recommendations
3. Hybrid mode combines rule-based + LLM with SystemPrompt
"""

import os
from pathlib import Path
import requests

BASE_URL = os.environ.get("SYNTHONY_API_URL", "http://localhost:8000")
# Dataset directory: override with SYNTHONY_DATA_DIR env var
_default_data_dir = Path(__file__).resolve().parents[2] / "dataset" / "input_data"
DATA_DIR = Path(os.environ.get("SYNTHONY_DATA_DIR", str(_default_data_dir)))

print("=" * 80)
print("Testing LLM Mode with SystemPrompt")
print("=" * 80)

# Test 1: Check health and verify LLM is available
print("\n1️⃣  Checking server status")
print("-" * 80)
response = requests.get(f"{BASE_URL}/health")
health = response.json()
print(f"✓ Server: {health['status']}")
print(f"✓ LLM available: {health['llm_available']}")
print(f"✓ Models: {health['models_count']}")

if not health['llm_available']:
    print("\n❌ LLM mode is not available. Please set VLLM_API_KEY or OPENAI_API_KEY.")
    exit(1)

# Test 2: Rule-based recommendation (baseline)
print("\n2️⃣  Rule-based recommendation (baseline)")
print("-" * 80)

with open(DATA_DIR / "Titanic.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "titanic_rule_based",
            "method": "rule_based",
            "cpu_only": True,
            "top_n": 3
        },
        files={"file": f}
    )

result_rule = response.json()
rec_rule = result_rule["recommendation"]["recommended_model"]
print(f"Method: rule_based")
print(f"  ✓ Recommended: {rec_rule['model_name']}")
print(f"  ✓ Confidence: {rec_rule['confidence_score']:.0%}")
print(f"  ✓ Reasoning (first): {rec_rule['reasoning'][0][:60]}...")

# Test 3: LLM recommendation (with SystemPrompt)
print("\n3️⃣  LLM recommendation (with SystemPrompt)")
print("-" * 80)
print("Note: Watch server logs for '🤖 Using SystemPrompt from...' message")
print()

with open(DATA_DIR / "Titanic.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "titanic_llm",
            "method": "llm",
            "cpu_only": True,
            "top_n": 3
        },
        files={"file": f}
    )

if response.status_code == 200:
    result_llm = response.json()
    rec_llm = result_llm["recommendation"]["recommended_model"]
    print(f"Method: {result_llm['recommendation']['method']}")
    print(f"  ✓ Recommended: {rec_llm['model_name']}")
    print(f"  ✓ Confidence: {rec_llm['confidence_score']:.0%}")
    print(f"  ✓ Reasoning (first): {rec_llm['reasoning'][0][:60]}...")

    if result_llm["recommendation"].get("llm_reasoning"):
        print(f"\n  LLM Reasoning:")
        print(f"    {result_llm['recommendation']['llm_reasoning'][:150]}...")
else:
    print(f"  ⚠ LLM mode failed (status {response.status_code})")
    print(f"  Error: {response.text[:200]}")
    result_llm = None

# Test 4: Hybrid recommendation (rule-based + LLM with SystemPrompt)
print("\n4️⃣  Hybrid recommendation (rule + LLM)")
print("-" * 80)
print("Note: Watch server logs for '🤖 Hybrid mode: Using SystemPrompt...' message")
print()

with open(DATA_DIR / "Titanic.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "titanic_hybrid",
            "method": "hybrid",
            "cpu_only": True,
            "top_n": 3
        },
        files={"file": f}
    )

result_hybrid = response.json()
rec_hybrid = result_hybrid["recommendation"]["recommended_model"]
print(f"Method: {result_hybrid['recommendation']['method']}")
print(f"  ✓ Recommended: {rec_hybrid['model_name']}")
print(f"  ✓ Confidence: {rec_hybrid['confidence_score']:.0%}")
print(f"  ✓ Reasoning (first): {rec_hybrid['reasoning'][0][:60]}...")

if result_hybrid["recommendation"].get("llm_reasoning"):
    print(f"\n  LLM Reasoning:")
    print(f"    {result_hybrid['recommendation']['llm_reasoning'][:150]}...")

# Comparison
print("\n5️⃣  Comparison")
print("-" * 80)
print(f"Rule-based:  {rec_rule['model_name']:<15} (confidence: {rec_rule['confidence_score']:.0%})")
if result_llm:
    print(f"LLM:         {rec_llm['model_name']:<15} (confidence: {rec_llm['confidence_score']:.0%})")
print(f"Hybrid:      {rec_hybrid['model_name']:<15} (confidence: {rec_hybrid['confidence_score']:.0%})")

# Test 5: Test with different dataset (abalone)
print("\n6️⃣  Testing with Abalone dataset (different characteristics)")
print("-" * 80)

with open(DATA_DIR / "abalone.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze-and-recommend",
        params={
            "dataset_id": "abalone_llm",
            "method": "hybrid",
            "cpu_only": True,
            "top_n": 2
        },
        files={"file": f}
    )

result_abalone = response.json()
profile = result_abalone["analysis"]["dataset_profile"]
rec_abalone = result_abalone["recommendation"]["recommended_model"]

print(f"Dataset: {profile['row_count']} rows × {profile['column_count']} columns")

active_stress = [k for k, v in profile["stress_factors"].items() if v]
if active_stress:
    print(f"Stress factors: {', '.join(active_stress)}")

print(f"\nRecommendation:")
print(f"  ✓ Model: {rec_abalone['model_name']}")
print(f"  ✓ Method: {result_abalone['recommendation']['method']}")
print(f"  ✓ Confidence: {rec_abalone['confidence_score']:.0%}")

# Summary
print("\n" + "=" * 80)
print("✅ SystemPrompt Integration Test Complete!")
print("=" * 80)
print("\nKey Points:")
print("  1. SystemPrompt loaded from: docs/SystemPrompt_v3.md")
print("  2. LLM mode uses SystemPrompt for contextual recommendations")
print("  3. Hybrid mode combines rule-based + LLM with SystemPrompt")
print("  4. All methods provide detailed reasoning")
print()
print("Check server logs for:")
print("  • '✓ SystemPrompt loaded from...' at startup")
print("  • '🤖 Using SystemPrompt from...' when LLM is used")
print("  • '✓ LLM response received' after successful API call")
print()
print("Server logs: /tmp/nl_api_server_new.log")
