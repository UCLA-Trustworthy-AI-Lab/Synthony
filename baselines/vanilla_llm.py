"""Baseline 2: Vanilla LLM Selector (GPT-4o-mini zero-shot).

Provides dataset summary stats to GPT-4o-mini and asks it to rank the 6 models.
No Synthony stress profiles, no capability scores, no feedback loop.

Requires OPENAI_API_KEY environment variable.

Usage:
    OPENAI_API_KEY=sk-... python baselines/vanilla_llm.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from baselines.common import (
    FOCUS_NAMES,
    OVERLAP_MODELS,
    evaluate_baseline,
    find_csv,
    load_dataset_summary,
    load_ground_truth,
)

# Configurable model name
LLM_MODEL = os.getenv("BASELINE_LLM_MODEL", "gpt-4o-mini")
MAX_RETRIES = 2


def _build_prompt(summary: Dict, focus: str) -> str:
    """Build the zero-shot ranking prompt."""
    focus_descriptions = {
        "privacy": "minimize re-identification risk (lower Proportion Closer to Real is better)",
        "fidelity": "maximize statistical similarity (higher Column Shape Score is better)",
        "latency": "minimize training + generation time (lower seconds is better)",
    }

    summary_str = json.dumps(summary, indent=2)

    return f"""You are a synthetic data expert. Given the following dataset summary and
the user's focus ({focus}), rank these 6 tabular data synthesizers from
best to worst: AIM, AutoDiff, DPCART, TabDDPM, TVAE, ARF.

Dataset summary:
{summary_str}

Focus: {focus}
- {focus}: {focus_descriptions[focus]}

Brief model descriptions:
- AIM: Adaptive Iterative Mechanism (differential privacy, fast, statistical queries)
- AutoDiff: Autoregressive diffusion model (high quality, slow, GPU needed)
- DPCART: Differentially Private CART (very fast, tree-based, DP guarantee)
- TabDDPM: Tabular Denoising Diffusion (diffusion-based, good quality, medium speed)
- TVAE: Triplet-loss VAE (fast VAE, decent quality, handles mixed types)
- ARF: Adversarial Random Forests (tree ensemble, fast, good for small data)

Return ONLY a JSON array of model names ranked best to worst.
Example: ["ARF", "TabDDPM", "TVAE", "AIM", "DPCART", "AutoDiff"]"""


def _parse_response(text: str) -> Optional[List[str]]:
    """Parse LLM response into a ranked list of 6 models."""
    # Try to find a JSON array in the response
    # Handle markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip("`").strip()

    try:
        result = json.loads(text)
        if isinstance(result, list) and len(result) == 6:
            # Validate all models present
            if set(result) == set(OVERLAP_MODELS):
                return result
    except json.JSONDecodeError:
        pass

    # Try to extract array from larger text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and set(result) == set(OVERLAP_MODELS):
                return result
        except json.JSONDecodeError:
            pass

    return None


def predict(csv_path: Path, focus: str, client=None) -> List[str]:
    """Query GPT-4o-mini to rank 6 models. Returns ranked list.

    Falls back to alphabetical order if API fails.
    """
    if client is None:
        return sorted(OVERLAP_MODELS)  # fallback

    summary = load_dataset_summary(csv_path)
    prompt = _build_prompt(summary, focus)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = response.choices[0].message.content
            result = _parse_response(text)
            if result is not None:
                return result
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  WARNING: API call failed after {MAX_RETRIES + 1} attempts: {e}")

    # Fallback: alphabetical
    return sorted(OVERLAP_MODELS)


def run_all(gt: Dict, split: Dict) -> Dict[str, List[str]]:
    """Run LLM baseline on all 21 dataset-focus pairs."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set. "
              "Skipping Vanilla LLM baseline.")
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("WARNING: openai package not installed. Run: pip install openai")
        return {}

    print(f"Using model: {LLM_MODEL}")
    predictions = {}
    for key, entry in sorted(gt.items()):
        csv_path = find_csv(entry["dataset"])
        print(f"  Querying {LLM_MODEL} for {key}...", end=" ", flush=True)
        ranking = predict(csv_path, entry["focus"], client=client)
        predictions[key] = ranking
        print(f"-> {ranking[0]}")

    return predictions


if __name__ == "__main__":
    gt, split = load_ground_truth()
    predictions = run_all(gt, split)

    if not predictions:
        print("No predictions (API key missing or openai not installed).")
    else:
        print(f"\nVanilla LLM ({LLM_MODEL}) Rankings:")
        print(f"{'Key':<35} {'Predicted #1':<12} {'GT Best':<12} {'Match'}")
        print("-" * 65)
        for key in sorted(predictions):
            pred = predictions[key]
            gt_best = gt[key]["best_model"]
            match = "Y" if pred[0] == gt_best else ""
            print(f"{key:<35} {pred[0]:<12} {gt_best:<12} {match}")

        results = evaluate_baseline(f"Vanilla LLM ({LLM_MODEL})", predictions, gt, split)
        print(f"\nTrain: Top-1={results['train']['top1']:.3f}  "
              f"Top-3={results['train']['top3']:.3f}  "
              f"Spearman={results['train']['spearman']:.3f}  "
              f"NDCG={results['train']['ndcg']:.3f}")
        print(f"Test:  Top-1={results['test']['top1']:.3f}  "
              f"Top-3={results['test']['top3']:.3f}  "
              f"Spearman={results['test']['spearman']:.3f}  "
              f"NDCG={results['test']['ndcg']:.3f}")
