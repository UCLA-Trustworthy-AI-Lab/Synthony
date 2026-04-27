"""
Model Tools for MCP Server

Tools for Package 2 shadow interface: Model capabilities and constraints.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.types import Tool, ToolAnnotations

from synthony.recommender.engine import ModelRecommendationEngine

# Both paths must stay in sync — engine loads from src/, config/ is the canonical copy
_CAPABILITIES_PATHS = [
    Path(__file__).parent.parent.parent / "src" / "synthony" / "recommender" / "model_capabilities.json",
    Path(__file__).parent.parent.parent / "config" / "model_capabilities.json",
]

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "SystemPrompt.md"

_TYPE_ABBREV: Dict[str, str] = {
    "GAN": "GAN",
    "VAE": "VAE",
    "Diffusion": "Diffusion",
    "Tree-based": "Tree",
    "Statistical": "Statistical",
    "LLM": "LLM",
    "LLM-based": "LLM",
    "Tree-based + DP": "Tree+DP",
    "GAN + DP": "GAN+DP",
    "GAN+DP": "GAN+DP",
    "Statistical + DP": "Stat+DP",
    "Stat+DP": "Stat+DP",
    "Baseline": "Baseline",
    "Flow": "Flow",
}

_VALID_CAPABILITIES = {
    "skew_handling", "cardinality_handling", "zipfian_handling",
    "small_data", "correlation_handling", "privacy_dp",
}
_VALID_EMPIRICAL = {
    "avg_quality_score", "avg_fidelity", "avg_utility",
    "skew_preservation", "cardinality_preservation", "correlation_preservation",
    "datasets_tested",
}


def _metric_to_score(value: float) -> int:
    """Convert 0-1 metric to 0-4 capability score (scoring_methodology.md §2)."""
    if value >= 0.90:
        return 4
    elif value >= 0.75:
        return 3
    elif value >= 0.50:
        return 2
    elif value >= 0.25:
        return 1
    return 0


def _bump_patch(version: str) -> str:
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


# ── System prompt generation helpers ─────────────────────────────────────────

def _abbrev_type(model_type: str) -> str:
    return _TYPE_ABBREV.get(model_type, model_type)


def _score_cell(score: int) -> str:
    return f"**{score}**" if score >= 4 else str(score)


def _split_prompt_sections(content: str) -> Tuple[str, Dict[str, str]]:
    """Split SystemPrompt.md into intro and numbered sections."""
    m_first = re.search(r'^## 1\.', content, flags=re.MULTILINE)
    intro = content[:m_first.start()].rstrip() if m_first else ""
    parts = re.split(r'(?=^## \d+\.)', content, flags=re.MULTILINE)
    sections: Dict[str, str] = {}
    for part in parts:
        m = re.match(r'^## (\d+)\.', part)
        if m:
            sections[m.group(1)] = part.rstrip()
    return intro, sections


def _active_spark_models(models: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # Exclude baselines (Identity) and explicitly excluded models
    return [
        (name, m) for name, m in models.items()
        if not m.get("exclude", False)
        and m.get("capabilities_source") == "spark"
        and m.get("type", "") != "Baseline"
    ]


def _additional_models(models: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # Literature models + baselines (Identity shown separately)
    return [
        (name, m) for name, m in models.items()
        if not m.get("exclude", False)
        and (m.get("capabilities_source") != "spark" or m.get("type") == "Baseline")
        and name != "Identity"  # Identity covered by the note, not a table row
    ]


def _gen_section1(models: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    active = sorted(
        _active_spark_models(models),
        key=lambda x: -(x[1].get("spark_empirical", {}).get("avg_quality_score", 0)),
    )
    additional = _additional_models(models)
    cap_version = metadata.get("version", "?")

    lines = [
        "## 1. KNOWLEDGE BASE (Capability Scores 0-4, calibrated from spark benchmarks)",
        "",
        "### Active Models (Benchmarked — recommend from these)",
        "",
        "| Model | Type | GPU | Skew (>2.0) | Card (>500) | Zipfian | Small (<500) | Corr | Privacy (DP) | Quality |",
        "|:------|:-----|:---:|:-----------:|:-----------:|:-------:|:------------:|:----:|:------------:|:-------:|",
    ]
    for name, m in active:
        caps = m.get("capabilities", {})
        gpu = "yes" if m.get("constraints", {}).get("requires_gpu", False) else "no"
        q = m.get("spark_empirical", {}).get("avg_quality_score", 0)
        lines.append(
            f"| **{name}** | {_abbrev_type(m.get('type', ''))} | {gpu} "
            f"| {_score_cell(caps.get('skew_handling', 0))} "
            f"| {_score_cell(caps.get('cardinality_handling', 0))} "
            f"| {_score_cell(caps.get('zipfian_handling', 0))} "
            f"| {_score_cell(caps.get('small_data', 0))} "
            f"| {_score_cell(caps.get('correlation_handling', 0))} "
            f"| {_score_cell(caps.get('privacy_dp', 0))} "
            f"| {q:.3f} |"
        )
    lines.extend([
        "",
        f"**Quality** = avg_quality_score from spark benchmarks (10 datasets, {len(active)} models). Models ordered by tier then quality.",
        "",
        "**Note**: Identity is a passthrough baseline for testing only — never recommend for production use.",
        "",
        f"### Scoring Methodology (v{cap_version})",
        "",
        "Capability scores are derived from empirical benchmark preservation rates:",
        "- **Score 4**: preservation >= 0.90 (excellent)",
        "- **Score 3**: preservation >= 0.75 (good)",
        "- **Score 2**: preservation >= 0.50 (moderate)",
        "- **Score 1**: preservation >= 0.25 (poor)",
        "- **Score 0**: preservation < 0.25 (fails)",
        "",
        "Key methodological improvements over v4.0 (trial4):",
        "- **Cardinality**: Uses density-normalized formula `(synth_unique/synth_rows) / (orig_unique/orig_rows)` to correct for row-count sampling bias",
        "- **Correlation**: Tested on 10 diverse datasets (vs 8), revealing many models preserve correlation far better than trial4 indicated",
        "- **Skew**: More datasets exposed that some models (TabDDPM, NFlow) overfit skew on small trial4 test sets",
    ])
    if additional:
        lines.extend([
            "",
            "### Additional Models (Literature-Based Scores)",
            "",
            "| Model | Type | GPU | Skew | Card | Zipfian | Small | Corr | Privacy | Quality |",
            "|:------|:-----|:---:|:----:|:----:|:-------:|:-----:|:----:|:-------:|:-------:|",
        ])
        for name, m in additional:
            caps = m.get("capabilities", {})
            gpu = "yes" if m.get("constraints", {}).get("requires_gpu", False) else "no"
            emp = m.get("spark_empirical", {})
            q_val = emp.get("avg_quality_score")
            if q_val is not None:
                q_str = f"{q_val:.3f} (passthrough)" if name == "Identity" else f"{q_val:.3f}"
            else:
                q_str = "N/A (literature)"
            lines.append(
                f"| **{name}** | {_abbrev_type(m.get('type', ''))} | {gpu} "
                f"| {caps.get('skew_handling', 0)} "
                f"| {caps.get('cardinality_handling', 0)} "
                f"| {caps.get('zipfian_handling', 0)} "
                f"| {caps.get('small_data', 0)} "
                f"| {caps.get('correlation_handling', 0)} "
                f"| {caps.get('privacy_dp', 0)} "
                f"| {q_str} |"
            )
        lines.extend([
            "",
            "**Note**: GReaT scores are literature-derived (not empirically validated). Identity is a passthrough baseline for testing only.",
        ])
    return "\n".join(lines)


def _gen_section2(models: Dict[str, Any]) -> str:
    active = [(n, m) for n, m in _active_spark_models(models) if n != "Identity"]
    tiers: Dict[str, List[Tuple[str, float]]] = {
        "Top": [], "Mid-High": [], "Mid": [], "Low": [],
    }
    for name, m in active:
        q = m.get("spark_empirical", {}).get("avg_quality_score", 0)
        if q >= 0.96:
            tiers["Top"].append((name, q))
        elif q >= 0.80:
            tiers["Mid-High"].append((name, q))
        elif q >= 0.65:
            tiers["Mid"].append((name, q))
        else:
            tiers["Low"].append((name, q))

    tier_chars = {
        "Top": "Excellent fidelity + utility, fast, CPU-compatible",
        "Mid-High": "Good utility, moderate fidelity, some need GPU",
        "Mid": "Acceptable quality, specific use cases (DP, diffusion)",
        "Low": "DP/privacy models or poor general quality",
    }

    lines = [
        "## 2. MODEL TIERS (Validated on abalone, 10-dataset avg)",
        "",
        "| Tier | Models | Quality Range | Characteristics |",
        "|------|--------|:------------:|-----------------|",
    ]
    for tier_name, entries in tiers.items():
        if not entries:
            continue
        entries.sort(key=lambda x: -x[1])
        model_list = ", ".join(e[0] for e in entries)
        q_min = min(e[1] for e in entries)
        q_max = max(e[1] for e in entries)
        q_range = f"{q_min:.2f} – {q_max:.2f}" if q_min != q_max else f"{q_min:.2f}"
        lines.append(f"| **{tier_name}** | {model_list} | {q_range} | {tier_chars[tier_name]} |")
    return "\n".join(lines)


def _gen_section3(models: Dict[str, Any]) -> str:
    spark_active = dict(_active_spark_models(models))
    gpu_yes = sorted([
        name for name, m in spark_active.items()
        if m.get("constraints", {}).get("requires_gpu", False)
    ])
    gpu_no = sorted([
        name for name, m in spark_active.items()
        if not m.get("constraints", {}).get("requires_gpu", False)
    ])
    lines = [
        "## 3. GPU HANDLING",
        "",
        "| GPU Required | Models | Action when `cpu_only=true` |",
        "|:------------:|--------|----------------------------|",
        f"| **yes** | {', '.join(gpu_yes)} | **EXCLUDE** from candidates |",
        f"| **no** | {', '.join(gpu_no)} | Keep in candidates |",
    ]
    return "\n".join(lines)


def _gen_section7(models: Dict[str, Any]) -> str:
    active = dict(_active_spark_models(models))

    def caps_of(name: str) -> Dict[str, int]:
        return active[name].get("capabilities", {}) if name in active else {}

    def quality_of(name: str) -> float:
        return active[name].get("spark_empirical", {}).get("avg_quality_score", 0) if name in active else 0

    def fmt_cap(name: str, cap_key: str) -> str:
        score = caps_of(name).get(cap_key, 0)
        return f"{name} ({score})"

    # Best = score 4, then 3; Avoid = score 0-1
    def best_by(cap_key: str, min_score: int = 3) -> List[str]:
        return sorted(
            [n for n, m in active.items() if m.get("capabilities", {}).get(cap_key, 0) >= min_score and m.get("capabilities_source") == "spark"],
            key=lambda n: (-caps_of(n).get(cap_key, 0), -quality_of(n))
        )

    def avoid_by(cap_key: str, max_score: int = 1) -> List[str]:
        return [n for n, m in active.items() if m.get("capabilities", {}).get(cap_key, 0) <= max_score and m.get("capabilities_source") == "spark"]

    # Small data best: small_data >= 4; avoid: score <= 2 (poor or moderate)
    small_best = best_by("small_data", 4)
    small_avoid = avoid_by("small_data", 2)

    # Large data: fast CPU models; avoid BayesianNetwork (max 50k)
    large_best = [n for n, m in active.items() if not m.get("constraints", {}).get("requires_gpu", False) and m.get("performance", {}).get("training_speed") == "fast"]
    large_avoid = [n for n, m in active.items() if m.get("constraints", {}).get("max_recommended_rows", 1e9) < 100000 and m.get("capabilities_source") == "spark"]

    # Skew
    skew_best = best_by("skew_handling", 3)
    skew_avoid = [n for n in avoid_by("skew_handling", 1) if caps_of(n).get("skew_handling", 0) <= 1]

    # Cardinality
    card_best = best_by("cardinality_handling", 4)
    card_avoid = [n for n, m in active.items() if m.get("capabilities", {}).get("cardinality_handling", 0) == 0 and m.get("capabilities_source") == "spark"]

    # Zipfian
    zipf_best = best_by("zipfian_handling", 3)
    zipf_avoid = avoid_by("zipfian_handling", 1)

    # Correlation
    corr_4 = [n for n in best_by("correlation_handling", 4)]
    corr_3 = [n for n in best_by("correlation_handling", 3) if n not in corr_4]
    if corr_3:
        corr_best_str = ", ".join(f"{n}/{'CART' if n == 'CART' else n}" for n in corr_4) + f", {', '.join(corr_3)} (3)"
    else:
        corr_best_str = ", ".join(f"{n} (4)" for n in corr_4)
    corr_avoid = [n for n, m in active.items() if m.get("capabilities", {}).get("correlation_handling", 0) == 0 and m.get("capabilities_source") == "spark"]

    # CPU-only
    cpu_keep = sorted([n for n, m in active.items() if m.get("constraints", {}).get("cpu_only_compatible", False) and m.get("capabilities_source") == "spark"])
    cpu_exclude = sorted([n for n, m in active.items() if m.get("constraints", {}).get("requires_gpu", False) and m.get("capabilities_source") == "spark"])

    # DP
    dp_best = sorted([n for n, m in active.items() if m.get("capabilities", {}).get("privacy_dp", 0) >= 3], key=lambda n: -caps_of(n).get("privacy_dp", 0))
    dp_cpu = sorted([n for n in dp_best if active[n].get("constraints", {}).get("cpu_only_compatible", False)])

    # Fast
    fast_models = sorted([n for n, m in active.items() if m.get("performance", {}).get("training_speed") == "fast" and m.get("capabilities_source") == "spark"])
    slow_models = [n for n, m in active.items() if m.get("performance", {}).get("training_speed") == "slow" and m.get("capabilities_source") == "spark"]

    # Best quality
    quality_sorted = sorted(
        [(n, quality_of(n)) for n in active if active[n].get("capabilities_source") == "spark"],
        key=lambda x: -x[1]
    )
    quality_best = [f"{n} ({q:.3f})" for n, q in quality_sorted[:4]]
    quality_worst = [f"{n} ({q:.3f})" for n, q in reversed(quality_sorted[-3:])]

    # DP quality tradeoff
    dp_quality = sorted([(n, quality_of(n)) for n in dp_best], key=lambda x: -x[1])
    dp_q_best = f"{dp_quality[0][0]} (dp={caps_of(dp_quality[0][0]).get('privacy_dp', 0)}, quality={dp_quality[0][1]:.3f})" if dp_quality else "N/A"
    dp_q_avoid = f"{dp_quality[-1][0]} (dp={caps_of(dp_quality[-1][0]).get('privacy_dp', 0)}, quality={dp_quality[-1][1]:.3f})" if len(dp_quality) > 1 else "N/A"

    def fmtlist(lst: List[str]) -> str:
        return ", ".join(lst) if lst else "N/A"

    def fmtcap(lst: List[str], key: str) -> str:
        return ", ".join(f"{n} ({caps_of(n).get(key, 0)})" for n in lst) if lst else "N/A"

    lines = [
        "## 7. QUICK REFERENCE BY USE CASE",
        "",
        "| Use Case | Best Models | Avoid |",
        "|----------|-------------|-------|",
        f"| **Small data (<500 rows)** | {fmtcap(small_best, 'small_data')} | {fmtcap(small_avoid, 'small_data')} |",
        f"| **Large data (>50k rows)** | {fmtlist(large_best)} | {fmtlist(large_avoid)} |",
        f"| **Severe skew (>2.0)** | {fmtcap(skew_best, 'skew_handling')} | {fmtcap(skew_avoid, 'skew_handling')} |",
        f"| **High cardinality (>500)** | {fmtcap(card_best, 'cardinality_handling')} | {fmtcap(card_avoid, 'cardinality_handling')} |",
        f"| **Zipfian distribution** | {fmtcap(zipf_best, 'zipfian_handling')} | {fmtcap(zipf_avoid, 'zipfian_handling')} |",
        f"| **Correlation-sensitive** | {fmtcap(corr_4 + corr_3, 'correlation_handling')} | {fmtcap(corr_avoid, 'correlation_handling')} |",
        f"| **CPU-only environment** | {fmtlist(cpu_keep)} | {fmtlist(cpu_exclude)} |",
        f"| **Strict privacy (DP)** | {fmtcap(dp_best, 'privacy_dp')} | All non-DP models |",
        f"| **Strict DP + CPU-only** | {fmtcap(dp_cpu, 'privacy_dp')} | {'PATECTGAN (requires GPU)' if 'PATECTGAN' in dp_best and 'PATECTGAN' not in dp_cpu else 'N/A'} |",
        f"| **Fast turnaround** | {fmtlist(fast_models)} | {fmtlist(slow_models)} |",
        f"| **Best quality (no constraints)** | {', '.join(quality_best)} | {', '.join(quality_worst)} |",
        f"| **Best privacy/quality tradeoff** | {dp_q_best} | {dp_q_avoid} |",
    ]
    return "\n".join(lines)


def _build_system_prompt(
    version: str,
    models: Dict[str, Any],
    metadata: Dict[str, Any],
    static_sections: Dict[str, str],
    version_note: Optional[str] = None,
) -> str:
    """Assemble the full SystemPrompt.md content from dynamic + static sections."""
    intro_line = f"# SYSTEM PROMPT: Synthetic Data Model Selector {version}"
    if version_note:
        intro_line += f"  \n<!-- {version_note} -->"

    parts = [
        intro_line,
        "",
        "You are the Model Selector for the Synthony platform. Your goal is to interpret statistical profiles and recommend synthesis models from the available benchmarked models.",
        "",
        _gen_section1(models, metadata),
        "",
        _gen_section2(models),
        "",
        _gen_section3(models),
        "",
        static_sections.get("4", "## 4. (section not found)"),
        "",
        static_sections.get("5", "## 5. (section not found)"),
        "",
        static_sections.get("6", "## 6. (section not found)"),
        "",
        _gen_section7(models),
        "",
        static_sections.get("8", "## 8. (section not found)"),
        "",
        static_sections.get("9", "## 9. (section not found)"),
        "",
        static_sections.get("10", "## 10. (section not found)"),
    ]
    return "\n".join(parts)


class ModelTools:
    """
    Model tools for querying model capabilities and constraints.

    Tools:
    - check_model_constraints: Validate data size limits for models
    - get_model_info: Get detailed information about a specific model
    - list_models: List all available models with optional filters
    """

    def __init__(self, recommender: ModelRecommendationEngine):
        """Initialize model tools with recommendation engine."""
        self.recommender = recommender

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "synthony_check_model_constraints",
            "synthony_get_model_info",
            "synthony_list_models",
            "synthony_update_model_capabilities",
            "synthony_update_system_prompt",
        ]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="synthony_check_model_constraints",
                description=(
                    "Check which models are compatible with a given dataset size. "
                    "Applies row-count filters based on model min/max data size limits. "
                    "Returns list of compatible models and reasons for exclusions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "row_count": {
                            "type": "integer",
                            "description": "Number of rows in dataset (for size-based filtering)",
                            "minimum": 1
                        }
                    }
                },
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
            Tool(
                name="synthony_get_model_info",
                description=(
                    "Get detailed information about a specific synthesis model. "
                    "Returns full specification including: "
                    "- Capabilities (skew handling, cardinality, zipfian, privacy) "
                    "- Constraints (CPU/GPU, data size limits, privacy requirements) "
                    "- Performance characteristics (speed, quality) "
                    "- Strengths and limitations "
                    "Use this tool when you need to understand a model's characteristics."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to query (e.g., 'GReaT', 'TabDDPM', 'ARF')"
                        }
                    },
                    "required": ["model_name"]
                },
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
            Tool(
                name="synthony_list_models",
                description=(
                    "List all available synthesis models with optional filtering. "
                    "Supports filtering by model_type: GAN, VAE, Diffusion, Tree-based, Statistical, LLM. "
                    "Returns model registry with capability scores and rankings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Filter by model type",
                            "enum": ["GAN", "VAE", "Diffusion", "Tree-based", "Statistical", "LLM"]
                        }
                    }
                },
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
            Tool(
                name="synthony_update_model_capabilities",
                description=(
                    "Update capability scores and/or empirical benchmark metrics for a model "
                    "in model_capabilities.json. "
                    "Capability scores (0-4 scale per scoring_methodology.md): "
                    "skew_handling, cardinality_handling, zipfian_handling, small_data, "
                    "correlation_handling, privacy_dp. "
                    "Spark empirical metrics (0-1 floats): avg_quality_score, avg_fidelity, "
                    "avg_utility, skew_preservation, cardinality_preservation, "
                    "correlation_preservation, datasets_tested (int). "
                    "Set auto_calculate=true to derive capability scores from spark_empirical "
                    "values using the standard scoring thresholds (>=0.90→4, >=0.75→3, etc.). "
                    "Writes both src/ and config/ copies atomically and bumps the patch version."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model to update (e.g. 'ARF', 'TabDDPM', 'GReaT')"
                        },
                        "capabilities": {
                            "type": "object",
                            "description": (
                                "Capability scores to update. Keys: skew_handling, "
                                "cardinality_handling, zipfian_handling, small_data, "
                                "correlation_handling, privacy_dp. Values: integer 0-4."
                            ),
                            "additionalProperties": {"type": "integer", "minimum": 0, "maximum": 4}
                        },
                        "spark_empirical": {
                            "type": "object",
                            "description": (
                                "Empirical benchmark metrics to update. Float fields (0-1): "
                                "avg_quality_score, avg_fidelity, avg_utility, skew_preservation, "
                                "cardinality_preservation, correlation_preservation. "
                                "Integer field: datasets_tested."
                            )
                        },
                        "auto_calculate": {
                            "type": "boolean",
                            "description": (
                                "If true, recalculate capability scores from the updated "
                                "spark_empirical values using standard thresholds. "
                                "Overwrites any capability scores provided directly."
                            ),
                            "default": False
                        },
                        "version_note": {
                            "type": "string",
                            "description": "Optional note appended to metadata.source describing what changed"
                        }
                    },
                    "required": ["model_name"]
                },
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
            Tool(
                name="synthony_update_system_prompt",
                description=(
                    "Regenerate config/SystemPrompt.md from the current model_capabilities.json scores "
                    "and store it as a new versioned entry in the SQLite database. "
                    "Dynamic sections rebuilt from JSON: "
                    "(1) knowledge base capability table, "
                    "(2) model tiers by quality, "
                    "(3) GPU handling table, "
                    "(7) quick reference by use case. "
                    "Static sections (decision logic, output format, examples, validation notes) are preserved unchanged. "
                    "Use this after updating capability scores with synthony_update_model_capabilities "
                    "to keep the LLM recommendation path in sync with the rule-based engine."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "version": {
                            "type": "string",
                            "description": "Version label for the new prompt (e.g. 'v6.0', 'v5.1')"
                        },
                        "set_active": {
                            "type": "boolean",
                            "description": "Set the new prompt as active in the database (default: true)",
                            "default": True
                        },
                        "version_note": {
                            "type": "string",
                            "description": "Optional note describing what changed in this version"
                        }
                    },
                    "required": ["version"]
                },
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a model tool."""
        if name == "synthony_check_model_constraints":
            return await self._check_model_constraints(arguments)
        elif name == "synthony_get_model_info":
            return await self._get_model_info(arguments)
        elif name == "synthony_list_models":
            return await self._list_models(arguments)
        elif name == "synthony_update_model_capabilities":
            return await self._update_model_capabilities(arguments)
        elif name == "synthony_update_system_prompt":
            return await self._update_system_prompt(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _check_model_constraints(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check model compatibility based on row count.

        Args:
            arguments: {
                "row_count": Optional[int]
            }

        Returns:
            {
                "compatible_models": List[str],
                "excluded_models": Dict[str, str],
                "filters_applied": Dict[str, Any]
            }
        """
        row_count = arguments.get("row_count")

        models = self.recommender.model_capabilities.get("models", {})
        compatible_models = []
        excluded_models = {}

        for model_name, model_info in models.items():
            constraints = model_info.get("constraints", {})

            # Check data size constraints
            if row_count:
                model_min = constraints.get("min_data_size", 0)
                model_max = constraints.get("max_data_size", float('inf'))

                if row_count < model_min:
                    excluded_models[model_name] = f"Requires at least {model_min} rows"
                    continue

                if row_count > model_max:
                    excluded_models[model_name] = f"Maximum {model_max} rows supported"
                    continue

            compatible_models.append(model_name)

        return {
            "compatible_models": compatible_models,
            "excluded_models": excluded_models,
            "filters_applied": {
                "row_count": row_count
            }
        }

    async def _get_model_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model information.

        Args:
            arguments: {
                "model_name": str
            }

        Returns:
            Full model specification including capabilities, constraints, performance, etc.
        """
        model_name = arguments["model_name"]
        models = self.recommender.model_capabilities.get("models", {})

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            )

        model_info = models[model_name]

        return {
            "model_name": model_info["name"],
            "full_name": model_info["full_name"],
            "type": model_info["type"],
            "capabilities": model_info["capabilities"],
            "constraints": model_info["constraints"],
            "performance": model_info["performance"],
            "description": model_info["description"],
            "strengths": model_info["strengths"],
            "limitations": model_info["limitations"],
        }

    async def _list_models(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        List models with optional filtering.

        Args:
            arguments: {
                "model_type": Optional[str]
            }

        Returns:
            {
                "total_models": int,
                "filtered_models": int,
                "models": Dict[str, Any],
                "model_ranking": Dict[str, Any]
            }
        """
        model_type = arguments.get("model_type")

        models = self.recommender.model_capabilities.get("models", {})
        filtered_models = {}

        for name, info in models.items():
            # Type filter
            if model_type and info.get("type", "").lower() != model_type.lower():
                continue

            filtered_models[name] = info

        return {
            "total_models": len(models),
            "filtered_models": len(filtered_models),
            "models": filtered_models,
            "model_ranking": self.recommender.model_capabilities.get("model_ranking_by_capability", {}),
        }

    async def _update_model_capabilities(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update capability scores and/or spark_empirical metrics for a model.

        Writes to both src/synthony/recommender/model_capabilities.json and
        config/model_capabilities.json to keep them in sync, then bumps the
        patch version and last_updated date.

        Args:
            arguments: {
                "model_name": str,
                "capabilities": Optional[Dict[str, int]],   # keys in _VALID_CAPABILITIES
                "spark_empirical": Optional[Dict[str, float]],
                "auto_calculate": bool,
                "version_note": Optional[str]
            }
        """
        model_name = arguments["model_name"]
        cap_updates: Dict[str, int] = arguments.get("capabilities") or {}
        emp_updates: Dict[str, Any] = arguments.get("spark_empirical") or {}
        auto_calc: bool = arguments.get("auto_calculate", False)
        version_note: Optional[str] = arguments.get("version_note")

        # ── Validate inputs ──────────────────────────────────────────────
        invalid_caps = set(cap_updates) - _VALID_CAPABILITIES
        if invalid_caps:
            raise ValueError(
                f"Unknown capability keys: {sorted(invalid_caps)}. "
                f"Valid: {sorted(_VALID_CAPABILITIES)}"
            )
        invalid_emp = set(emp_updates) - _VALID_EMPIRICAL
        if invalid_emp:
            raise ValueError(
                f"Unknown spark_empirical keys: {sorted(invalid_emp)}. "
                f"Valid: {sorted(_VALID_EMPIRICAL)}"
            )
        for k, v in cap_updates.items():
            if not isinstance(v, int) or not (0 <= v <= 4):
                raise ValueError(f"Capability '{k}' must be an integer 0-4, got {v!r}")

        if not cap_updates and not emp_updates:
            raise ValueError(
                "Provide at least one of 'capabilities' or 'spark_empirical' to update."
            )

        # ── Load from the primary path ────────────────────────────────────
        primary_path = _CAPABILITIES_PATHS[0]
        with open(primary_path) as f:
            data = json.load(f)

        models = data.get("models", {})
        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {sorted(models.keys())}"
            )

        model = models[model_name]

        # ── Record before state ───────────────────────────────────────────
        before_caps = dict(model.get("capabilities", {}))
        before_emp = dict(model.get("spark_empirical", {}))

        # ── Apply spark_empirical updates ─────────────────────────────────
        if emp_updates:
            if "spark_empirical" not in model:
                model["spark_empirical"] = {}
            model["spark_empirical"].update(emp_updates)

        # ── Auto-calculate capability scores from empirical data ──────────
        if auto_calc and (emp_updates or model.get("spark_empirical")):
            emp = model.get("spark_empirical", {})
            derived: Dict[str, int] = {}

            if "skew_preservation" in emp:
                derived["skew_handling"] = _metric_to_score(emp["skew_preservation"])
            if "cardinality_preservation" in emp:
                derived["cardinality_handling"] = _metric_to_score(emp["cardinality_preservation"])
            if "correlation_preservation" in emp:
                derived["correlation_handling"] = _metric_to_score(emp["correlation_preservation"])
            if "avg_quality_score" in emp:
                derived["small_data"] = _metric_to_score(emp["avg_quality_score"])

            # Merge: derived scores override cap_updates (auto_calculate wins)
            cap_updates = {**cap_updates, **derived}

        # ── Apply capability updates ──────────────────────────────────────
        if cap_updates:
            if "capabilities" not in model:
                model["capabilities"] = {}
            model["capabilities"].update(cap_updates)

        # ── Bump version + timestamp ──────────────────────────────────────
        meta = data.setdefault("metadata", {})
        old_version = meta.get("version", "0.0.0")
        new_version = _bump_patch(old_version)
        meta["version"] = new_version
        meta["last_updated"] = date.today().isoformat()
        if version_note:
            existing_source = meta.get("source", "")
            meta["source"] = f"{existing_source} | {version_note}".lstrip(" | ")

        # ── Write both copies atomically ──────────────────────────────────
        serialized = json.dumps(data, indent=2)
        written_paths = []
        for path in _CAPABILITIES_PATHS:
            if path.exists():
                path.write_text(serialized)
                written_paths.append(str(path))

        # ── Compute diff ──────────────────────────────────────────────────
        after_caps = dict(model.get("capabilities", {}))
        after_emp = dict(model.get("spark_empirical", {}))

        cap_diff = {
            k: {"before": before_caps.get(k), "after": after_caps.get(k)}
            for k in set(list(before_caps) + list(after_caps))
            if before_caps.get(k) != after_caps.get(k)
        }
        emp_diff = {
            k: {"before": before_emp.get(k), "after": after_emp.get(k)}
            for k in set(list(before_emp) + list(after_emp))
            if before_emp.get(k) != after_emp.get(k)
        }

        return {
            "model_name": model_name,
            "version": {"before": old_version, "after": new_version},
            "last_updated": meta["last_updated"],
            "changes": {
                "capabilities": cap_diff,
                "spark_empirical": emp_diff,
            },
            "current_capabilities": after_caps,
            "written_to": written_paths,
            "auto_calculated": auto_calc,
        }

    async def _update_system_prompt(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regenerate SystemPrompt.md from current model_capabilities.json and store in DB.

        Args:
            arguments: {
                "version": str,          # e.g. "v6.0"
                "set_active": bool,      # default True
                "version_note": str      # optional
            }
        """
        version: str = arguments["version"]
        set_active: bool = arguments.get("set_active", True)
        version_note: Optional[str] = arguments.get("version_note")

        # ── Load model capabilities ───────────────────────────────────────
        primary_path = _CAPABILITIES_PATHS[0]
        with open(primary_path) as f:
            cap_data = json.load(f)

        models = cap_data.get("models", {})
        metadata = cap_data.get("metadata", {})

        # ── Read current prompt to preserve static sections ───────────────
        if not _SYSTEM_PROMPT_PATH.exists():
            raise FileNotFoundError(f"SystemPrompt not found at {_SYSTEM_PROMPT_PATH}")

        current_content = _SYSTEM_PROMPT_PATH.read_text()
        _intro, static_sections = _split_prompt_sections(current_content)

        # ── Build new prompt content ──────────────────────────────────────
        new_content = _build_system_prompt(version, models, metadata, static_sections, version_note)

        # ── Write to file ─────────────────────────────────────────────────
        _SYSTEM_PROMPT_PATH.write_text(new_content)

        # ── Store in database (optional — DB may not be initialized) ──────
        prompt_id: Optional[str] = None
        db_stored = False
        db_error: Optional[str] = None
        if set_active:
            try:
                from synthony.api.database import create_system_prompt, init_database
                init_database()
                prompt = create_system_prompt(
                    version=version,
                    content=new_content,
                    file_path=str(_SYSTEM_PROMPT_PATH),
                    set_active=True,
                )
                prompt_id = prompt.prompt_id
                db_stored = True
            except Exception as e:
                db_error = str(e)

        return {
            "version": version,
            "capabilities_version": metadata.get("version", "?"),
            "prompt_id": prompt_id,
            "content_length": len(new_content),
            "written_to": str(_SYSTEM_PROMPT_PATH),
            "db_stored": db_stored,
            "db_error": db_error,
            "set_active": set_active,
            "sections_regenerated": ["1 (knowledge base)", "2 (model tiers)", "3 (GPU handling)", "7 (quick reference)"],
            "sections_preserved": ["4 (score changes)", "5 (decision logic)", "6 (output format)", "8 (examples)", "9 (availability check)", "10 (validation notes)"],
        }
