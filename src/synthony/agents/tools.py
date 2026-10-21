"""Tools exposed to the LLM inside ``RecommenderAgent``'s reasoning loop.

Each tool is a ``ToolSpec`` (the provider-neutral schema) paired with an
async handler ``(AgentContext, arguments) -> dict``. Handlers never raise —
failures are reported as a plain dict containing ``{"is_error": True,
"content": "..."}``. The reasoning loop (``RecommenderAgent``) pops
``is_error`` off the returned dict, JSON-serializes the remainder into
``ToolResult.content``, and sets ``ToolResult.is_error`` accordingly — that
mapping lives in the loop, not here, so this module only deals in plain
dicts.

``submit_recommendation`` is the loop's terminal tool: its schema lives
here (so both the loop and tests share one source of truth) but it is
handled specially by ``RecommenderAgent`` rather than dispatched through
the regular handler registry, since a successful call ends the loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from synthony.agents.context import AgentContext
from synthony.llm import ToolSpec
from synthony.runners.registry import executable_models


ToolHandler = Callable[[AgentContext, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class AgentTool:
    spec: ToolSpec
    handler: ToolHandler


# ---------------------------------------------------------------------------
# get_profile_overview
# ---------------------------------------------------------------------------

_GET_PROFILE_OVERVIEW_SPEC = ToolSpec(
    name="get_profile_overview",
    description=(
        "Get the dataset-level statistical profile: row/column counts, "
        "stress factors (severe skew, high cardinality, zipfian, small/large "
        "data, higher-order correlation), and detailed skewness/cardinality/"
        "zipfian/correlation metrics. Call this first -- it has no arguments."
    ),
    input_schema={"type": "object", "properties": {}},
)


async def _get_profile_overview(context: AgentContext, arguments: dict[str, Any]) -> dict[str, Any]:
    profile = context.state.dataset_profile
    if profile is None:
        return {"is_error": True, "content": "Dataset profile not available yet."}
    return profile.model_dump(exclude={"correlation": {"correlation_matrix"}})


# ---------------------------------------------------------------------------
# get_column_stats
# ---------------------------------------------------------------------------

_GET_COLUMN_STATS_SPEC = ToolSpec(
    name="get_column_stats",
    description=(
        "Get per-column difficulty scores and stress factors. Pass `columns` "
        "to filter to specific column names, or omit it for all columns."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column names to include; omit for all columns.",
            }
        },
    },
)


async def _get_column_stats(context: AgentContext, arguments: dict[str, Any]) -> dict[str, Any]:
    analysis = context.state.column_analysis
    if analysis is None:
        return {"is_error": True, "content": "Column analysis not available yet."}

    requested = arguments.get("columns")
    if not requested:
        return {"columns": {name: col.model_dump() for name, col in analysis.columns.items()}}

    known = set(analysis.columns)
    valid = [c for c in requested if c in known]
    unknown = [c for c in requested if c not in known]

    if not valid:
        return {
            "is_error": True,
            "content": f"None of the requested columns exist. Valid columns: {sorted(known)}",
        }

    result: dict[str, Any] = {
        "columns": {name: analysis.columns[name].model_dump() for name in valid}
    }
    if unknown:
        result["warnings"] = [f"Unknown column(s) ignored: {unknown}"]
    return result


# ---------------------------------------------------------------------------
# list_candidate_models
# ---------------------------------------------------------------------------

_LIST_CANDIDATE_MODELS_SPEC = ToolSpec(
    name="list_candidate_models",
    description=(
        "List all candidate synthetic-data models with their type, "
        "capability scores (0-4 scale), constraints, and whether a runner "
        "can currently execute them (`executable_by_runner`)."
    ),
    input_schema={"type": "object", "properties": {}},
)


async def _list_candidate_models(
    context: AgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    models = context.engine.model_capabilities.get("models", {})
    return {
        "models": {
            name: {
                "type": info.get("type"),
                "capabilities": info.get("capabilities", {}),
                "constraints": info.get("constraints", {}),
                "executable_by_runner": name in executable_models(),
            }
            for name, info in models.items()
        }
    }


# ---------------------------------------------------------------------------
# get_model_details
# ---------------------------------------------------------------------------

_GET_MODEL_DETAILS_SPEC = ToolSpec(
    name="get_model_details",
    description="Get full details for a single candidate model by name.",
    input_schema={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Model name, e.g. 'CTGAN'."}},
        "required": ["name"],
    },
)


async def _get_model_details(context: AgentContext, arguments: dict[str, Any]) -> dict[str, Any]:
    models = context.engine.model_capabilities.get("models", {})
    name = arguments.get("name", "")
    info = models.get(name)
    if info is None:
        return {
            "is_error": True,
            "content": f"Unknown model '{name}'. Known models: {sorted(models)}",
        }
    return {"model": info, "executable_by_runner": name in executable_models()}


# ---------------------------------------------------------------------------
# get_rule_based_scores
# ---------------------------------------------------------------------------

_GET_RULE_BASED_SCORES_SPEC = ToolSpec(
    name="get_rule_based_scores",
    description=(
        "Get the deterministic rule-based recommendation (no LLM involved) "
        "as an anchor before submitting your own recommendation. Call this "
        "at least once before `submit_recommendation`."
    ),
    input_schema={"type": "object", "properties": {}},
)


async def _get_rule_based_scores(
    context: AgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    state = context.state
    if state.dataset_profile is None:
        return {"is_error": True, "content": "Dataset profile not available yet."}

    if state.rule_based_result is None:
        # Memoized: deterministic and may be asked for more than once per run.
        state.rule_based_result = context.engine.recommend(
            state.dataset_profile,
            state.column_analysis,
            constraints=state.config.constraints,
            method="rule_based",
            top_n=3,
        )
    return state.rule_based_result.model_dump()


# ---------------------------------------------------------------------------
# get_prior_attempts
# ---------------------------------------------------------------------------

_GET_PRIOR_ATTEMPTS_SPEC = ToolSpec(
    name="get_prior_attempts",
    description=(
        "Get prior training/benchmark attempts from earlier iterations of "
        "this run, if any (always empty in advise mode -- there's no "
        "training loop to have attempted anything)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "iteration": {
                "type": "integer",
                "description": "Optional iteration number to filter to.",
            }
        },
    },
)


async def _get_prior_attempts(context: AgentContext, arguments: dict[str, Any]) -> dict[str, Any]:
    attempts = context.state.attempts
    iteration = arguments.get("iteration")
    if iteration is not None:
        attempts = [a for a in attempts if a.get("iteration") == iteration]
    return {"attempts": attempts}


# ---------------------------------------------------------------------------
# submit_recommendation -- terminal, schema lives here, handled by the loop
# ---------------------------------------------------------------------------


class RankedCandidate(BaseModel):
    model_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    hyperparameter_hints: dict[str, Any] = Field(default_factory=dict)
    rationale: str


class SubmissionPayload(BaseModel):
    ranked_candidates: list[RankedCandidate] = Field(min_length=1, max_length=3)
    reasoning: str


SUBMIT_RECOMMENDATION_SPEC = ToolSpec(
    name="submit_recommendation",
    description=(
        "Submit your final ranked model recommendation (1-3 candidates) and "
        "end the reasoning loop. Call `get_rule_based_scores` at least once "
        "before calling this."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ranked_candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "hyperparameter_hints": {"type": "object"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["model_name", "confidence", "rationale"],
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["ranked_candidates", "reasoning"],
    },
)


def validate_submission(arguments: dict[str, Any]) -> SubmissionPayload:
    """Raises ``pydantic.ValidationError`` on a malformed submission."""
    return SubmissionPayload.model_validate(arguments)


def build_recommender_tools() -> list[AgentTool]:
    """The regular (non-terminal) tools dispatched during the reasoning loop."""
    return [
        AgentTool(spec=_GET_PROFILE_OVERVIEW_SPEC, handler=_get_profile_overview),
        AgentTool(spec=_GET_COLUMN_STATS_SPEC, handler=_get_column_stats),
        AgentTool(spec=_LIST_CANDIDATE_MODELS_SPEC, handler=_list_candidate_models),
        AgentTool(spec=_GET_MODEL_DETAILS_SPEC, handler=_get_model_details),
        AgentTool(spec=_GET_RULE_BASED_SCORES_SPEC, handler=_get_rule_based_scores),
        AgentTool(spec=_GET_PRIOR_ATTEMPTS_SPEC, handler=_get_prior_attempts),
    ]
