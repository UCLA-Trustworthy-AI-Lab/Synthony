"""Shared run/agent state for Agentic Mode Phase 3 (advise mode).

Three related but distinct objects are threaded through the orchestrator
and agents for one run:

- ``RunConfig`` — the wire-schema config for a run (Pydantic), stored
  verbatim into ``Run.config_json``.
- ``RunState`` — the mutable, run-scoped accumulator (plain dataclass, not
  Pydantic: it carries live asyncio primitives and progressively-filled
  results, not a validated wire payload).
- ``AgentContext`` — the per-run dependency bag (state + LLM client +
  shared engine + loaded DataFrame) every agent receives.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from synthony.agents.monitoring import StepRecorder
from synthony.core.schemas import ColumnAnalysisResult, DatasetProfile, RecommendationConstraints
from synthony.llm import LLMClient, LLMMessage
from synthony.recommender.engine import ModelRecommendationEngine, RecommendationResult


class RunConfig(BaseModel):
    """Config for one agentic run. Stored verbatim into ``Run.config_json``."""

    model_config = {"protected_namespaces": ()}

    dataset_id: str | None = None
    dataset_path: str | None = None
    mode: Literal["advise", "full"] = "advise"
    n_rows: int | None = None
    quality_threshold: float = 0.75
    max_iterations: int = 3  # inert until Phase 4 (iterate loop)
    llm_provider: str | None = None
    llm_model: str | None = None
    runner_backend: str | None = None  # inert until Phase 4 (synthesizer runners)
    constraints: RecommendationConstraints = Field(default_factory=RecommendationConstraints)
    max_reasoning_turns: int = 8
    max_run_tokens: int = 150_000
    step_timeout_s: int = 900


@dataclass
class RunState:
    """Mutable accumulator threaded through the orchestrator and agents for one run."""

    run_id: str
    config: RunConfig
    recorder: StepRecorder
    cancel_event: asyncio.Event
    dataset_profile: DatasetProfile | None = None
    column_analysis: ColumnAnalysisResult | None = None
    rule_based_result: RecommendationResult | None = None  # memoized deterministic anchor
    final_recommendation: dict[str, Any] | None = None
    iteration: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)  # Phase 4: prior train/benchmark attempts
    conversation: list[LLMMessage] = field(default_factory=list)
    tokens_used: int = 0  # running total, avoids a DB read every reasoning turn


@dataclass
class AgentContext:
    """Per-run dependency bag passed to every agent.

    ``engine`` is a single ``ModelRecommendationEngine`` instance shared
    across all concurrent runs (it holds no per-request mutable state, so
    this is safe) — do not construct a new engine per run.
    """

    state: RunState
    llm_client: LLMClient | None
    engine: ModelRecommendationEngine
    df: pd.DataFrame
