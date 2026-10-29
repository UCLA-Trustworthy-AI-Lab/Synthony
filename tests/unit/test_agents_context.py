"""Unit tests for RunConfig/RunState/AgentContext (Agentic Mode Phase 3)."""

import asyncio

import pandas as pd
import pytest
from pydantic import ValidationError

from synthony.agents.context import AgentContext, RunConfig, RunState
from synthony.agents.monitoring import StepRecorder
from synthony.core.schemas import RecommendationConstraints
from synthony.recommender.engine import ModelRecommendationEngine


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.mode == "advise"
        assert cfg.max_reasoning_turns == 8
        assert cfg.max_run_tokens == 150_000
        assert cfg.step_timeout_s == 900
        assert cfg.max_iterations == 3
        assert isinstance(cfg.constraints, RecommendationConstraints)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            RunConfig(mode="bogus")

    def test_round_trips_through_model_dump(self):
        cfg = RunConfig(dataset_id="abc", llm_provider="anthropic", llm_model="claude-opus-4-8")
        dumped = cfg.model_dump()
        assert dumped["dataset_id"] == "abc"
        restored = RunConfig(**dumped)
        assert restored.llm_provider == "anthropic"
        assert restored.llm_model == "claude-opus-4-8"


class TestRunStateAndAgentContext:
    def test_run_state_defaults(self):
        state = RunState(
            run_id="r1",
            config=RunConfig(),
            recorder=StepRecorder(),
            cancel_event=asyncio.Event(),
        )
        assert state.dataset_profile is None
        assert state.column_analysis is None
        assert state.rule_based_result is None
        assert state.final_recommendation is None
        assert state.conversation == []
        assert state.tokens_used == 0
        assert not state.cancel_event.is_set()

    def test_agent_context_construction(self):
        state = RunState(
            run_id="r1", config=RunConfig(), recorder=StepRecorder(), cancel_event=asyncio.Event()
        )
        engine = ModelRecommendationEngine()
        df = pd.DataFrame({"a": [1, 2, 3]})

        ctx = AgentContext(state=state, llm_client=None, engine=engine, df=df)

        assert ctx.llm_client is None
        assert ctx.engine is engine
        assert ctx.df.shape == (3, 1)
