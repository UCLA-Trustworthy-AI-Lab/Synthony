"""Unit tests for ProfilerAgent (Agentic Mode Phase 3)."""

import asyncio

import pytest

from synthony.agents.context import AgentContext, RunConfig, RunState
from synthony.agents.monitoring import StepRecorder
from synthony.agents.profiler import ProfilerAgent
from synthony.api.database import create_run, get_run_with_steps
from synthony.recommender.engine import ModelRecommendationEngine


@pytest.mark.asyncio
async def test_profiler_agent_populates_state_and_records_one_step(isolated_db, sample_normal_df):
    run = create_run(mode="advise")
    state = RunState(
        run_id=run.run_id,
        config=RunConfig(dataset_id="ds1"),
        recorder=StepRecorder(),
        cancel_event=asyncio.Event(),
    )
    ctx = AgentContext(
        state=state, llm_client=None, engine=ModelRecommendationEngine(), df=sample_normal_df
    )

    summary = await ProfilerAgent(ctx).run()

    assert state.dataset_profile is not None
    assert state.dataset_profile.row_count == 1000
    assert state.column_analysis is not None
    assert set(state.column_analysis.columns) == set(sample_normal_df.columns)
    assert summary["row_count"] == 1000
    assert summary["column_count"] == len(sample_normal_df.columns)

    detail = get_run_with_steps(run.run_id)
    assert len(detail["steps"]) == 1
    assert detail["steps"][0]["agent"] == "profiler"
    assert detail["steps"][0]["step_name"] == "analyze"
    assert detail["steps"][0]["status"] == "completed"
