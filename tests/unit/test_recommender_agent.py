"""Unit tests for RecommenderAgent's multi-turn reasoning loop
(Agentic Mode Phase 3): happy path, schema repair, budget exhaustion
(turns and tokens) forcing a submit, and graceful fallback to rule-based.
"""

import asyncio

import pytest

from synthony.agents.context import AgentContext, RunConfig, RunState
from synthony.agents.monitoring import StepRecorder
from synthony.agents.recommender import RecommenderAgent
from synthony.api.database import create_run, get_run_with_steps
from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.llm import LLMResponse, LLMUsage, ToolCall
from synthony.recommender.engine import ModelRecommendationEngine
from tests.conftest import FakeLLMClient


def _build_context(df, llm_client, config=None):
    run = create_run(mode="advise")
    profile = StochasticDataAnalyzer().analyze(df)
    column_analysis = ColumnAnalyzer().analyze(df, profile)
    state = RunState(
        run_id=run.run_id,
        config=config or RunConfig(dataset_id="ds1"),
        recorder=StepRecorder(),
        cancel_event=asyncio.Event(),
        dataset_profile=profile,
        column_analysis=column_analysis,
    )
    ctx = AgentContext(
        state=state, llm_client=llm_client, engine=ModelRecommendationEngine(), df=df
    )
    return ctx, run.run_id


def _tool_call(name, call_id="c", **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _submit_call(call_id="submit", model_name="CTGAN"):
    return ToolCall(
        id=call_id,
        name="submit_recommendation",
        arguments={
            "ranked_candidates": [
                {
                    "model_name": model_name,
                    "confidence": 0.8,
                    "hyperparameter_hints": {},
                    "rationale": "fits the profile",
                }
            ],
            "reasoning": "chosen using the rule-based anchor",
        },
    )


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_submits_after_profile_and_rule_based_calls(self, isolated_db, sample_normal_df):
        fake = FakeLLMClient(
            [
                LLMResponse(
                    tool_calls=[_tool_call("get_profile_overview")], stop_reason="tool_use"
                ),
                LLMResponse(
                    tool_calls=[_tool_call("get_rule_based_scores")], stop_reason="tool_use"
                ),
                LLMResponse(tool_calls=[_submit_call()], stop_reason="tool_use"),
            ]
        )
        ctx, run_id = _build_context(sample_normal_df, fake)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is False
        assert result["ranked_candidates"][0]["model_name"] == "CTGAN"
        assert ctx.state.final_recommendation == result
        assert len(fake.calls) == 3

        step_names = [s["step_name"] for s in get_run_with_steps(run_id)["steps"]]
        assert step_names == ["reason_turn_1", "reason_turn_2", "reason_turn_3"]


class TestSchemaRepair:
    @pytest.mark.asyncio
    async def test_invalid_then_valid_submission_repairs_once(self, isolated_db, sample_normal_df):
        bad_submit = ToolCall(
            id="bad",
            name="submit_recommendation",
            # Empty ranked_candidates list violates the schema's min_length=1.
            arguments={"ranked_candidates": [], "reasoning": "x"},
        )
        fake = FakeLLMClient(
            [
                LLMResponse(tool_calls=[bad_submit], stop_reason="tool_use"),
                LLMResponse(tool_calls=[_submit_call(call_id="good")], stop_reason="tool_use"),
            ]
        )
        ctx, run_id = _build_context(sample_normal_df, fake)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is False
        tool_messages = [m for m in ctx.state.conversation if m.role == "tool"]
        assert any(tr.is_error for m in tool_messages for tr in m.tool_results)


class TestBudgetExhaustion:
    @pytest.mark.asyncio
    async def test_turn_budget_exhaustion_forces_submit(self, isolated_db, sample_normal_df):
        never_submit = LLMResponse(
            tool_calls=[_tool_call("get_profile_overview")], stop_reason="tool_use"
        )
        forced_submit = LLMResponse(tool_calls=[_submit_call()], stop_reason="tool_use")
        fake = FakeLLMClient([never_submit, never_submit, forced_submit])
        config = RunConfig(dataset_id="ds1", max_reasoning_turns=2)
        ctx, run_id = _build_context(sample_normal_df, fake, config=config)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is False
        assert len(fake.calls) == 3  # 2 regular turns + 1 forced
        _, forced_kwargs = fake.calls[-1]
        assert forced_kwargs["tool_choice"] == "submit_recommendation"

    @pytest.mark.asyncio
    async def test_token_budget_trips_before_turn_budget(self, isolated_db, sample_normal_df):
        huge_usage_response = LLMResponse(
            tool_calls=[_tool_call("get_profile_overview")],
            stop_reason="tool_use",
            usage=LLMUsage(input_tokens=100_000, output_tokens=100_000),
        )
        fake = FakeLLMClient(
            [huge_usage_response, LLMResponse(tool_calls=[_submit_call()], stop_reason="tool_use")]
        )
        config = RunConfig(dataset_id="ds1", max_reasoning_turns=8, max_run_tokens=150_000)
        ctx, run_id = _build_context(sample_normal_df, fake, config=config)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is False
        assert len(fake.calls) == 2  # 1 regular turn (trips the token cap) + 1 forced


class TestGracefulFallback:
    @pytest.mark.asyncio
    async def test_fallback_to_rule_based_when_forced_submit_also_fails(
        self, isolated_db, sample_normal_df
    ):
        never_submit = LLMResponse(
            tool_calls=[_tool_call("get_profile_overview")], stop_reason="tool_use"
        )
        fake = FakeLLMClient([never_submit])  # repeats forever, including the forced call
        config = RunConfig(dataset_id="ds1", max_reasoning_turns=2)
        ctx, run_id = _build_context(sample_normal_df, fake, config=config)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is True
        assert result["ranked_candidates"]
        assert ctx.state.rule_based_result is not None

        step_names = [s["step_name"] for s in get_run_with_steps(run_id)["steps"]]
        assert step_names[-1] == "fallback_rule_based"


class TestNoLLMClient:
    @pytest.mark.asyncio
    async def test_no_llm_client_goes_straight_to_rule_based(self, isolated_db, sample_normal_df):
        ctx, run_id = _build_context(sample_normal_df, llm_client=None)

        result = await RecommenderAgent(ctx).run()

        assert result["degraded"] is True
        step_names = [s["step_name"] for s in get_run_with_steps(run_id)["steps"]]
        assert step_names == ["fallback_rule_based"]
