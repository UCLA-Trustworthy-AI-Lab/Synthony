"""Unit tests for Orchestrator: status transitions across advise mode
(PENDING -> PROFILING -> REASONING -> COMPLETED/CANCELLED, Phase 3) and full
mode (... -> TRAINING -> BENCHMARKING -> ITERATING -> ..., Phase 4), plus the
coarse cancel_event.is_set() cancellation check.
"""

import asyncio
import json

import pytest

from synthony.agents.context import AgentContext, RunConfig, RunState
from synthony.agents.monitoring import StepRecorder
from synthony.agents.orchestrator import Orchestrator
from synthony.api.database import create_run, get_run, get_run_with_steps
from synthony.llm import LLMResponse, ToolCall
from synthony.recommender.engine import ModelRecommendationEngine
from tests.conftest import (
    FakeLLMClient,
    FakeRunner,
    happy_path_llm_responses,
    submit_recommendation_call,
)


def _build_context(df, llm_client, config=None):
    run = create_run(mode=(config.mode if config else "advise"))
    state = RunState(
        run_id=run.run_id,
        config=config or RunConfig(dataset_id="ds1"),
        recorder=StepRecorder(),
        cancel_event=asyncio.Event(),
    )
    ctx = AgentContext(
        state=state, llm_client=llm_client, engine=ModelRecommendationEngine(), df=df
    )
    return ctx, run.run_id


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_run_reaches_completed_with_expected_steps(
        self, isolated_db, sample_normal_df
    ):
        fake = FakeLLMClient(happy_path_llm_responses())
        ctx, run_id = _build_context(sample_normal_df, fake)

        await Orchestrator(ctx).run()

        detail = get_run_with_steps(run_id)
        assert detail["status"] == "completed"
        assert detail["result"]["degraded"] is False

        step_names = [s["step_name"] for s in detail["steps"]]
        assert step_names == ["analyze", "reason_turn_1", "reason_turn_2", "reason_turn_3"]
        assert all(s["status"] == "completed" for s in detail["steps"])


class TestFullMode:
    """FakeRunner-driven full-mode state-machine scenarios (Phase 4). Uses
    FakeLLMClient's "repeats the last response once exhausted" behavior so
    a single 3-turn happy-path script (profile -> rule-based -> submit
    CTGAN) covers any number of REASONING re-entries without re-scripting
    per iteration -- there's no code-level requirement that get_rule_based_
    scores be called before a resubmission.
    """

    @pytest.mark.asyncio
    async def test_met_threshold_on_first_iteration(self, isolated_db, sample_normal_df):
        config = RunConfig(dataset_id="ds1", mode="full", quality_threshold=0.75, max_iterations=3)
        fake_llm = FakeLLMClient(happy_path_llm_responses())
        fake_runner = FakeRunner("success")
        ctx, run_id = _build_context(sample_normal_df, fake_llm, config=config)

        await Orchestrator(ctx, runner=fake_runner).run()

        run = get_run(run_id)
        assert run.status == "completed"
        assert ctx.state.iteration == 0  # never advanced past iteration 0
        assert len(fake_runner.calls) == 1

        result = json.loads(run.result_json)
        assert result["met_threshold"] is True
        assert result["best_attempt"]["model_name"] == "CTGAN"

    @pytest.mark.asyncio
    async def test_met_threshold_on_second_iteration(self, isolated_db, sample_normal_df):
        config = RunConfig(dataset_id="ds1", mode="full", quality_threshold=0.75, max_iterations=3)
        fake_llm = FakeLLMClient(happy_path_llm_responses())
        fake_runner = FakeRunner(["low_quality", "success"])
        ctx, run_id = _build_context(sample_normal_df, fake_llm, config=config)

        await Orchestrator(ctx, runner=fake_runner).run()

        run = get_run(run_id)
        assert run.status == "completed"
        assert len(fake_runner.calls) == 2

        result = json.loads(run.result_json)
        assert result["met_threshold"] is True
        assert len(result["attempts"]) == 2

    @pytest.mark.asyncio
    async def test_never_meets_threshold_completes_with_best_so_far(
        self, isolated_db, sample_normal_df
    ):
        config = RunConfig(dataset_id="ds1", mode="full", quality_threshold=0.75, max_iterations=2)
        fake_llm = FakeLLMClient(happy_path_llm_responses())
        fake_runner = FakeRunner("low_quality")
        ctx, run_id = _build_context(sample_normal_df, fake_llm, config=config)

        await Orchestrator(ctx, runner=fake_runner).run()

        run = get_run(run_id)
        assert run.status == "completed"
        assert run.current_iteration == 2  # exhausted max_iterations
        assert len(fake_runner.calls) == 2

        result = json.loads(run.result_json)
        assert result["met_threshold"] is False
        assert result["best_attempt"] is not None  # best-so-far, not None

    @pytest.mark.asyncio
    async def test_training_failure_tries_next_ranked_candidate(
        self, isolated_db, sample_normal_df
    ):
        """A failed candidate doesn't fail the run -- the orchestrator tries
        the next ranked candidate from the same reasoning turn."""
        submit_two_candidates = LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit",
                    name="submit_recommendation",
                    arguments={
                        "ranked_candidates": [
                            {
                                "model_name": "CTGAN",
                                "confidence": 0.8,
                                "hyperparameter_hints": {},
                                "rationale": "primary",
                            },
                            {
                                "model_name": "TVAE",
                                "confidence": 0.6,
                                "hyperparameter_hints": {},
                                "rationale": "fallback",
                            },
                        ],
                        "reasoning": "test",
                    },
                )
            ],
            stop_reason="tool_use",
        )
        config = RunConfig(dataset_id="ds1", mode="full", quality_threshold=0.75, max_iterations=3)
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    tool_calls=[ToolCall(id="c1", name="get_profile_overview", arguments={})],
                    stop_reason="tool_use",
                ),
                submit_two_candidates,
            ]
        )
        fake_runner = FakeRunner(["fail", "success"])
        ctx, run_id = _build_context(sample_normal_df, fake_llm, config=config)

        await Orchestrator(ctx, runner=fake_runner).run()

        run = get_run(run_id)
        assert run.status == "completed"
        assert len(fake_runner.calls) == 2
        assert fake_runner.calls[0].model_name == "CTGAN"
        assert fake_runner.calls[1].model_name == "TVAE"

        result = json.loads(run.result_json)
        assert result["met_threshold"] is True
        assert result["best_attempt"]["model_name"] == "TVAE"
        # the failed CTGAN attempt is recorded too, not silently dropped
        failed_attempts = [a for a in result["attempts"] if a.get("training_failed")]
        assert len(failed_attempts) == 1
        assert failed_attempts[0]["model_name"] == "CTGAN"

    @pytest.mark.asyncio
    async def test_cancelled_during_training(self, isolated_db, sample_normal_df):
        """Simulates RunManager.cancel() firing while TRAINING is in
        flight: FakeRunner observes cancel_event was set and the run ends
        up cancelled, not stuck retrying candidates."""
        config = RunConfig(dataset_id="ds1", mode="full")
        fake_llm = FakeLLMClient(happy_path_llm_responses())
        ctx, run_id = _build_context(sample_normal_df, fake_llm, config=config)

        class CancellingRunner(FakeRunner):
            async def train_and_generate(self, job, cancel_event=None):
                if cancel_event is not None:
                    cancel_event.set()
                return await super().train_and_generate(job, cancel_event=cancel_event)

        await Orchestrator(ctx, runner=CancellingRunner("success")).run()

        run = get_run(run_id)
        assert run.status == "cancelled"


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_before_start_marks_run_cancelled(self, isolated_db, sample_normal_df):
        submit_response = LLMResponse(
            tool_calls=[submit_recommendation_call()], stop_reason="tool_use"
        )
        fake = FakeLLMClient([submit_response])
        ctx, run_id = _build_context(sample_normal_df, fake)
        ctx.state.cancel_event.set()

        await Orchestrator(ctx).run()

        run = get_run(run_id)
        assert run.status == "cancelled"
        assert fake.calls == []  # never got as far as calling the LLM

    @pytest.mark.asyncio
    async def test_cancelled_between_profiling_and_reasoning(self, isolated_db, sample_normal_df):
        """cancel_event is set by the profiler itself mid-stage (simulating
        RunManager.cancel() firing while PROFILING is in flight) -- the
        orchestrator's check before REASONING should catch it."""
        submit_response = LLMResponse(
            tool_calls=[submit_recommendation_call()], stop_reason="tool_use"
        )
        fake = FakeLLMClient([submit_response])
        ctx, run_id = _build_context(sample_normal_df, fake)

        from synthony.agents.profiler import ProfilerAgent

        class CancellingProfiler(ProfilerAgent):
            async def run(self):
                result = await super().run()
                self.context.state.cancel_event.set()
                return result

        import synthony.agents.orchestrator as orchestrator_module

        orchestrator_module.ProfilerAgent = CancellingProfiler
        try:
            await Orchestrator(ctx).run()
        finally:
            orchestrator_module.ProfilerAgent = ProfilerAgent

        run = get_run(run_id)
        assert run.status == "cancelled"
        assert fake.calls == []  # reasoning stage never actually called the LLM
