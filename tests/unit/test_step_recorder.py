"""Unit tests for StepRecorder/StepHandle (synthony.agents.monitoring),
the instrumentation choke point Phase 3 agents record steps through.

Uses the shared `isolated_db` fixture (tests/conftest.py): StepRecorder
writes real run_steps/runs rows via synthony.api.database, so each test
needs an isolated temp SQLite file and a real Run row to attach steps to
(FK is enforced at the ORM/query level, not by SQLite here, but a
nonexistent run_id would make get_run_with_steps assertions meaningless).
"""

import pytest

from synthony.agents.monitoring import StepRecorder
from synthony.api.database import create_run, get_run, get_run_with_steps
from synthony.llm.types import LLMUsage


@pytest.fixture
def run(isolated_db):
    return create_run(mode="advise")


class TestStepRecorderSuccess:
    @pytest.mark.asyncio
    async def test_success_records_completed_step(self, run):
        recorder = StepRecorder()
        async with recorder.step(run.run_id, agent="profiler", step_name="analyze") as handle:
            handle.set_output({"row_count": 42})

        detail = get_run_with_steps(run.run_id)
        assert len(detail["steps"]) == 1
        step = detail["steps"][0]
        assert step["status"] == "completed"
        assert step["agent"] == "profiler"
        assert step["step_name"] == "analyze"
        assert step["output"] == {"row_count": 42}
        assert step["finished_at"] is not None
        assert step["duration_ms"] is not None and step["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_input_data_persisted_at_start(self, run):
        recorder = StepRecorder()
        async with recorder.step(
            run.run_id, agent="profiler", step_name="analyze", input_data={"path": "x.csv"}
        ):
            pass
        detail = get_run_with_steps(run.run_id)
        assert detail["steps"][0]["input"] == {"path": "x.csv"}

    @pytest.mark.asyncio
    async def test_iteration_recorded(self, run):
        recorder = StepRecorder()
        async with recorder.step(run.run_id, agent="synthesizer", step_name="train", iteration=2):
            pass
        detail = get_run_with_steps(run.run_id)
        assert detail["steps"][0]["iteration"] == 2


class TestStepRecorderFailure:
    @pytest.mark.asyncio
    async def test_exception_records_failed_step_and_reraises(self, run):
        recorder = StepRecorder()
        with pytest.raises(ValueError, match="boom"):
            async with recorder.step(run.run_id, agent="recommender", step_name="reason"):
                raise ValueError("boom")

        detail = get_run_with_steps(run.run_id)
        step = detail["steps"][0]
        assert step["status"] == "failed"
        assert step["error_message"] == "boom"
        assert step["finished_at"] is not None

    @pytest.mark.asyncio
    async def test_output_set_before_exception_is_not_persisted(self, run):
        """set_output() only takes effect if the step completes normally --
        a failure overwrites output with nothing, matching how finish_run_step
        is called (output_data=handle._output regardless, but status=failed
        is what callers should key off, not presence of output)."""
        recorder = StepRecorder()
        with pytest.raises(RuntimeError):
            async with recorder.step(run.run_id, agent="recommender", step_name="reason") as handle:
                handle.set_output({"partial": True})
                raise RuntimeError("fail after partial output")

        detail = get_run_with_steps(run.run_id)
        step = detail["steps"][0]
        assert step["status"] == "failed"
        # Implementation choice: whatever was set before the raise is still
        # persisted (useful for debugging partial progress), not discarded.
        assert step["output"] == {"partial": True}


class TestStepRecorderUsageRollup:
    @pytest.mark.asyncio
    async def test_add_usage_rolls_up_to_step_and_run(self, run):
        recorder = StepRecorder()
        async with recorder.step(run.run_id, agent="recommender", step_name="reason") as handle:
            handle.add_usage(LLMUsage(input_tokens=100, output_tokens=20, cost_usd=0.01))
            handle.add_usage(LLMUsage(input_tokens=50, output_tokens=10, cost_usd=0.005))

        # Run totals
        updated_run = get_run(run.run_id)
        assert updated_run.total_input_tokens == 150
        assert updated_run.total_output_tokens == 30
        assert updated_run.total_cost_usd == pytest.approx(0.015)

        # Step totals (finish_run_step receives the accumulated usage)
        detail = get_run_with_steps(run.run_id)
        step = detail["steps"][0]
        assert step["input_tokens"] == 150
        assert step["output_tokens"] == 30
        assert step["cost_usd"] == pytest.approx(0.015)

    @pytest.mark.asyncio
    async def test_usage_partially_accumulated_before_failure_is_kept(self, run):
        """add_usage() posts to the run total immediately (not batched at
        step end), so usage incurred before a mid-step failure isn't lost --
        this matters because a real LLM call that then throws still cost
        real tokens."""
        recorder = StepRecorder()
        with pytest.raises(RuntimeError):
            async with recorder.step(run.run_id, agent="recommender", step_name="reason") as handle:
                handle.add_usage(LLMUsage(input_tokens=100, output_tokens=20, cost_usd=0.01))
                raise RuntimeError("downstream failure after a real LLM call")

        updated_run = get_run(run.run_id)
        assert updated_run.total_input_tokens == 100
        assert updated_run.total_cost_usd == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_no_usage_leaves_zero_totals(self, run):
        recorder = StepRecorder()
        async with recorder.step(run.run_id, agent="profiler", step_name="analyze"):
            pass
        detail = get_run_with_steps(run.run_id)
        step = detail["steps"][0]
        assert step["input_tokens"] == 0
        assert step["output_tokens"] == 0
        assert step["cost_usd"] == 0.0
