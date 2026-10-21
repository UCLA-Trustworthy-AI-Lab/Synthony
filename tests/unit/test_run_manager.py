"""Unit tests for RunManager (Agentic Mode Phase 3): concurrency limiting,
real cancellation (Event + Task.cancel()), and startup_recovery.
"""

import asyncio
import contextlib
import json

import pytest

from synthony.agents.context import RunConfig
from synthony.agents.manager import RunManager
from synthony.api.database import create_run, finish_run, get_run, update_run_status
from synthony.llm import LLMResponse, ToolCall
from synthony.recommender.engine import ModelRecommendationEngine
from tests.conftest import FakeLLMClient, happy_path_llm_responses


@pytest.mark.asyncio
async def test_start_run_returns_immediately_and_completes(isolated_db, sample_normal_df):
    fake = FakeLLMClient(happy_path_llm_responses())
    manager = RunManager(engine=ModelRecommendationEngine(), llm_client_factory=lambda cfg: fake)

    run_id = await manager.start_run(RunConfig(dataset_id="ds1"), sample_normal_df)
    task = manager._tasks[run_id]

    await task  # wait for the guarded run to finish

    assert get_run(run_id).status == "completed"
    assert run_id not in manager._tasks
    assert run_id not in manager._cancel_events


@pytest.mark.asyncio
async def test_semaphore_serializes_concurrent_runs(isolated_db, sample_normal_df):
    active = 0
    max_active = 0

    class TrackingClient(FakeLLMClient):
        async def complete(self, messages, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return await super().complete(messages, **kwargs)

    def make_fake(_config):
        return TrackingClient(happy_path_llm_responses())

    manager = RunManager(
        engine=ModelRecommendationEngine(), max_concurrent_runs=1, llm_client_factory=make_fake
    )

    run_id_1 = await manager.start_run(RunConfig(dataset_id="ds1"), sample_normal_df)
    run_id_2 = await manager.start_run(RunConfig(dataset_id="ds1"), sample_normal_df)

    await asyncio.gather(manager._tasks[run_id_1], manager._tasks[run_id_2])

    assert max_active == 1  # never both in-flight at once, despite starting both immediately
    assert get_run(run_id_1).status == "completed"
    assert get_run(run_id_2).status == "completed"


@pytest.mark.asyncio
async def test_cancel_interrupts_a_hanging_task(isolated_db, sample_normal_df):
    class HangingClient(FakeLLMClient):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(10)  # would hang forever without real Task.cancel()
            return await super().complete(messages, **kwargs)

    first_call = LLMResponse(
        tool_calls=[ToolCall(id="c1", name="get_profile_overview", arguments={})],
        stop_reason="tool_use",
    )
    fake = HangingClient([first_call])
    manager = RunManager(engine=ModelRecommendationEngine(), llm_client_factory=lambda cfg: fake)

    run_id = await manager.start_run(RunConfig(dataset_id="ds1"), sample_normal_df)
    await asyncio.sleep(0.05)  # let the task actually reach the hanging complete() call

    assert manager.cancel(run_id) is True

    task = manager._tasks.get(run_id)
    if task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    assert get_run(run_id).status == "cancelled"


def test_cancel_returns_false_for_unknown_run():
    manager = RunManager(engine=ModelRecommendationEngine())
    assert manager.cancel("does-not-exist") is False


class TestStartupRecovery:
    @pytest.mark.asyncio
    async def test_marks_only_nonterminal_runs_failed(self, isolated_db):
        pending = create_run(mode="advise")
        profiling = create_run(mode="advise")
        update_run_status(profiling.run_id, "profiling")
        completed = create_run(mode="advise")
        finish_run(completed.run_id, "completed", result={})

        manager = RunManager(engine=ModelRecommendationEngine())
        recovered = await manager.startup_recovery()

        assert recovered == 2
        assert get_run(pending.run_id).status == "failed"
        assert get_run(profiling.run_id).status == "failed"
        assert get_run(completed.run_id).status == "completed"


class TestRunsWithoutAnLLM:
    """Agentic mode must work with no LLM: explicitly (llm_provider="none") or
    because no credentials are configured. Both run the documented rule-based
    path and mark the result degraded, instead of failing on a 401."""

    @pytest.fixture(autouse=True)
    def _no_credentials(self, monkeypatch):
        for var in ("SYNTHONY_LLM_PROVIDER", "SYNTHONY_LLM_BASE_URL", "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY", "VLLM_URL", "VLLM_API_KEY", "VLLM_MODEL"):
            monkeypatch.delenv(var, raising=False)

    async def _run(self, config, df):
        manager = RunManager(engine=ModelRecommendationEngine())  # default (real) client factory
        run_id = await manager.start_run(config, df)
        await manager._tasks[run_id]
        return get_run(run_id)

    @pytest.mark.asyncio
    async def test_advise_without_credentials_completes_rule_based(
        self, isolated_db, sample_normal_df
    ):
        run = await self._run(RunConfig(dataset_id="ds1"), sample_normal_df)

        assert run.status == "completed", run.error_message
        result = json.loads(run.result_json) if isinstance(run.result_json, str) else run.result
        rec = result.get("recommendation", result)
        assert rec["degraded"] is True
        assert "no llm" in rec["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_advise_with_llm_provider_none_completes_rule_based(
        self, isolated_db, sample_normal_df, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-present-but-explicitly-unused")
        run = await self._run(RunConfig(dataset_id="ds1", llm_provider="none"), sample_normal_df)

        assert run.status == "completed", run.error_message
