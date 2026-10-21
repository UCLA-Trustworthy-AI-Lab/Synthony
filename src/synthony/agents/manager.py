"""RunManager -- schedules and tracks in-process agentic runs.

No durable job queue (single-uvicorn-worker assumption, documented in
``startup_recovery``): runs live only as long as the process does, and any
run left non-terminal across a restart is presumed crashed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from synthony.agents.context import AgentContext, RunConfig, RunState
from synthony.agents.monitoring import StepRecorder
from synthony.agents.orchestrator import Orchestrator
from synthony.api.database import create_run, finish_run, list_runs
from synthony.llm import LLMClient, resolve_llm_client

if TYPE_CHECKING:
    from synthony.recommender.engine import ModelRecommendationEngine
    from synthony.runners.base import SynthesizerRunner

LLMClientFactory = Callable[[RunConfig], "LLMClient | None"]
RunnerFactory = Callable[[RunConfig], "SynthesizerRunner | None"]

_NON_TERMINAL_STATUSES = (
    "pending",
    "profiling",
    "reasoning",
    "training",
    "benchmarking",
    "iterating",
)


def _default_llm_client_factory(config: RunConfig) -> LLMClient | None:
    # None (llm_provider="none", or no credentials) -> rule-based reasoning
    return resolve_llm_client(provider=config.llm_provider, model=config.llm_model)


class RunManager:
    def __init__(
        self,
        engine: ModelRecommendationEngine,
        max_concurrent_runs: int = 2,
        llm_client_factory: LLMClientFactory | None = None,
        runner_factory: RunnerFactory | None = None,
    ):
        self._engine = engine
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._llm_client_factory = llm_client_factory or _default_llm_client_factory
        # None -> Orchestrator falls back to a real SynthcityRunner (mode="full"
        # only; unused in advise mode). Tests inject a FakeRunner here to avoid
        # a real subprocess.
        self._runner_factory: RunnerFactory = runner_factory or (lambda config: None)
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def start_run(self, config: RunConfig, df: pd.DataFrame, owner: str | None = None) -> str:
        """Creates the ``Run`` row and schedules the orchestrator; returns
        immediately (does not block on the concurrency semaphore).

        ``owner`` is opaque process/session identity (Phase 5: the MCP
        server stamps its own instance ID here so ``cancel_run`` can refuse
        to touch a run started by a different session). The REST API
        doesn't pass one -- those runs stay ownerless.
        """
        run = create_run(
            mode=config.mode,
            dataset_id=config.dataset_id,
            owner=owner,
            config=config.model_dump(),
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
        )
        cancel_event = asyncio.Event()
        self._cancel_events[run.run_id] = cancel_event

        state = RunState(
            run_id=run.run_id,
            config=config,
            recorder=StepRecorder(),
            cancel_event=cancel_event,
        )
        llm_client = self._llm_client_factory(config)
        context = AgentContext(state=state, llm_client=llm_client, engine=self._engine, df=df)

        task = asyncio.create_task(self._run_guarded(context))
        self._tasks[run.run_id] = task
        return run.run_id

    async def _run_guarded(self, context: AgentContext) -> None:
        run_id = context.state.run_id
        try:
            async with self._semaphore:
                runner = self._runner_factory(context.state.config)
                await Orchestrator(context, runner=runner).run()
        finally:
            # Kept until the run actually finishes so cancel() stays valid
            # for the whole run duration, not just while queued/running.
            self._tasks.pop(run_id, None)
            self._cancel_events.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        """Sets the run's cancel event (clean cancelled-vs-failed status
        attribution) AND cancels its task (actual mid-await interruption --
        an Event alone cannot interrupt a blocked `await complete()` call).
        """
        event = self._cancel_events.get(run_id)
        task = self._tasks.get(run_id)
        if event is None and task is None:
            return False
        if event is not None:
            event.set()
        if task is not None and not task.done():
            task.cancel()
        return True

    async def startup_recovery(self) -> int:
        """Marks any run left in a non-terminal status by a prior crash as
        failed. Known limitation: single-uvicorn-worker only -- a durable
        queue that survives restarts is out of scope for Phase 3."""
        recovered = 0
        for status in _NON_TERMINAL_STATUSES:
            for run in list_runs(status=status, limit=10_000):
                finish_run(
                    run.run_id, "failed", error_message="orphaned at process start (prior crash)"
                )
                recovered += 1
        return recovered
