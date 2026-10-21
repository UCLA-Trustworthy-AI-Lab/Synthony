"""Orchestrator -- drives one run through its state machine.

Advise mode: PENDING -> PROFILING -> REASONING -> COMPLETED/FAILED/CANCELLED.

Full mode (Phase 4): PENDING -> PROFILING -> [REASONING -> TRAINING ->
BENCHMARKING -> (below threshold) ITERATING -> REASONING -> ...] ->
COMPLETED/FAILED/CANCELLED. A training failure is a *candidate* failure
(try the next ranked candidate from the same reasoning turn, not a run
failure); after ``max_iterations`` the run completes with the best attempt
seen so far and ``result_json.met_threshold=False``.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

from synthony.agents.benchmark import BenchmarkAgent
from synthony.agents.context import AgentContext
from synthony.agents.errors import RunCancelledError, is_transient
from synthony.agents.profiler import ProfilerAgent
from synthony.agents.recommender import RecommenderAgent
from synthony.agents.synthesizer import SynthesizerAgent
from synthony.api.database import finish_run, set_run_iteration, update_run_status
from synthony.runners.base import SynthesizerRunner
from synthony.runners.registry import executable_models

logger = logging.getLogger("synthony.runs")

_MAX_STAGE_RETRIES = 2


class RunStatus(str, Enum):
    PENDING = "pending"
    PROFILING = "profiling"
    REASONING = "reasoning"
    TRAINING = "training"
    BENCHMARKING = "benchmarking"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Orchestrator:
    """Drives a single run's agents, persisting status before each stage."""

    def __init__(self, context: AgentContext, runner: SynthesizerRunner | None = None):
        self.context = context
        # None -> SynthesizerAgent falls back to a real SynthcityRunner.
        # Tests inject a FakeRunner here to avoid a real (~seconds) subprocess.
        self._runner = runner

    async def run(self) -> None:
        state = self.context.state
        run_id = state.run_id

        try:
            update_run_status(run_id, RunStatus.PROFILING.value)
            if state.cancel_event.is_set():
                raise RunCancelledError()
            await self._run_with_retry(ProfilerAgent(self.context))

            if state.config.mode == "advise":
                update_run_status(run_id, RunStatus.REASONING.value)
                if state.cancel_event.is_set():
                    raise RunCancelledError()
                result = await self._run_with_retry(RecommenderAgent(self.context))
            else:
                result = await self._run_full_mode()

            finish_run(run_id, RunStatus.COMPLETED.value, result=result)

        except (RunCancelledError, asyncio.CancelledError):
            finish_run(run_id, RunStatus.CANCELLED.value, error_message="cancelled by user")
        except Exception as exc:
            finish_run(run_id, RunStatus.FAILED.value, error_message=str(exc))
            logger.exception(
                "run_failed",
                extra={"event": "run_failed", "run_id": run_id, "agent": None, "iteration": None},
            )

    # -- full mode: reason -> train -> benchmark -> iterate ------------------

    async def _run_full_mode(self) -> dict:
        state = self.context.state
        run_id = state.run_id
        cfg = state.config
        best_attempt: dict | None = None
        best_score = -1.0

        while True:
            update_run_status(run_id, RunStatus.REASONING.value)
            if state.cancel_event.is_set():
                raise RunCancelledError()
            recommendation = await self._run_with_retry(RecommenderAgent(self.context))
            state.final_recommendation = recommendation

            attempt = await self._try_candidates(recommendation["ranked_candidates"])
            if attempt is not None:
                state.attempts.append(attempt)
                if attempt["benchmark"]["overall_quality_score"] > best_score:
                    best_score = attempt["benchmark"]["overall_quality_score"]
                    best_attempt = attempt
                if best_score >= cfg.quality_threshold:
                    return {
                        "met_threshold": True,
                        "best_attempt": best_attempt,
                        "attempts": state.attempts,
                    }

            state.iteration += 1
            set_run_iteration(run_id, state.iteration)
            if state.iteration >= cfg.max_iterations:
                return {
                    "met_threshold": False,
                    "best_attempt": best_attempt,
                    "attempts": state.attempts,
                }

            update_run_status(run_id, RunStatus.ITERATING.value)
            if state.cancel_event.is_set():
                raise RunCancelledError()

    async def _try_candidates(self, ranked_candidates: list[dict]) -> dict | None:
        """Trains + benchmarks candidates from one reasoning turn in order,
        stopping at the first that trains successfully. Returns ``None`` if
        every candidate in the list failed to train (caller advances the
        iteration and re-enters REASONING)."""
        state = self.context.state
        run_id = state.run_id

        for candidate in ranked_candidates:
            model_name = candidate["model_name"]
            if model_name not in executable_models():
                # Defense-in-depth: RecommenderAgent already rejects a
                # non-executable *top* candidate in mode="full", but
                # alternatives (position 2-3) or a rule-based fallback may
                # still include one.
                continue

            update_run_status(run_id, RunStatus.TRAINING.value)
            if state.cancel_event.is_set():
                raise RunCancelledError()
            train_result = await self._run_with_retry(
                SynthesizerAgent(self.context, model_name=model_name, runner=self._runner)
            )
            if state.cancel_event.is_set():
                raise RunCancelledError()
            if not train_result["ok"]:
                state.attempts.append(
                    {
                        "iteration": state.iteration,
                        "model_name": model_name,
                        "training_failed": True,
                        "error": train_result["error"],
                    }
                )
                continue

            update_run_status(run_id, RunStatus.BENCHMARKING.value)
            if state.cancel_event.is_set():
                raise RunCancelledError()
            benchmark_result = await self._run_with_retry(
                BenchmarkAgent(self.context, output_path=train_result["output_path"])
            )
            return {
                "iteration": state.iteration,
                "model_name": model_name,
                "training_failed": False,
                "output_path": train_result["output_path"],
                "benchmark": benchmark_result,
            }

        return None

    async def _run_with_retry(self, agent) -> dict:
        """Retries transient failures by re-invoking ``agent.run()`` wholesale.

        Re-invoking rather than looping inside the agent means each attempt
        naturally gets its own fresh StepRecorder rows -- no extra
        bookkeeping needed for "each retry is its own recorded step".
        """
        attempt = 0
        while True:
            try:
                return await agent.run()
            except (RunCancelledError, asyncio.CancelledError):
                raise
            except Exception as exc:
                if is_transient(exc) and attempt < _MAX_STAGE_RETRIES:
                    attempt += 1
                    await asyncio.sleep(2**attempt)
                    continue
                raise
