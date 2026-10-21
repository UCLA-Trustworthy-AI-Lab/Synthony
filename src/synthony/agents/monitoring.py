"""Structured run/step logging and the StepRecorder instrumentation choke point.

Every agent step recorded through StepRecorder gets a persisted run_steps
row (via synthony.api.database) plus one JSON line to logs/runs.jsonl and
stderr, so run history is queryable both from SQL and by tailing/grepping
the log file. This is the single place agents (Phase 3+) route step
timing/status/usage through, so later monitoring backends (Prometheus/OTel)
can be added here without touching agent code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from synthony.api.database import accumulate_run_usage, add_run_step, finish_run_step, log_audit
from synthony.llm.types import LLMUsage

_LOG_DIR = Path("./logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("synthony.runs")
logger.setLevel(logging.INFO)


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: {ts, level, event, run_id, agent, iteration,
    duration_ms, tokens, cost_usd, msg}. Fields the caller didn't set via
    `extra=` come through as null rather than raising."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
            "run_id": getattr(record, "run_id", None),
            "agent": getattr(record, "agent", None),
            "iteration": getattr(record, "iteration", None),
            "duration_ms": getattr(record, "duration_ms", None),
            "tokens": getattr(record, "tokens", None),
            "cost_usd": getattr(record, "cost_usd", None),
            "msg": record.getMessage(),
        }
        return json.dumps(payload)


def _configure_handlers() -> None:
    """Attach handlers once. Guarded so re-importing this module (tests,
    reloaders) doesn't stack duplicate handlers onto the shared logger."""
    if logger.handlers:
        return
    file_handler = RotatingFileHandler(
        _LOG_DIR / "runs.jsonl", maxBytes=10_000_000, backupCount=3
    )
    file_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(stderr_handler)


_configure_handlers()


@dataclass
class StepHandle:
    """Handle to a running step, yielded by StepRecorder.step().

    Callers set the step's output via set_output() and roll any LLM usage
    incurred during the step up to both the step and run totals via
    add_usage() — each call immediately accumulates onto the run row so
    partial usage isn't lost if the step later fails.
    """

    run_id: str
    step_id: int
    agent: str
    step_name: str
    iteration: int
    _output: dict[str, Any] | None = field(default=None, repr=False)
    _usage: LLMUsage = field(default_factory=LLMUsage, repr=False)

    def set_output(self, output: dict[str, Any]) -> None:
        self._output = output

    def add_usage(self, usage: LLMUsage) -> None:
        self._usage = self._usage + usage
        accumulate_run_usage(
            self.run_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )


class StepRecorder:
    """Single instrumentation choke point for agent steps.

    Usage:
        recorder = StepRecorder()
        async with recorder.step(run_id, agent="profiler", step_name="analyze") as handle:
            profile = await do_work()
            handle.set_output({"row_count": profile.row_count})

    On exception, the step is recorded as 'failed' with the error message
    and the exception is re-raised (StepRecorder observes, it doesn't
    swallow) — the orchestrator decides retry/fallback policy.
    """

    @asynccontextmanager
    async def step(
        self,
        run_id: str,
        agent: str,
        step_name: str,
        iteration: int = 0,
        input_data: dict[str, Any] | None = None,
    ) -> AsyncIterator[StepHandle]:
        step_row = add_run_step(
            run_id, agent=agent, step_name=step_name, iteration=iteration, input_data=input_data
        )
        handle = StepHandle(
            run_id=run_id,
            step_id=step_row.step_id,
            agent=agent,
            step_name=step_name,
            iteration=iteration,
        )
        start = time.perf_counter()
        status = "completed"
        error_message: str | None = None
        try:
            yield handle
        except BaseException as e:
            # BaseException (not Exception): asyncio.CancelledError is a
            # BaseException subclass since Python 3.8, and Phase 3 cancels
            # in-flight steps via Task.cancel() -- without this, a cancelled
            # step would silently finish_run_step(status="completed").
            status = "cancelled" if isinstance(e, asyncio.CancelledError) else "failed"
            error_message = str(e) or type(e).__name__
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            finish_run_step(
                step_row.step_id,
                status=status,
                output_data=handle._output,
                error_message=error_message,
                input_tokens=handle._usage.input_tokens,
                output_tokens=handle._usage.output_tokens,
                cost_usd=handle._usage.cost_usd,
            )
            logger.info(
                f"{agent}.{step_name} {status}",
                extra={
                    "event": "agent_step",
                    "run_id": run_id,
                    "agent": agent,
                    "iteration": iteration,
                    "duration_ms": duration_ms,
                    "tokens": handle._usage.total_tokens,
                    "cost_usd": handle._usage.cost_usd,
                },
            )
            log_audit(
                session_id=run_id,  # audit_logs.session_id is a free-text correlation
                action="agent_step",  # id, not an FK -- run_id is the natural one here
                endpoint=f"run:{run_id}:{agent}.{step_name}",
                ip_address="internal",
                success=(status == "completed"),
                error_message=error_message,
                metadata=json.dumps(
                    {"agent": agent, "step_name": step_name, "iteration": iteration}
                ),
            )
