"""Shared job/result types and the ``SynthesizerRunner`` interface (Phase 4).

A ``SynthesisJob`` is passed by *path*, never by pickled DataFrame -- the
training data is read from ``train_csv_path`` inside whichever process
actually trains the model (a spawned child, for ``SynthcityRunner``), and
the result is written to ``output_path`` (parquet) rather than shipped back
in memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesisJob:
    model_name: str
    plugin_name: str
    train_csv_path: str
    output_path: str
    n_rows: int
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    timeout_s: int = 900


@dataclass
class SynthesisJobResult:
    ok: bool
    error: str | None
    train_seconds: float
    generate_seconds: float
    output_path: str | None


class SynthesizerRunner(ABC):
    """Trains + generates one job. Implementations own their own process/
    isolation model; the caller only sees ``SynthesisJobResult``."""

    @abstractmethod
    def supported_models(self) -> frozenset[str]:
        """Model names (as used in ``config/model_capabilities.json``) this
        runner can execute."""

    @abstractmethod
    async def train_and_generate(
        self, job: SynthesisJob, cancel_event: Any = None
    ) -> SynthesisJobResult:
        """Train ``job.model_name`` on ``job.train_csv_path`` and generate
        ``job.n_rows`` synthetic rows to ``job.output_path``.

        If ``cancel_event`` (an ``asyncio.Event``) is set while this is
        in flight, the implementation should abort promptly and return a
        non-ok result rather than raise -- cancellation is a normal
        outcome, not an error.
        """
