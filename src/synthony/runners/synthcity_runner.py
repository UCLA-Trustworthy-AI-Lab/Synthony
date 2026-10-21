"""``SynthesizerRunner`` backed by synthcity, executed in a spawned child
process (see ``runners/subprocess.py``)."""

from __future__ import annotations

import asyncio

from synthony.runners.base import SynthesisJob, SynthesisJobResult, SynthesizerRunner
from synthony.runners.registry import executable_models, get_plugin_name
from synthony.runners.subprocess import run_job_in_subprocess


class SynthcityRunner(SynthesizerRunner):
    def supported_models(self) -> frozenset[str]:
        return executable_models()

    async def train_and_generate(
        self, job: SynthesisJob, cancel_event: asyncio.Event | None = None
    ) -> SynthesisJobResult:
        return await run_job_in_subprocess(job, cancel_event=cancel_event)

    @staticmethod
    def build_job(
        *,
        model_name: str,
        train_csv_path: str,
        output_path: str,
        n_rows: int,
        params: dict | None = None,
        seed: int = 0,
        timeout_s: int = 900,
    ) -> SynthesisJob:
        plugin_name = get_plugin_name(model_name)
        if plugin_name is None:
            raise ValueError(f"'{model_name}' has no synthcity plugin mapping")
        return SynthesisJob(
            model_name=model_name,
            plugin_name=plugin_name,
            train_csv_path=train_csv_path,
            output_path=output_path,
            n_rows=n_rows,
            params=params or {},
            seed=seed,
            timeout_s=timeout_s,
        )
