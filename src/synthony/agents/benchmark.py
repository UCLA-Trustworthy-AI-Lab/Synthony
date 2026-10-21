"""BenchmarkAgent -- the BENCHMARKING stage of a full-mode run (Phase 4).

Compares the just-generated synthetic parquet against the run's original
dataset via the existing (unchanged) ``DataQualityBenchmark.compare``. One
instance handles exactly one attempt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from synthony.agents.base import BaseAgent
from synthony.benchmark.metrics import DataQualityBenchmark


class BenchmarkAgent(BaseAgent):
    name = "benchmark"

    def __init__(self, context, output_path: str):
        super().__init__(context)
        self.output_path = output_path

    async def run(self) -> dict[str, Any]:
        state = self.context.state
        async with state.recorder.step(
            state.run_id, agent=self.name, step_name="compare", iteration=state.iteration
        ) as handle:
            synthetic_df = await asyncio.to_thread(pd.read_parquet, self.output_path)
            result = await asyncio.to_thread(
                DataQualityBenchmark().compare, self.context.df, synthetic_df
            )
            result_dict = result.to_dict()
            handle.set_output({"overall_quality_score": result_dict["overall_quality_score"]})
        return result_dict
