"""ProfilerAgent -- the PROFILING stage of an advise-mode run."""

from __future__ import annotations

import asyncio
from typing import Any

from synthony.agents.base import BaseAgent
from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer


class ProfilerAgent(BaseAgent):
    """Runs ``StochasticDataAnalyzer`` and ``ColumnAnalyzer`` inside one
    recorded step. Both are computed here (not split into two steps) since
    ``RecommenderAgent``'s tools need both the dataset-level profile and the
    per-column analysis, and neither is expensive enough to warrant its own
    step row.
    """

    name = "profiler"

    async def run(self) -> dict[str, Any]:
        state = self.context.state
        async with state.recorder.step(
            state.run_id, agent=self.name, step_name="analyze"
        ) as handle:
            profile = await asyncio.to_thread(StochasticDataAnalyzer().analyze, self.context.df)
            state.dataset_profile = profile

            column_analysis = await asyncio.to_thread(
                ColumnAnalyzer().analyze, self.context.df, profile
            )
            state.column_analysis = column_analysis

            summary = {
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "stress_factors": profile.stress_factors.model_dump(),
            }
            handle.set_output(summary)
        return summary
