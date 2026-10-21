"""SynthesizerAgent -- the TRAINING/GENERATING stage of a full-mode run
(Phase 4). Trains one candidate model in a subprocess-isolated runner.

One instance handles exactly one candidate for one iteration; the
orchestrator's full-mode loop constructs a fresh instance per attempt.
Training failure is reported as ``{"ok": False, ...}``, never raised --
per the Phase 4 plan, "a training failure = candidate failure (try next
ranked candidate), not run failure."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthony.agents.base import BaseAgent
from synthony.runners.base import SynthesisJob, SynthesizerRunner
from synthony.runners.registry import get_plugin_name
from synthony.runners.synthcity_runner import SynthcityRunner

OUTPUT_ROOT = Path("output/runs")


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def __init__(self, context, model_name: str, runner: SynthesizerRunner | None = None):
        super().__init__(context)
        self.model_name = model_name
        self.runner = runner or SynthcityRunner()

    async def run(self) -> dict[str, Any]:
        state = self.context.state
        iter_dir = OUTPUT_ROOT / state.run_id / f"iter{state.iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        train_csv_path = iter_dir / "train.csv"
        output_path = iter_dir / f"{self.model_name}.parquet"

        async with state.recorder.step(
            state.run_id,
            agent=self.name,
            step_name="train_and_generate",
            iteration=state.iteration,
            input_data={"model_name": self.model_name},
        ) as handle:
            plugin_name = get_plugin_name(self.model_name)
            if plugin_name is None:
                result_dict = {
                    "ok": False,
                    "error": f"'{self.model_name}' has no runner mapping",
                    "output_path": None,
                }
                handle.set_output(result_dict)
                return result_dict

            self.context.df.to_csv(train_csv_path, index=False)
            job = SynthesisJob(
                model_name=self.model_name,
                plugin_name=plugin_name,
                train_csv_path=str(train_csv_path),
                output_path=str(output_path),
                n_rows=len(self.context.df),
                seed=state.iteration,
                timeout_s=state.config.step_timeout_s,
            )
            result = await self.runner.train_and_generate(job, cancel_event=state.cancel_event)
            result_dict = {
                "ok": result.ok,
                "error": result.error,
                "model_name": self.model_name,
                "train_seconds": result.train_seconds,
                "generate_seconds": result.generate_seconds,
                "output_path": result.output_path,
            }
            handle.set_output(result_dict)
        return result_dict
