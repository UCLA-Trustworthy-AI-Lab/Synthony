"""Real (not faked) synthcity training smoke test for ``SynthcityRunner``.

Excluded from default CI: gated on ``SYNTHONY_RUN_SLOW_TESTS=1`` (in
addition to ``@pytest.mark.slow``, since none of this repo's CI pytest
invocations currently pass ``-m "not slow"`` -- the env-var gate is what
actually keeps this out of every default run without needing a CI YAML
change). Run explicitly with:

    SYNTHONY_RUN_SLOW_TESTS=1 pytest tests/functional/test_synthcity_runner_e2e.py -v

Uses ARF, not the plan's originally-suggested ``bayesian_network`` --
verified 2026-07-09 that ``bayesian_network`` fails to import in this
environment (``cannot import name '_lazywhere' from 'scipy._lib._util'``,
a scipy/pgmpy version incompatibility, not a Synthony bug). ARF is
CPU-only, fast (~2s fit+generate once synthcity is warm), and was the
model manually verified end-to-end while building Phase 4.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("synthcity")

from synthony.runners.base import SynthesisJob  # noqa: E402
from synthony.runners.synthcity_runner import SynthcityRunner  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("SYNTHONY_RUN_SLOW_TESTS"),
    reason="real synthcity training; set SYNTHONY_RUN_SLOW_TESTS=1 to run",
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_arf_trains_and_generates_real_synthetic_data(tmp_path):
    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "num_0": range(200),
            "num_1": [x * 0.5 for x in range(200)],
            "cat_0": ["a", "b", "c", "d"] * 50,
        }
    )
    df.to_csv(train_csv, index=False)

    job = SynthesisJob(
        model_name="ARF",
        plugin_name="arf",
        train_csv_path=str(train_csv),
        output_path=str(tmp_path / "synth.parquet"),
        n_rows=100,
        timeout_s=120,
    )
    runner = SynthcityRunner()
    result = await runner.train_and_generate(job)

    assert result.ok, result.error
    assert result.train_seconds > 0
    assert result.generate_seconds > 0
    synth_df = pd.read_parquet(result.output_path)
    assert len(synth_df) == 100
    assert set(synth_df.columns) == set(df.columns)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cancellation_terminates_a_real_child_process(tmp_path):
    import asyncio

    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"num_0": range(200), "num_1": range(200)}).to_csv(train_csv, index=False)

    job = SynthesisJob(
        model_name="CTGAN",
        plugin_name="ctgan",
        train_csv_path=str(train_csv),
        output_path=str(tmp_path / "synth.parquet"),
        n_rows=100,
        params={"n_iter": 5000},  # slow enough to reliably cancel mid-train
        timeout_s=120,
    )
    runner = SynthcityRunner()
    cancel_event = asyncio.Event()

    async def cancel_soon():
        await asyncio.sleep(2)
        cancel_event.set()

    result, _ = await asyncio.gather(
        runner.train_and_generate(job, cancel_event=cancel_event), cancel_soon()
    )

    assert result.ok is False
    assert "cancelled" in result.error
