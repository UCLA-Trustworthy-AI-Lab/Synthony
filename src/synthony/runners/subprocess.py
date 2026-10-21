"""One-process-per-training-job execution model (Phase 4).

Rationale for a spawned ``multiprocessing.Process`` per job rather than a
``ProcessPoolExecutor``: real cancellation via ``process.terminate()``
(pool workers can't be interrupted mid-task), crash/OOM isolation from the
API server (a pool worker crash poisons the pool), and fresh state per job.
Uses the ``spawn`` start method explicitly (not fork-safety-dependent)
since torch/synthcity are only ever imported inside the child.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from multiprocessing.connection import Connection

from synthony.runners.base import SynthesisJob, SynthesisJobResult

_JOIN_TIMEOUT_S = 5


def _child_target(job: SynthesisJob, conn: Connection) -> None:
    """Runs inside the spawned child. Imports synthcity/pandas here only --
    never at module level -- so the parent process (API server) never pays
    synthcity's ~seconds-to-tens-of-seconds import cost and never loads
    torch/CUDA state it doesn't need."""
    try:
        import pandas as pd
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        df = pd.read_csv(job.train_csv_path)
        loader = GenericDataLoader(df)

        t0 = time.perf_counter()
        plugin = Plugins().get(job.plugin_name, random_state=job.seed, **job.params)
        plugin.fit(loader)
        train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        synth = plugin.generate(count=job.n_rows)
        synth_df = synth.dataframe()
        generate_seconds = time.perf_counter() - t0

        synth_df.to_parquet(job.output_path)
        conn.send(
            SynthesisJobResult(
                ok=True,
                error=None,
                train_seconds=train_seconds,
                generate_seconds=generate_seconds,
                output_path=job.output_path,
            )
        )
    except Exception as exc:  # the whole point: never let a bad plugin/dataset crash the parent
        conn.send(
            SynthesisJobResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                train_seconds=0.0,
                generate_seconds=0.0,
                output_path=None,
            )
        )
    finally:
        conn.close()


def _recv_or_none(conn: Connection) -> SynthesisJobResult | None:
    try:
        return conn.recv()
    except EOFError:
        return None


async def run_job_in_subprocess(
    job: SynthesisJob, cancel_event: asyncio.Event | None = None
) -> SynthesisJobResult:
    """Spawns one child process for ``job``, races its result against
    ``cancel_event`` and ``job.timeout_s``, and always tears the process
    down cleanly (terminate -> join -> kill) before returning."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_child_target, args=(job, child_conn), daemon=True)
    process.start()
    child_conn.close()  # only the child should hold the write end

    recv_task = asyncio.create_task(asyncio.to_thread(_recv_or_none, parent_conn))
    waiters = [recv_task]
    cancel_task: asyncio.Task | None = None
    if cancel_event is not None:
        cancel_task = asyncio.create_task(cancel_event.wait())
        waiters.append(cancel_task)

    done, _pending = await asyncio.wait(
        waiters, timeout=job.timeout_s, return_when=asyncio.FIRST_COMPLETED
    )

    if recv_task in done:
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        await asyncio.to_thread(process.join, _JOIN_TIMEOUT_S)
        result = recv_task.result()
        if result is not None:
            return result
        return SynthesisJobResult(
            ok=False,
            error="child process exited without producing a result",
            train_seconds=0.0,
            generate_seconds=0.0,
            output_path=None,
        )

    # Either cancelled or timed out -- tear the child down.
    reason = "cancelled by user" if (cancel_task is not None and cancel_task.done()) else (
        f"timed out after {job.timeout_s}s"
    )
    process.terminate()
    await asyncio.to_thread(process.join, _JOIN_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join, _JOIN_TIMEOUT_S)
    recv_task.cancel()
    parent_conn.close()
    return SynthesisJobResult(
        ok=False, error=reason, train_seconds=0.0, generate_seconds=0.0, output_path=None
    )
