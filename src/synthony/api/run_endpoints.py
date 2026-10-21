"""REST endpoints for agentic-mode runs.

Phase 2 shipped the read-only history endpoints. Phase 3 adds the write
side (``POST /runs`` to start one, ``POST /runs/{run_id}/cancel``), backed
by a `RunManager` injected via `set_run_manager()` -- the same
dependency-injection shape `endpoints.py` uses for its own services
(`set_services`), so this module has no request-time dependency on how the
manager was constructed.
"""

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from synthony.api.database import (
    Run,
    count_runs,
    get_dataset,
    get_run,
    get_run_with_steps,
    list_run_steps,
    list_runs,
    log_audit,
)
from synthony.api.security import get_client_info
from synthony.core.schemas import RecommendationConstraints

if TYPE_CHECKING:
    # Deferred to avoid a circular import: synthony.api's package __init__
    # eagerly imports server.py -> this module, and synthony.agents.context
    # transitively imports synthony.api.database, which would re-enter the
    # still-initializing synthony.api package. RunConfig is only actually
    # constructed inside start_run(), which imports it locally.
    from synthony.agents.manager import RunManager

router = APIRouter()

_run_manager: "RunManager | None" = None


def set_run_manager(manager: "RunManager") -> None:
    global _run_manager
    _run_manager = manager


# ============================================================================
# Response Models
# ============================================================================


class RunSummary(BaseModel):
    run_id: str
    mode: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    current_iteration: int
    llm_provider: str | None
    llm_model: str | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


class RunListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    runs: list[RunSummary]


class RunStepDetail(BaseModel):
    step_id: int
    run_id: str
    iteration: int
    agent: str
    step_name: str
    status: str
    started_at: str
    finished_at: str | None
    duration_ms: int | None
    input: Any | None
    output: Any | None
    error_message: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _to_run_summary(run: Run) -> RunSummary:
    d = run.to_dict()
    return RunSummary(
        run_id=d["run_id"],
        mode=d["mode"],
        status=d["status"],
        created_at=d["created_at"],
        started_at=d["started_at"],
        finished_at=d["finished_at"],
        current_iteration=d["current_iteration"],
        llm_provider=d["llm_provider"],
        llm_model=d["llm_model"],
        total_input_tokens=d["total_input_tokens"],
        total_output_tokens=d["total_output_tokens"],
        total_cost_usd=d["total_cost_usd"],
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/runs", response_model=RunListResponse)
async def list_runs_endpoint(
    status: str | None = Query(None, description="Filter by run status"),
    limit: int = Query(100, ge=1, le=500, description="Max runs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List agentic-mode runs, most recent first."""
    runs = list_runs(status=status, limit=limit, offset=offset)
    total = count_runs(status=status)
    return RunListResponse(
        total=total,
        limit=limit,
        offset=offset,
        runs=[_to_run_summary(r) for r in runs],
    )


@router.get("/runs/{run_id}", response_model=dict[str, Any])
async def get_run_detail(run_id: str):
    """Retrieve a run with its full step history, config, and result."""
    detail = get_run_with_steps(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return detail


@router.get("/runs/{run_id}/steps", response_model=list[RunStepDetail])
async def list_run_steps_endpoint(run_id: str):
    """Retrieve just the step history for a run (no run-level metadata)."""
    if not get_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return [RunStepDetail(**s.to_dict()) for s in list_run_steps(run_id)]


# ============================================================================
# Write-side (Phase 3 advise mode, Phase 4 full mode)
# ============================================================================


class StartRunRequest(BaseModel):
    dataset_id: str
    mode: Literal["advise", "full"] = "advise"
    llm_provider: str | None = None
    llm_model: str | None = None
    constraints: dict[str, Any] | None = None
    max_reasoning_turns: int = Field(default=8, ge=1, le=50)
    max_run_tokens: int = Field(default=150_000, ge=1_000)
    step_timeout_s: int = Field(default=900, ge=10)


class StartRunResponse(BaseModel):
    run_id: str


@router.post("/runs", response_model=StartRunResponse, status_code=202)
async def start_run(request: StartRunRequest, req: Request):
    """Start an agentic run. Returns immediately with a `run_id`; poll
    `GET /runs/{run_id}` for status."""
    from synthony.agents.context import RunConfig  # local: see TYPE_CHECKING note above

    if _run_manager is None:
        raise HTTPException(status_code=503, detail="Run manager not initialized")

    dataset = get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

    if dataset.format == "parquet":
        df = pd.read_parquet(dataset.file_path)
    else:
        df = pd.read_csv(dataset.file_path)

    config = RunConfig(
        dataset_id=request.dataset_id,
        dataset_path=dataset.file_path,
        mode=request.mode,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        constraints=RecommendationConstraints(**(request.constraints or {})),
        max_reasoning_turns=request.max_reasoning_turns,
        max_run_tokens=request.max_run_tokens,
        step_timeout_s=request.step_timeout_s,
    )
    run_id = await _run_manager.start_run(config, df)

    ip_address, _ = get_client_info(req)
    log_audit(
        session_id=dataset.session_id,
        action="run_start",
        endpoint="/runs",
        ip_address=ip_address,
        success=True,
        metadata=f'{{"run_id": "{run_id}", "mode": "{request.mode}"}}',
    )
    return StartRunResponse(run_id=run_id)


@router.post("/runs/{run_id}/cancel", response_model=dict[str, Any])
async def cancel_run(run_id: str):
    """Request cancellation of a run. No-op (reports the current status) if
    the run is already terminal."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if run.status in ("completed", "failed", "cancelled"):
        return {
            "run_id": run_id,
            "status": run.status,
            "message": "run already terminal, no action taken",
        }

    ok = _run_manager.cancel(run_id) if _run_manager is not None else False
    return {"run_id": run_id, "cancelling": ok}
