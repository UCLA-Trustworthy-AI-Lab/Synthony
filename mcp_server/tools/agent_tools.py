"""
Agent Tools for MCP Server (Agentic Mode Phase 5)

Tools for starting, monitoring, and cancelling agentic runs through the
same ``RunManager``/``Orchestrator`` (Phases 3-4) the REST API uses --
``RunManager`` is transport-agnostic asyncio, so it works identically in
this long-lived stdio server. The one difference from the REST API: a
dataset here is resolved by *name* from the configured data directory
(``SYNTHONY_DATA_DIR``, same as ``DataTools``), not a database-registered
upload, since MCP clients don't go through ``POST /analyze`` first.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from mcp.types import Tool

from synthony.agents.context import RunConfig
from synthony.agents.manager import RunManager
from synthony.api.database import count_runs, get_run, get_run_with_steps
from synthony.api.database import list_runs as db_list_runs
from synthony.core.schemas import RecommendationConstraints
from synthony.utils.constants import DATA_DIR

_RUN_STATUSES = (
    "pending",
    "profiling",
    "reasoning",
    "training",
    "benchmarking",
    "iterating",
    "completed",
    "failed",
    "cancelled",
)


class AgentTools:
    """
    Agentic-mode run tools.

    Tools:
    - run_agentic_synthesis: start an agentic run on a named dataset
    - get_run_status: poll a run's status, config, result, and step history
    - list_runs: list runs, optionally filtered by status
    - cancel_run: cancel a run started by this same MCP session
    """

    def __init__(self, run_manager: RunManager, data_dir: Path = None):
        self.run_manager = run_manager
        self.data_dir = data_dir or DATA_DIR
        # One ID per MCP server process, stamped as `runs.owner` on every run
        # this tool starts -- lets cancel_run refuse a run started by a
        # different session (the REST API, or another MCP client) instead of
        # silently cancelling something it doesn't own.
        self._owner_id = f"mcp:{uuid4()}"

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return ["run_agentic_synthesis", "get_run_status", "list_runs", "cancel_run"]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="run_agentic_synthesis",
                description=(
                    "Start an agentic synthesis run on a dataset already available in the "
                    "configured data directory (see list_datasets). Returns immediately with "
                    "a run_id -- poll get_run_status to follow progress. "
                    "mode='advise' (default) just recommends a model: PENDING -> PROFILING -> "
                    "REASONING -> COMPLETED. mode='full' actually trains, generates, and "
                    "benchmarks the recommended model, iterating up to max_iterations times "
                    "if quality_threshold isn't met: ... -> TRAINING -> BENCHMARKING -> "
                    "(below threshold) ITERATING -> REASONING -> ... -> COMPLETED."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": (
                                "Name of the dataset (e.g., 'Bean', 'Titanic'), as returned "
                                "by list_datasets."
                            ),
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "parquet"],
                            "description": "File format (auto-detected if only one exists).",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["advise", "full"],
                            "default": "advise",
                            "description": (
                                "'advise' recommends only; 'full' trains + generates + "
                                "benchmarks the recommendation."
                            ),
                        },
                        "quality_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.75,
                            "description": "mode='full' only: benchmark score (0-1) to accept.",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 3,
                            "description": "mode='full' only: max reasoning/train/benchmark cycles.",
                        },
                        "llm_provider": {
                            "type": "string",
                            "description": "Override the default LLM provider (e.g. 'openai', 'anthropic').",
                        },
                        "llm_model": {
                            "type": "string",
                            "description": "Override the default LLM model.",
                        },
                        "constraints": {
                            "type": "object",
                            "description": (
                                "Optional recommendation constraints, e.g. "
                                "{\"cpu_only\": true, \"strict_dp\": true}."
                            ),
                        },
                    },
                    "required": ["dataset_name"],
                },
            ),
            Tool(
                name="get_run_status",
                description=(
                    "Get a run's current status, config, result (once available), and full "
                    "step history (per-step agent, timing, tokens/cost, output). Poll this "
                    "after run_agentic_synthesis until status is completed/failed/cancelled."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"run_id": {"type": "string", "description": "Run ID to look up."}},
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="list_runs",
                description="List runs, most recent first, optionally filtered by status.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": list(_RUN_STATUSES),
                            "description": "Filter to a single status; omit for all runs.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 50,
                        },
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                },
            ),
            Tool(
                name="cancel_run",
                description=(
                    "Request cancellation of a run started by run_agentic_synthesis in this "
                    "same MCP session. Returns a clear error, not a silent no-op, if the run "
                    "belongs to a different session (the REST API or another MCP client) -- "
                    "use get_run_status first if you're unsure whose run it is."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"run_id": {"type": "string", "description": "Run ID to cancel."}},
                    "required": ["run_id"],
                },
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent tool."""
        if name == "run_agentic_synthesis":
            return await self._run_agentic_synthesis(arguments)
        elif name == "get_run_status":
            return await self._get_run_status(arguments)
        elif name == "list_runs":
            return await self._list_runs(arguments)
        elif name == "cancel_run":
            return await self._cancel_run(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _resolve_data_dir(self) -> Path:
        """Resolve the data directory to an absolute path. Mirrors
        ``DataTools._resolve_data_dir`` -- kept local rather than shared
        since each MCP tool module in this codebase is self-contained."""
        data_dir = Path(os.environ.get("SYNTHONY_DATA_DIR", str(self.data_dir)))
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return data_dir

    def _find_dataset_file(self, dataset_name: str, fmt: Optional[str] = None) -> Path:
        """Resolve dataset name to file path. Mirrors ``DataTools._find_dataset_file``."""
        data_dir = self._resolve_data_dir()

        if fmt:
            target = data_dir / f"{dataset_name}.{fmt}"
            if target.exists():
                return target
            raise FileNotFoundError(f"Dataset '{dataset_name}.{fmt}' not found in {data_dir}")

        for ext in (".csv", ".parquet"):
            target = data_dir / f"{dataset_name}{ext}"
            if target.exists():
                return target

        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir} (tried .csv and .parquet)"
        )

    async def _run_agentic_synthesis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start an agentic run.

        Returns:
            {"run_id": str, "status": "pending", "mode": str}
        """
        dataset_name = arguments["dataset_name"]
        file_path = self._find_dataset_file(dataset_name, arguments.get("format"))

        if file_path.suffix.lstrip(".") == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        config = RunConfig(
            dataset_id=dataset_name,
            dataset_path=str(file_path),
            mode=arguments.get("mode", "advise"),
            quality_threshold=arguments.get("quality_threshold", 0.75),
            max_iterations=arguments.get("max_iterations", 3),
            llm_provider=arguments.get("llm_provider"),
            llm_model=arguments.get("llm_model"),
            constraints=RecommendationConstraints(**(arguments.get("constraints") or {})),
        )
        run_id = await self.run_manager.start_run(config, df, owner=self._owner_id)
        return {"run_id": run_id, "status": "pending", "mode": config.mode}

    async def _get_run_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the same shape as ``GET /runs/{run_id}`` (run detail + steps)."""
        run_id = arguments["run_id"]
        detail = get_run_with_steps(run_id)
        if detail is None:
            raise ValueError(f"Run '{run_id}' not found")
        return detail

    async def _list_runs(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
            {"runs": [...], "total": int, "limit": int, "offset": int}
        """
        status = arguments.get("status")
        limit = arguments.get("limit", 50)
        offset = arguments.get("offset", 0)

        runs = db_list_runs(status=status, limit=limit, offset=offset)
        total = count_runs(status=status)
        return {
            "runs": [r.to_dict() for r in runs],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _cancel_run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
            {"run_id": str, "cancelling": bool} or a terminal-run no-op message.

        Raises ValueError (surfaced as an MCP tool error, not a silent
        no-op) if the run doesn't exist or isn't owned by this session.
        """
        run_id = arguments["run_id"]
        run = get_run(run_id)
        if run is None:
            raise ValueError(f"Run '{run_id}' not found")
        if run.owner != self._owner_id:
            raise ValueError(
                f"Run '{run_id}' is not owned by this MCP session (owner={run.owner!r}); "
                "refusing to cancel a foreign run."
            )
        if run.status in ("completed", "failed", "cancelled"):
            return {
                "run_id": run_id,
                "status": run.status,
                "message": "run already terminal, no action taken",
            }

        ok = self.run_manager.cancel(run_id)
        return {"run_id": run_id, "cancelling": ok}
