"""
Data Tools for MCP Server

Tools for dataset discovery and loading from a configured data directory.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from mcp.types import Tool, ToolAnnotations

from synthony.utils.constants import DATA_DIR


class DataTools:
    """
    Data discovery and loading tools.

    Tools:
    - list_datasets: List available datasets in the configured data directory
    - load_dataset: Load a dataset by name and return metadata + preview
    """

    def __init__(self, data_dir: Path = None):
        """Initialize data tools with data directory."""
        self.data_dir = data_dir or DATA_DIR

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "synthony_list_datasets",
            "synthony_load_dataset",
        ]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="synthony_list_datasets",
                description=(
                    "List available datasets in the configured data directory. "
                    "Returns dataset names, formats (CSV/Parquet), and file sizes. "
                    "Use this tool first to discover what datasets are available "
                    "before loading or analyzing them."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format_filter": {
                            "type": "string",
                            "enum": ["csv", "parquet", "all"],
                            "description": "Filter by file format (default: all)",
                            "default": "all"
                        }
                    }
                },
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
            Tool(
                name="synthony_load_dataset",
                description=(
                    "Load a dataset by name from the configured data directory. "
                    "Returns dataset metadata (shape, columns, dtypes, memory usage) "
                    "and a preview of the first rows. Use this tool after synthony_list_datasets "
                    "to inspect a specific dataset before running analysis."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the dataset (e.g., 'Bean', 'Titanic')"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "parquet"],
                            "description": "File format to load (auto-detected if only one exists)"
                        },
                        "preview_rows": {
                            "type": "integer",
                            "description": "Number of preview rows to return (default: 5)",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 5
                        }
                    },
                    "required": ["dataset_name"]
                },
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                )
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data tool."""
        if name == "synthony_list_datasets":
            return await self._list_datasets(arguments)
        elif name == "synthony_load_dataset":
            return await self._load_dataset(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _resolve_data_dir(self) -> Path:
        """Resolve the data directory to an absolute path."""
        data_dir = Path(os.environ.get("SYNTHONY_DATA_DIR", str(self.data_dir)))
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return data_dir

    async def _list_datasets(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        List available datasets in the data directory.

        Returns:
            {
                "data_dir": str,
                "datasets": [
                    {"name": str, "format": str, "size_bytes": int, "path": str}
                ],
                "total_count": int
            }
        """
        format_filter = arguments.get("format_filter", "all")
        data_dir = self._resolve_data_dir()

        extensions = {".csv", ".parquet"}
        if format_filter == "csv":
            extensions = {".csv"}
        elif format_filter == "parquet":
            extensions = {".parquet"}

        datasets = []
        seen_names = {}

        for file_path in sorted(data_dir.iterdir()):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                name = file_path.stem
                fmt = file_path.suffix.lstrip(".")
                size = file_path.stat().st_size

                if name not in seen_names:
                    seen_names[name] = []
                seen_names[name].append({
                    "name": name,
                    "format": fmt,
                    "size_bytes": size,
                    "path": str(file_path),
                })

        for name in sorted(seen_names.keys()):
            datasets.extend(seen_names[name])

        return {
            "data_dir": str(data_dir),
            "datasets": datasets,
            "total_count": len(datasets),
        }

    def _find_dataset_file(self, dataset_name: str, fmt: str = None) -> Path:
        """Resolve dataset name to file path."""
        data_dir = self._resolve_data_dir()

        if fmt:
            target = data_dir / f"{dataset_name}.{fmt}"
            if target.exists():
                return target
            raise FileNotFoundError(
                f"Dataset '{dataset_name}.{fmt}' not found in {data_dir}"
            )

        # Auto-detect: prefer csv, then parquet
        for ext in (".csv", ".parquet"):
            target = data_dir / f"{dataset_name}{ext}"
            if target.exists():
                return target

        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir} "
            f"(tried .csv and .parquet)"
        )

    async def _load_dataset(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a dataset by name and return metadata + preview.

        Returns:
            {
                "dataset_name": str,
                "file_path": str,
                "format": str,
                "rows": int,
                "columns": int,
                "column_names": list,
                "dtypes": dict,
                "memory_usage_mb": float,
                "preview": list[dict]
            }
        """
        dataset_name = arguments["dataset_name"]
        fmt = arguments.get("format")
        preview_rows = arguments.get("preview_rows", 5)

        file_path = self._find_dataset_file(dataset_name, fmt)
        file_format = file_path.suffix.lstrip(".")

        if file_format == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        preview = df.head(preview_rows).to_dict(orient="records")
        # Convert non-serializable values to strings
        for row in preview:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    row[key] = str(value)

        return {
            "dataset_name": dataset_name,
            "file_path": str(file_path),
            "format": file_format,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": dtypes,
            "memory_usage_mb": round(memory_mb, 2),
            "preview": preview,
        }
