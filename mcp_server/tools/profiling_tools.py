"""
Profiling Tools for MCP Server

Tools for Package 1 integration: Data profiling and stress analysis.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from mcp.types import Tool

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.core.schemas import DatasetProfile, ColumnAnalysisResult
from synthony.benchmark.generators import BenchmarkDatasetGenerator
from synthony.utils.constants import DATA_DIR


class ProfilingTools:
    """
    Profiling tools for data analysis and stress profiling.

    Tools:
    - analyze_stress_profile: Extract skewness, cardinality, zipfian ratio from tabular data
    - generate_benchmark_dataset: Create synthetic control datasets for validation
    """

    def __init__(self, analyzer: StochasticDataAnalyzer, column_analyzer: ColumnAnalyzer):
        """Initialize profiling tools with analyzers."""
        self.analyzer = analyzer
        self.column_analyzer = column_analyzer
        self.benchmark_generator = BenchmarkDatasetGenerator()

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "analyze_stress_profile",
            "generate_benchmark_dataset",
        ]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="analyze_stress_profile",
                description=(
                    "Analyze a tabular dataset (CSV or Parquet) and extract stress profile. "
                    "Returns dataset-level metrics (skewness, cardinality, zipfian ratio, correlation) "
                    "and column-level analysis (per-column stress factors, difficulty scores). "
                    "Use this tool when you need to understand dataset characteristics before "
                    "recommending a synthesis model. "
                    "Provide either 'dataset_name' (resolves from configured data directory) "
                    "or 'data_path' (absolute file path)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of dataset in configured data directory (e.g., 'Bean', 'Titanic')"
                        },
                        "data_path": {
                            "type": "string",
                            "description": "Absolute path to CSV or Parquet file (alternative to dataset_name)"
                        },
                        "dataset_id": {
                            "type": "string",
                            "description": "Optional identifier for this dataset (for caching)"
                        }
                    }
                }
            ),
            Tool(
                name="generate_benchmark_dataset",
                description=(
                    "Generate synthetic control datasets for model validation. "
                    "Creates test datasets with known stress characteristics: "
                    "- 'long_tail': LogNormal distribution with severe skew (>2.0) "
                    "- 'needle_haystack': Zipfian distribution with 1000+ categories "
                    "- 'small_data_trap': 200 rows multivariate dataset "
                    "Use this tool when you need to validate model recommendations empirically."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_type": {
                            "type": "string",
                            "enum": ["long_tail", "needle_haystack", "small_data_trap"],
                            "description": "Type of benchmark dataset to generate"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save generated CSV file"
                        },
                        "num_rows": {
                            "type": "integer",
                            "description": "Number of rows to generate (overrides defaults)",
                            "minimum": 100
                        }
                    },
                    "required": ["dataset_type", "output_path"]
                }
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a profiling tool."""
        if name == "analyze_stress_profile":
            return await self._analyze_stress_profile(arguments)
        elif name == "generate_benchmark_dataset":
            return await self._generate_benchmark_dataset(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _resolve_data_dir(self) -> Path:
        """Resolve the data directory to an absolute path."""
        data_dir = Path(os.environ.get("SYNTHONY_DATA_DIR", str(DATA_DIR)))
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        return data_dir

    def _resolve_dataset_path(self, dataset_name: str) -> Path:
        """Resolve a dataset name to its file path in the data directory."""
        data_dir = self._resolve_data_dir()
        for ext in (".csv", ".parquet"):
            target = data_dir / f"{dataset_name}{ext}"
            if target.exists():
                return target
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir} "
            f"(tried .csv and .parquet)"
        )

    async def _analyze_stress_profile(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dataset stress profile.

        Args:
            arguments: {
                "dataset_name": Optional[str],
                "data_path": Optional[str],
                "dataset_id": Optional[str]
            }

        Returns:
            {
                "dataset_id": str,
                "dataset_profile": dict,
                "column_analysis": dict,
                "message": str
            }
        """
        dataset_name = arguments.get("dataset_name")
        data_path = arguments.get("data_path")
        dataset_id = arguments.get("dataset_id")

        # Resolve file path from dataset_name or data_path
        if dataset_name:
            file_path = self._resolve_dataset_path(dataset_name)
        elif data_path:
            file_path = Path(data_path)
        else:
            raise ValueError("Either 'dataset_name' or 'data_path' must be provided")

        # Validate file path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not (file_path.suffix == ".csv" or file_path.suffix == ".parquet"):
            raise ValueError("Only CSV and Parquet files are supported")

        # Generate dataset_id if not provided
        if not dataset_id:
            dataset_id = file_path.stem

        # Load DataFrame
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        # Run dataset-level analysis
        dataset_profile = self.analyzer.analyze(df)

        # Run column-level analysis
        column_analysis = self.column_analyzer.analyze(df, dataset_profile)

        # Convert to dict with JSON-serializable types (exclude correlation matrix)
        profile_dict = dataset_profile.model_dump(
            mode='json',
            exclude={'correlation': {'correlation_matrix'}}
        )
        column_dict = column_analysis.model_dump(mode='json')

        return {
            "dataset_id": dataset_id,
            "dataset_profile": profile_dict,
            "column_analysis": column_dict,
            "message": f"Analysis completed: {dataset_profile.row_count} rows × {dataset_profile.column_count} columns"
        }

    async def _generate_benchmark_dataset(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark dataset.

        Args:
            arguments: {
                "dataset_type": str,
                "output_path": str,
                "num_rows": Optional[int]
            }

        Returns:
            {
                "dataset_type": str,
                "output_path": str,
                "num_rows": int,
                "num_columns": int,
                "expected_stress_factors": dict
            }
        """
        dataset_type = arguments["dataset_type"]
        output_path = arguments["output_path"]
        num_rows = arguments.get("num_rows")

        # Generate dataset based on type
        if dataset_type == "long_tail":
            df, metadata = self.benchmark_generator.generate_long_tail(num_rows=num_rows)
        elif dataset_type == "needle_haystack":
            df, metadata = self.benchmark_generator.generate_needle_haystack(num_rows=num_rows)
        elif dataset_type == "small_data_trap":
            df, metadata = self.benchmark_generator.generate_small_data_trap(num_rows=num_rows)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Save to output path
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return {
            "dataset_type": dataset_type,
            "output_path": output_path,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "expected_stress_factors": metadata
        }
