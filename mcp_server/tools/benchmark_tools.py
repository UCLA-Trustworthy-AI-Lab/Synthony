"""
Benchmark Tools for MCP Server

Tools for comparing original and synthetic datasets using quality metrics.
"""

from typing import Any, Dict, List

from mcp.types import Tool

from synthony.benchmark.metrics import DataQualityBenchmark
from synthony.core.loaders import DataLoader


class BenchmarkTools:
    """
    Benchmark tools for data quality comparison.

    Tools:
    - benchmark_compare: Compare original and synthetic datasets
    """

    def __init__(self):
        """Initialize benchmark tools."""
        self.benchmark = DataQualityBenchmark()

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return ["benchmark_compare"]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="benchmark_compare",
                description=(
                    "Compare original and synthetic datasets to measure data quality. "
                    "Computes fidelity (mean/std/correlation preservation), "
                    "utility (column correlation, distribution similarity), "
                    "privacy (DCR, min distance ratio, duplicate rate, privacy score), "
                    "and per-column KL/JS divergence. "
                    "Optionally computes differential privacy metrics (DCR ratio, "
                    "membership inference, attribute inference) when an evaluation "
                    "(holdout) dataset is provided."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "original_path": {
                            "type": "string",
                            "description": "Path to the original dataset (CSV or Parquet)"
                        },
                        "synthetic_path": {
                            "type": "string",
                            "description": "Path to the synthetic dataset (CSV or Parquet)"
                        },
                        "evaluation_path": {
                            "type": "string",
                            "description": "Optional path to holdout dataset for differential privacy metrics"
                        },
                    },
                    "required": ["original_path", "synthetic_path"]
                }
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a benchmark tool."""
        if name == "benchmark_compare":
            return await self._benchmark_compare(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _benchmark_compare(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare original and synthetic datasets.

        Args:
            arguments: {
                "original_path": str,
                "synthetic_path": str,
                "evaluation_path": Optional[str]
            }

        Returns:
            Complete benchmark result dict with fidelity, utility, privacy,
            column metrics, and optional differential privacy metrics.
        """
        original_path = arguments["original_path"]
        synthetic_path = arguments["synthetic_path"]
        evaluation_path = arguments.get("evaluation_path")

        # Load datasets
        orig_df = DataLoader.load(original_path, validate=True)
        synth_df = DataLoader.load(synthetic_path, validate=True)

        # Run benchmark comparison
        result = self.benchmark.compare(orig_df, synth_df)
        result_dict = result.to_dict()

        # Compute differential privacy metrics if evaluation dataset provided
        if evaluation_path:
            eval_df = DataLoader.load(evaluation_path, validate=True)
            dp_metrics = self.benchmark.calculate_differential_privacy(
                training_data=orig_df,
                synthetic_data=synth_df,
                evaluation_data=eval_df,
            )
            result_dict["differential_privacy"] = {
                "dcr_train": dp_metrics.dcr_train,
                "dcr_eval": dp_metrics.dcr_eval,
                "dcr_ratio": dp_metrics.dcr_ratio,
                "membership_advantage": dp_metrics.membership_advantage,
                "membership_auc": dp_metrics.membership_auc,
                "attribute_inference_risk": dp_metrics.attribute_inference_risk,
                "empirical_dp_score": dp_metrics.empirical_dp_score,
                "estimated_epsilon": dp_metrics.estimated_epsilon,
            }

        return result_dict
