"""
Benchmark Data Resources for MCP Server

Provides access to historical benchmark results and thresholds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkData:
    """
    Benchmark data resource provider.

    Resources:
    - benchmarks://thresholds: Stress detector thresholds
    - benchmarks://results/{model}/{dataset_type}: Historical validation results
    """

    def __init__(self):
        """Initialize benchmark data provider."""
        # Define standard thresholds
        self.thresholds = {
            "skewness": {
                "severe_skew": 2.0,
                "moderate_skew": 1.0,
                "description": "Fisher-Pearson skewness coefficient thresholds"
            },
            "cardinality": {
                "high_cardinality": 500,
                "very_high_cardinality": 1000,
                "description": "Unique value count thresholds"
            },
            "zipfian": {
                "zipfian_ratio": 0.05,
                "high_concentration": 0.80,
                "description": "Zipfian distribution detection (top 20% concentration)"
            },
            "data_size": {
                "small_data": 500,
                "large_data": 50000,
                "description": "Row count thresholds for model selection"
            }
        }

        # Path to benchmark results directory
        self.benchmark_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"

    def get_resource_definitions(self) -> List[Dict[str, str]]:
        """Get MCP resource definitions."""
        return [
            {
                "uri": "benchmarks://thresholds",
                "name": "Benchmark Thresholds",
                "description": "Stress detector thresholds (Skew>2.0, Zipfian>0.05, Cardinality>500)",
                "mimeType": "application/json"
            },
            {
                "uri": "benchmarks://results/{model}/{dataset_type}",
                "name": "Benchmark Results",
                "description": "Historical WD/TVD validation results (parameterized by model and dataset_type)",
                "mimeType": "application/json"
            }
        ]

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a benchmark data resource."""
        if uri == "benchmarks://thresholds":
            return await self._get_thresholds()
        elif uri.startswith("benchmarks://results/"):
            # Parse model and dataset_type from URI
            parts = uri.replace("benchmarks://results/", "").split("/")
            if len(parts) != 2:
                raise ValueError(f"Invalid benchmark results URI: {uri}. Expected format: benchmarks://results/{{model}}/{{dataset_type}}")

            model_name, dataset_type = parts
            return await self._get_benchmark_results(model_name, dataset_type)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _get_thresholds(self) -> Dict[str, Any]:
        """Get stress detector thresholds."""
        return {
            "uri": "benchmarks://thresholds",
            "type": "thresholds",
            "thresholds": self.thresholds,
            "description": "Standard thresholds for detecting hard problems in datasets"
        }

    async def _get_benchmark_results(self, model_name: str, dataset_type: str) -> Dict[str, Any]:
        """Get benchmark results for a specific model and dataset type."""
        # Check if benchmark results exist
        benchmark_file = self.benchmark_dir / f"{model_name}_{dataset_type}.json"

        if not benchmark_file.exists():
            # Return placeholder for missing benchmarks
            return {
                "uri": f"benchmarks://results/{model_name}/{dataset_type}",
                "type": "benchmark_results",
                "model_name": model_name,
                "dataset_type": dataset_type,
                "status": "not_available",
                "message": f"No benchmark results found for {model_name} on {dataset_type} dataset. Run validation to generate results."
            }

        # Load existing results
        try:
            with open(benchmark_file, 'r') as f:
                results = json.load(f)

            return {
                "uri": f"benchmarks://results/{model_name}/{dataset_type}",
                "type": "benchmark_results",
                "model_name": model_name,
                "dataset_type": dataset_type,
                "status": "available",
                "results": results,
                "description": f"Historical validation results for {model_name} on {dataset_type}"
            }

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse benchmark results: {e}")
