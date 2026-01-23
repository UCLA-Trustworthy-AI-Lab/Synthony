"""
Model Tools for MCP Server

Tools for Package 2 shadow interface: Model capabilities and constraints.
"""

import json
from typing import Any, Dict, List, Optional

from mcp.types import Tool

from synthony.recommender.engine import ModelRecommendationEngine


class ModelTools:
    """
    Model tools for querying model capabilities and constraints.

    Tools:
    - check_model_constraints: Validate constraints (cpu_only, strict_dp, data size limits)
    - get_model_info: Get detailed information about a specific model
    - list_models: List all available models with optional filters
    """

    def __init__(self, recommender: ModelRecommendationEngine):
        """Initialize model tools with recommendation engine."""
        self.recommender = recommender

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "check_model_constraints",
            "get_model_info",
            "list_models",
        ]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="check_model_constraints",
                description=(
                    "Check which models satisfy given constraints. "
                    "Applies hard filters based on: "
                    "- cpu_only: Exclude GPU-dependent models (TabDDPM, TabSyn, GReaT) "
                    "- strict_dp: Keep only differential privacy models (PATE-CTGAN, DPCART, AIM) "
                    "- data_size: Filter based on row count limits "
                    "Returns list of compatible models and reasons for exclusions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cpu_only": {
                            "type": "boolean",
                            "description": "Only include CPU-compatible models (excludes GPU models)",
                            "default": False
                        },
                        "strict_dp": {
                            "type": "boolean",
                            "description": "Only include models with strict differential privacy",
                            "default": False
                        },
                        "row_count": {
                            "type": "integer",
                            "description": "Number of rows in dataset (for size-based filtering)",
                            "minimum": 1
                        },
                        "min_data_size": {
                            "type": "integer",
                            "description": "Minimum data size supported by model",
                            "minimum": 1
                        },
                        "max_data_size": {
                            "type": "integer",
                            "description": "Maximum data size supported by model"
                        }
                    }
                }
            ),
            Tool(
                name="get_model_info",
                description=(
                    "Get detailed information about a specific synthesis model. "
                    "Returns full specification including: "
                    "- Capabilities (skew handling, cardinality, zipfian, privacy) "
                    "- Constraints (CPU/GPU, data size limits, privacy requirements) "
                    "- Performance characteristics (speed, quality) "
                    "- Strengths and limitations "
                    "Use this tool when you need to understand a model's characteristics."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to query (e.g., 'GReaT', 'TabDDPM', 'ARF')"
                        }
                    },
                    "required": ["model_name"]
                }
            ),
            Tool(
                name="list_models",
                description=(
                    "List all available synthesis models with optional filtering. "
                    "Supports filtering by: "
                    "- model_type: GAN, VAE, Diffusion, Tree-based, Statistical, LLM "
                    "- cpu_only: CPU-compatible models only "
                    "- requires_dp: Differential privacy support "
                    "Returns model registry with capability scores and rankings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Filter by model type",
                            "enum": ["GAN", "VAE", "Diffusion", "Tree-based", "Statistical", "LLM"]
                        },
                        "cpu_only": {
                            "type": "boolean",
                            "description": "Only include CPU-compatible models"
                        },
                        "requires_dp": {
                            "type": "boolean",
                            "description": "Only include models with differential privacy support"
                        }
                    }
                }
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a model tool."""
        if name == "check_model_constraints":
            return await self._check_model_constraints(arguments)
        elif name == "get_model_info":
            return await self._get_model_info(arguments)
        elif name == "list_models":
            return await self._list_models(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _check_model_constraints(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check model constraints.

        Args:
            arguments: {
                "cpu_only": Optional[bool],
                "strict_dp": Optional[bool],
                "row_count": Optional[int],
                "min_data_size": Optional[int],
                "max_data_size": Optional[int]
            }

        Returns:
            {
                "compatible_models": List[str],
                "excluded_models": Dict[str, str],
                "constraints_applied": Dict[str, Any]
            }
        """
        cpu_only = arguments.get("cpu_only", False)
        strict_dp = arguments.get("strict_dp", False)
        row_count = arguments.get("row_count")
        min_data_size = arguments.get("min_data_size")
        max_data_size = arguments.get("max_data_size")

        models = self.recommender.model_capabilities.get("models", {})
        compatible_models = []
        excluded_models = {}

        for model_name, model_info in models.items():
            constraints = model_info.get("constraints", {})
            capabilities = model_info.get("capabilities", {})

            # Check CPU-only constraint
            if cpu_only and not constraints.get("cpu_only_compatible", False):
                excluded_models[model_name] = "Requires GPU"
                continue

            # Check strict DP constraint
            if strict_dp and capabilities.get("privacy_dp", 0) < 3:
                excluded_models[model_name] = "Does not support strict differential privacy"
                continue

            # Check data size constraints
            if row_count:
                model_min = constraints.get("min_data_size", 0)
                model_max = constraints.get("max_data_size", float('inf'))

                if row_count < model_min:
                    excluded_models[model_name] = f"Requires at least {model_min} rows"
                    continue

                if row_count > model_max:
                    excluded_models[model_name] = f"Maximum {model_max} rows supported"
                    continue

            compatible_models.append(model_name)

        return {
            "compatible_models": compatible_models,
            "excluded_models": excluded_models,
            "constraints_applied": {
                "cpu_only": cpu_only,
                "strict_dp": strict_dp,
                "row_count": row_count,
                "min_data_size": min_data_size,
                "max_data_size": max_data_size
            }
        }

    async def _get_model_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model information.

        Args:
            arguments: {
                "model_name": str
            }

        Returns:
            Full model specification including capabilities, constraints, performance, etc.
        """
        model_name = arguments["model_name"]
        models = self.recommender.model_capabilities.get("models", {})

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            )

        model_info = models[model_name]

        return {
            "model_name": model_info["name"],
            "full_name": model_info["full_name"],
            "type": model_info["type"],
            "capabilities": model_info["capabilities"],
            "constraints": model_info["constraints"],
            "performance": model_info["performance"],
            "description": model_info["description"],
            "strengths": model_info["strengths"],
            "limitations": model_info["limitations"],
        }

    async def _list_models(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        List models with optional filtering.

        Args:
            arguments: {
                "model_type": Optional[str],
                "cpu_only": Optional[bool],
                "requires_dp": Optional[bool]
            }

        Returns:
            {
                "total_models": int,
                "filtered_models": int,
                "models": Dict[str, Any],
                "model_ranking": Dict[str, Any]
            }
        """
        model_type = arguments.get("model_type")
        cpu_only = arguments.get("cpu_only")
        requires_dp = arguments.get("requires_dp")

        models = self.recommender.model_capabilities.get("models", {})
        filtered_models = {}

        for name, info in models.items():
            # Type filter
            if model_type and info.get("type", "").lower() != model_type.lower():
                continue

            # CPU-only filter
            if cpu_only and not info.get("constraints", {}).get("cpu_only_compatible", False):
                continue

            # Differential privacy filter
            if requires_dp and info.get("capabilities", {}).get("privacy_dp", 0) == 0:
                continue

            filtered_models[name] = info

        return {
            "total_models": len(models),
            "filtered_models": len(filtered_models),
            "models": filtered_models,
            "model_ranking": self.recommender.model_capabilities.get("model_ranking_by_capability", {}),
        }
