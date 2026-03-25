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
    - check_model_constraints: Validate data size limits for models
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
                    "Check which models are compatible with a given dataset size. "
                    "Applies row-count filters based on model min/max data size limits. "
                    "Returns list of compatible models and reasons for exclusions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "row_count": {
                            "type": "integer",
                            "description": "Number of rows in dataset (for size-based filtering)",
                            "minimum": 1
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
                    "Supports filtering by model_type: GAN, VAE, Diffusion, Tree-based, Statistical, LLM. "
                    "Returns model registry with capability scores and rankings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Filter by model type",
                            "enum": ["GAN", "VAE", "Diffusion", "Tree-based", "Statistical", "LLM"]
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
        Check model compatibility based on row count.

        Args:
            arguments: {
                "row_count": Optional[int]
            }

        Returns:
            {
                "compatible_models": List[str],
                "excluded_models": Dict[str, str],
                "filters_applied": Dict[str, Any]
            }
        """
        row_count = arguments.get("row_count")

        models = self.recommender.model_capabilities.get("models", {})
        compatible_models = []
        excluded_models = {}

        for model_name, model_info in models.items():
            constraints = model_info.get("constraints", {})

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
            "filters_applied": {
                "row_count": row_count
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
                "model_type": Optional[str]
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

        models = self.recommender.model_capabilities.get("models", {})
        filtered_models = {}

        for name, info in models.items():
            # Type filter
            if model_type and info.get("type", "").lower() != model_type.lower():
                continue

            filtered_models[name] = info

        return {
            "total_models": len(models),
            "filtered_models": len(filtered_models),
            "models": filtered_models,
            "model_ranking": self.recommender.model_capabilities.get("model_ranking_by_capability", {}),
        }
