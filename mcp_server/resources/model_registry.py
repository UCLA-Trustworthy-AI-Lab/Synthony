"""
Model Registry Resources for MCP Server

Exposes model capabilities and registry information as MCP resources.
"""

from typing import Any, Dict, List

from synthony.recommender.engine import ModelRecommendationEngine


class ModelRegistry:
    """
    Model registry resource provider.

    Resources:
    - models://registry: Full model catalog with capability scores
    - models://model/{model_name}: Individual model details
    """

    def __init__(self, recommender: ModelRecommendationEngine):
        """Initialize model registry with recommendation engine."""
        self.recommender = recommender

    def get_resource_definitions(self) -> List[Dict[str, str]]:
        """Get MCP resource definitions."""
        resources = [
            {
                "uri": "models://registry",
                "name": "Model Registry",
                "description": "Complete catalog of all synthesis models with capability scores (0-4 scale)",
                "mimeType": "application/json"
            }
        ]

        # Add individual model resources
        models = self.recommender.model_capabilities.get("models", {})
        for model_name in models.keys():
            resources.append({
                "uri": f"models://model/{model_name}",
                "name": f"Model: {model_name}",
                "description": f"Detailed information for {model_name} synthesis model",
                "mimeType": "application/json"
            })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a model registry resource."""
        if uri == "models://registry":
            return await self._get_full_registry()
        elif uri.startswith("models://model/"):
            model_name = uri.replace("models://model/", "")
            return await self._get_model_details(model_name)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _get_full_registry(self) -> Dict[str, Any]:
        """Get complete model registry."""
        return {
            "uri": "models://registry",
            "type": "model_registry",
            "models": self.recommender.model_capabilities.get("models", {}),
            "model_ranking": self.recommender.model_capabilities.get("model_ranking_by_capability", {}),
            "total_models": len(self.recommender.model_capabilities.get("models", {})),
            "description": "Complete catalog of synthesis models with 0-4 capability scores"
        }

    async def _get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information for a specific model."""
        models = self.recommender.model_capabilities.get("models", {})

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            )

        model_info = models[model_name]

        return {
            "uri": f"models://model/{model_name}",
            "type": "model_details",
            "model_name": model_info["name"],
            "full_name": model_info["full_name"],
            "model_type": model_info["type"],
            "capabilities": model_info["capabilities"],
            "constraints": model_info["constraints"],
            "performance": model_info["performance"],
            "description": model_info["description"],
            "strengths": model_info["strengths"],
            "limitations": model_info["limitations"],
        }
