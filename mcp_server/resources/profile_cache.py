"""
Profile Cache Resources for MCP Server

Manages cached dataset profiles from previous analyses.
"""

import json
from typing import Any, Dict, List

from synthony.api.database import (
    get_analysis,
    get_analysis_by_dataset,
    get_dataset,
)
from synthony.core.schemas import DatasetProfile, ColumnAnalysisResult


class ProfileCache:
    """
    Profile cache resource provider.

    Resources:
    - datasets://profiles/{dataset_id}: Cached analysis results for a dataset
    """

    def get_resource_definitions(self) -> List[Dict[str, str]]:
        """Get MCP resource definitions."""
        return [
            {
                "uri": "datasets://profiles/{dataset_id}",
                "name": "Dataset Profile Cache",
                "description": "Cached dataset profiles from previous analyses (parameterized by dataset_id)",
                "mimeType": "application/json"
            }
        ]

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a profile cache resource."""
        if uri.startswith("datasets://profiles/"):
            dataset_id = uri.replace("datasets://profiles/", "")
            return await self._get_cached_profile(dataset_id)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _get_cached_profile(self, dataset_id: str) -> Dict[str, Any]:
        """Get cached profile for a dataset."""
        # Get analysis from database
        analysis = get_analysis_by_dataset(dataset_id)

        if not analysis:
            raise ValueError(
                f"No cached profile found for dataset_id '{dataset_id}'. "
                "Please run analyze_stress_profile tool first."
            )

        # Get dataset metadata
        dataset = get_dataset(dataset_id)

        # Parse stored JSON
        try:
            profile_dict = json.loads(analysis.profile_json)
            column_dict = json.loads(analysis.column_analysis_json) if analysis.column_analysis_json else None
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode cached profile: {e}")

        return {
            "uri": f"datasets://profiles/{dataset_id}",
            "type": "cached_profile",
            "dataset_id": dataset_id,
            "analysis_id": analysis.analysis_id,
            "dataset_profile": profile_dict,
            "column_analysis": column_dict,
            "metadata": {
                "filename": dataset.filename if dataset else None,
                "file_size": dataset.file_size if dataset else None,
                "format": dataset.format if dataset else None,
                "analyzed_at": analysis.created_at.isoformat(),
            }
        }
