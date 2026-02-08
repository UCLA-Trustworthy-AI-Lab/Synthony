"""
Recommendation Tools for MCP Server

Tools for hybrid rule-based + LLM recommendation engine.
"""

import json
from typing import Any, Dict, List, Optional

from mcp.types import Tool

from synthony.core.schemas import DatasetProfile, ColumnAnalysisResult
from synthony.recommender.engine import ModelRecommendationEngine, RecommendationResult


class RecommendationTools:
    """
    Recommendation tools for model selection and explanation.

    Tools:
    - rank_models_hybrid: Score models using rule-based + LLM decision logic
    - get_tie_breaker_logic: Resolve conflicts when models score within 5%
    - explain_recommendation_reasoning: Generate user-friendly explanation
    """

    def __init__(self, recommender: ModelRecommendationEngine):
        """Initialize recommendation tools with recommendation engine."""
        self.recommender = recommender

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "rank_models_hybrid",
            "rank_models_rule",
            "rank_models_llm",
            "get_tie_breaker_logic",
            "explain_recommendation_reasoning",
        ]

    def get_tool_definitions(self) -> List[Tool]:
        """Get MCP tool definitions."""
        return [
            Tool(
                name="rank_models_hybrid",
                description=(
                    "Rank synthesis models using hybrid rule-based + LLM approach. "
                    "Process: "
                    "1. Apply row-count filters "
                    "2. Detect stress factors (skew>2.0, cardinality>500, zipfian>0.05) "
                    "3. Score models 0-4 on capabilities (skew, cardinality, zipfian, privacy) "
                    "4. Apply tie-breaking rules if top models within 5% "
                    "5. Optionally use LLM for final ranking "
                    "Returns ranked list with primary recommendation and alternatives."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_profile": {
                            "type": "object",
                            "description": "Dataset stress profile from analyze_stress_profile tool"
                        },
                        "column_analysis": {
                            "type": "object",
                            "description": "Optional column-level analysis for enhanced recommendations"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["rule_based", "llm", "hybrid"],
                            "description": "Recommendation method to use",
                            "default": "hybrid"
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of alternative recommendations to return",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3
                        }
                    },
                    "required": ["dataset_profile"]
                }
            ),
            Tool(
                name="rank_models_rule",
                description=(
                    "Rank synthesis models using ONLY rule-based approach (pure Python, no LLM). "
                    "Process: "
                    "1. Apply row-count filters "
                    "2. Detect stress factors (skew>2.0, cardinality>500, zipfian>0.05) "
                    "3. Score models 0-4 on capabilities using model_capabilities.json "
                    "4. Apply tie-breaking rules if top models within 5% "
                    "Fast, deterministic, no API costs. Use when you need consistent, "
                    "explainable decisions without LLM overhead."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_profile": {
                            "type": "object",
                            "description": "Dataset stress profile from analyze_stress_profile tool"
                        },
                        "column_analysis": {
                            "type": "object",
                            "description": "Optional column-level analysis for enhanced recommendations"
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of alternative recommendations to return",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3
                        }
                    },
                    "required": ["dataset_profile"]
                }
            ),
            Tool(
                name="rank_models_llm",
                description=(
                    "Rank synthesis models using ONLY LLM-based approach (requires OpenAI API). "
                    "Process: "
                    "1. Build comprehensive prompt with dataset profile, column analysis, and SystemPrompt "
                    "2. Send to LLM for contextual reasoning about model suitability "
                    "3. Parse structured JSON response with model rankings and reasoning "
                    "4. Validate against model registry row-count limits "
                    "More nuanced than rule-based, captures complex dataset patterns. "
                    "Use when dataset has unusual characteristics that may need human-like reasoning."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_profile": {
                            "type": "object",
                            "description": "Dataset stress profile from analyze_stress_profile tool"
                        },
                        "column_analysis": {
                            "type": "object",
                            "description": "Optional column-level analysis for enhanced recommendations"
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of alternative recommendations to return",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3
                        }
                    },
                    "required": ["dataset_profile"]
                }
            ),
            Tool(
                name="get_tie_breaker_logic",
                description=(
                    "Get tie-breaking logic when top models score within 5%. "
                    "Rules: "
                    "- Rows < 500: Prefer ARF (best for small data) "
                    "- Rows > 50k + Hard Problem (Skew>2 & Card>500 & Zipf>0.05): Prefer TabDDPM (GReaT too slow) "
                    "- Otherwise: Prefer faster models (TVAE/ARF) over slower (Diffusion/LLMs) "
                    "Returns tie-breaking decision with reasoning."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tied_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model names with similar scores"
                        },
                        "dataset_profile": {
                            "type": "object",
                            "description": "Dataset stress profile for tie-breaking context"
                        }
                    },
                    "required": ["tied_models", "dataset_profile"]
                }
            ),
            Tool(
                name="explain_recommendation_reasoning",
                description=(
                    "Generate user-friendly explanation for model recommendation. "
                    "Translates technical decision (e.g., 'Selected ARF due to Small Data') "
                    "into clear narrative explaining: "
                    "- Why this model was chosen "
                    "- What dataset characteristics drove the decision "
                    "- Trade-offs vs alternatives "
                    "- Expected performance characteristics "
                    "Uses LLM to generate natural language explanation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "recommendation_result": {
                            "type": "object",
                            "description": "Result from rank_models_hybrid or recommend endpoint"
                        },
                        "dataset_profile": {
                            "type": "object",
                            "description": "Dataset stress profile for context"
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["brief", "detailed", "technical"],
                            "description": "Level of explanation detail",
                            "default": "detailed"
                        }
                    },
                    "required": ["recommendation_result", "dataset_profile"]
                }
            ),
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recommendation tool."""
        if name == "rank_models_hybrid":
            return await self._rank_models_hybrid(arguments)
        elif name == "rank_models_rule":
            return await self._rank_models_rule(arguments)
        elif name == "rank_models_llm":
            return await self._rank_models_llm(arguments)
        elif name == "get_tie_breaker_logic":
            return await self._get_tie_breaker_logic(arguments)
        elif name == "explain_recommendation_reasoning":
            return await self._explain_recommendation_reasoning(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _rank_models_hybrid(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank models using hybrid approach.

        Args:
            arguments: {
                "dataset_profile": dict,
                "column_analysis": Optional[dict],
                "method": str,
                "top_n": int
            }

        Returns:
            RecommendationResult as dict
        """
        # Parse dataset profile
        profile_dict = arguments["dataset_profile"]
        dataset_profile = DatasetProfile(**profile_dict)

        # Parse column analysis if provided
        column_analysis = None
        if arguments.get("column_analysis"):
            column_analysis = ColumnAnalysisResult(**arguments["column_analysis"])

        method = arguments.get("method", "hybrid")
        top_n = arguments.get("top_n", 3)

        # Run recommendation
        result = self.recommender.recommend(
            dataset_profile=dataset_profile,
            column_analysis=column_analysis,
            method=method,
            top_n=top_n
        )

        # Convert to dict with JSON-serializable types
        return result.model_dump(mode='json')

    async def _rank_models_rule(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank models using ONLY rule-based approach.

        Args:
            arguments: {
                "dataset_profile": dict,
                "column_analysis": Optional[dict],
                "top_n": int
            }

        Returns:
            RecommendationResult as dict
        """
        # Parse dataset profile
        profile_dict = arguments["dataset_profile"]
        dataset_profile = DatasetProfile(**profile_dict)

        # Parse column analysis if provided
        column_analysis = None
        if arguments.get("column_analysis"):
            column_analysis = ColumnAnalysisResult(**arguments["column_analysis"])

        top_n = arguments.get("top_n", 3)

        # Run recommendation with rule_based method
        result = self.recommender.recommend(
            dataset_profile=dataset_profile,
            column_analysis=column_analysis,
            method="rule_based",
            top_n=top_n
        )

        # Convert to dict with JSON-serializable types
        return result.model_dump(mode='json')

    async def _rank_models_llm(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank models using ONLY LLM approach.

        Args:
            arguments: {
                "dataset_profile": dict,
                "column_analysis": Optional[dict],
                "top_n": int
            }

        Returns:
            RecommendationResult as dict
        """
        # Parse dataset profile
        profile_dict = arguments["dataset_profile"]
        dataset_profile = DatasetProfile(**profile_dict)

        # Parse column analysis if provided
        column_analysis = None
        if arguments.get("column_analysis"):
            column_analysis = ColumnAnalysisResult(**arguments["column_analysis"])

        top_n = arguments.get("top_n", 3)

        # Run recommendation with llm method
        result = self.recommender.recommend(
            dataset_profile=dataset_profile,
            column_analysis=column_analysis,
            method="llm",
            top_n=top_n
        )

        # Convert to dict with JSON-serializable types
        return result.model_dump(mode='json')

    async def _get_tie_breaker_logic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tie-breaking logic.

        Args:
            arguments: {
                "tied_models": List[str],
                "dataset_profile": dict
            }

        Returns:
            {
                "winner": str,
                "reasoning": str,
                "rule_applied": str
            }
        """
        tied_models = arguments["tied_models"]
        profile_dict = arguments["dataset_profile"]

        dataset_profile = DatasetProfile(**profile_dict)
        row_count = dataset_profile.row_count

        # Get stress factors
        skewness_detected = any(
            s.skewness_coefficient > 2.0
            for s in dataset_profile.skewness.skewness_scores
        )
        cardinality_detected = any(
            c.unique_count > 500
            for c in dataset_profile.cardinality.cardinality_scores
        )
        zipfian_detected = dataset_profile.cardinality.zipfian_ratio > 0.05

        is_hard_problem = skewness_detected and cardinality_detected and zipfian_detected

        # Apply tie-breaking rules
        if row_count < 500:
            # Small data: Prefer ARF
            winner = "ARF" if "ARF" in tied_models else tied_models[0]
            reasoning = f"Dataset has only {row_count} rows (< 500). ARF is best suited for small data to prevent overfitting."
            rule_applied = "small_data_rule"

        elif row_count > 50000 and is_hard_problem:
            # Large + Hard Problem: Prefer TabDDPM (GReaT too slow)
            winner = "TabDDPM" if "TabDDPM" in tied_models else tied_models[0]
            reasoning = f"Dataset has {row_count} rows with severe skew, high cardinality, and zipfian distribution. TabDDPM is preferred over GReaT due to speed constraints on large datasets."
            rule_applied = "large_hard_problem_rule"

        else:
            # Default: quality-focused models
            quality_priority = ["TabDDPM", "TabSyn", "AutoDiff"]
            winner = next((m for m in quality_priority if m in tied_models), tied_models[0])
            reasoning = f"Models are closely matched. Defaulting to {winner} based on overall quality."
            rule_applied = "quality_default_rule"

        return {
            "winner": winner,
            "reasoning": reasoning,
            "rule_applied": rule_applied,
            "tied_models": tied_models,
            "dataset_characteristics": {
                "row_count": row_count,
                "is_hard_problem": is_hard_problem,
                "skewness_detected": skewness_detected,
                "cardinality_detected": cardinality_detected,
                "zipfian_detected": zipfian_detected
            }
        }

    async def _explain_recommendation_reasoning(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for recommendation.

        Args:
            arguments: {
                "recommendation_result": dict,
                "dataset_profile": dict,
                "detail_level": str
            }

        Returns:
            {
                "explanation": str,
                "key_factors": List[str],
                "model_strengths": str,
                "alternatives_comparison": str
            }
        """
        recommendation = arguments["recommendation_result"]
        profile_dict = arguments["dataset_profile"]
        detail_level = arguments.get("detail_level", "detailed")

        dataset_profile = DatasetProfile(**profile_dict)
        primary_model = recommendation["primary_model"]

        # Extract key factors
        key_factors = []
        row_count = dataset_profile.row_count

        # Check stress factors
        if any(s.skewness_coefficient > 2.0 for s in dataset_profile.skewness.skewness_scores):
            key_factors.append("Severe skewness detected (> 2.0)")

        if any(c.unique_count > 500 for c in dataset_profile.cardinality.cardinality_scores):
            key_factors.append("High cardinality (> 500 unique values)")

        if dataset_profile.cardinality.zipfian_ratio > 0.05:
            key_factors.append(f"Zipfian distribution (ratio: {dataset_profile.cardinality.zipfian_ratio:.3f})")

        if row_count < 500:
            key_factors.append(f"Small dataset ({row_count} rows)")
        elif row_count > 50000:
            key_factors.append(f"Large dataset ({row_count} rows)")

        # Generate explanation based on detail level
        if detail_level == "brief":
            explanation = f"Recommended {primary_model['name']} with confidence {primary_model['confidence']:.2f} based on dataset characteristics."
        elif detail_level == "detailed":
            explanation = f"""
Recommended Model: {primary_model['name']} ({primary_model['full_name']})
Confidence Score: {primary_model['confidence']:.2f}

Key Dataset Characteristics:
{chr(10).join(f"- {factor}" for factor in key_factors)}

Reasoning:
{primary_model['reasoning']}

Expected Performance:
- Quality: {primary_model['performance']['quality']}
- Speed: {primary_model['performance']['speed']}
"""
        else:  # technical
            explanation = f"""
TECHNICAL RECOMMENDATION REPORT
================================

Primary Model: {primary_model['name']}
Type: {primary_model['type']}
Confidence: {primary_model['confidence']:.2f}

Dataset Profile:
- Rows: {row_count}
- Columns: {dataset_profile.column_count}
- Skewness Scores: {[s.skewness_coefficient for s in dataset_profile.skewness.skewness_scores]}
- Cardinality Scores: {[c.unique_count for c in dataset_profile.cardinality.cardinality_scores]}
- Zipfian Ratio: {dataset_profile.cardinality.zipfian_ratio:.3f}

Capability Scores (0-4 scale):
{json.dumps(primary_model['capabilities'], indent=2)}

Constraints:
{json.dumps(primary_model['constraints'], indent=2)}

Decision Reasoning:
{primary_model['reasoning']}
"""

        # Get alternatives comparison
        alternatives = recommendation.get("alternatives", [])
        if alternatives:
            alternatives_comparison = "Alternative Models:\n" + "\n".join(
                f"- {alt['name']}: Confidence {alt['confidence']:.2f}"
                for alt in alternatives
            )
        else:
            alternatives_comparison = "No significant alternatives found."

        return {
            "explanation": explanation.strip(),
            "key_factors": key_factors,
            "model_strengths": "\n".join(primary_model.get("strengths", [])),
            "alternatives_comparison": alternatives_comparison
        }
