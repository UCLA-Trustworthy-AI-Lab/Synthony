"""
Hybrid recommendation engine: Rule-based + LLM-based recommendations.

Implements both:
1. Rule-based scoring from model_capabilities.json (v3 with Hard Problem detection)
2. LLM-based inference using OpenAI API with SystemPrompt
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from synthony.core.schemas import ColumnAnalysisResult, DatasetProfile


@dataclass
class EngineConfig:
    """Configuration for the recommendation engine.
    
    All thresholds are configurable to allow tuning based on
    empirical benchmarks and user feedback.
    """
    
    # Hard Problem thresholds
    skew_threshold: float = 2.0          # |skewness| above this = severe
    cardinality_threshold: int = 500     # Unique count above this = high
    zipfian_threshold: float = 0.05      # Top 20% ratio above this = Zipfian
    
    # Size thresholds
    small_data_threshold: int = 1000     # Rows below this = small data (tie-break to ARF)
    large_data_threshold: int = 50000    # Rows above this = large data (GReaT too slow)
    
    # Tie-breaking
    tie_threshold_percent: float = 5.0   # Scores within this % are considered tied
    
    # Speed preference
    prefer_speed_default: bool = False   # Default value for prefer_speed constraint
    
    # Hard Problem fallback priority (order matters)
    hard_problem_fallback: List[str] = field(
        default_factory=lambda: ["TabSyn", "ARF", "TabTree"]
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "skew_threshold": self.skew_threshold,
            "cardinality_threshold": self.cardinality_threshold,
            "zipfian_threshold": self.zipfian_threshold,
            "small_data_threshold": self.small_data_threshold,
            "large_data_threshold": self.large_data_threshold,
            "tie_threshold_percent": self.tie_threshold_percent,
            "prefer_speed_default": self.prefer_speed_default,
            "hard_problem_fallback": self.hard_problem_fallback,
        }


# Default configuration
DEFAULT_ENGINE_CONFIG = EngineConfig()


class ModelRecommendation(BaseModel):
    """Single model recommendation with reasoning."""
    
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix in field names

    model_name: str = Field(description="Name of the recommended model")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    capability_match: Dict[str, int] = Field(
        description="How well model capabilities match requirements (0-4 scale)"
    )
    reasoning: List[str] = Field(
        description="List of reasons why this model was chosen"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about limitations or constraints"
    )
    model_info: Dict[str, Any] = Field(
        description="Full model information from capabilities registry"
    )


class RecommendationResult(BaseModel):
    """Complete recommendation result with primary, alternatives, and reasoning."""

    dataset_id: str = Field(description="ID of the analyzed dataset")
    
    # Recommendation method used
    method: str = Field(description="Method used: 'rule_based', 'llm', or 'hybrid'")
    
    # Primary recommendation
    recommended_model: ModelRecommendation = Field(
        description="Primary recommended model"
    )
    
    # Alternatives
    alternative_models: List[ModelRecommendation] = Field(
        default_factory=list,
        description="Alternative models if primary is not available (sorted by score)"
    )
    
    # LLM reasoning (if LLM method used)
    llm_reasoning: Optional[str] = Field(
        default=None,
        description="Full reasoning from LLM (if LLM method was used)"
    )
    
    # Constraints applied
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="User constraints that filtered the model list"
    )
    
    # Dataset difficulty summary
    difficulty_summary: Dict[str, Any] = Field(
        description="Summary of dataset difficulty factors"
    )
    
    # Models excluded
    excluded_models: Dict[str, str] = Field(
        default_factory=dict,
        description="Models excluded and reasons why"
    )


class ModelRecommendationEngine:
    """
    Hybrid recommendation engine supporting both rule-based and LLM-based approaches.
    
    Modes:
    - rule_based: Fast, deterministic scoring using model_capabilities.json
      - v2: Includes Hard Problem detection and safety checks
      - v3: fine-tuned with synthetic data
    - llm: Uses OpenAI API with SystemPrompt for contextual reasoning
    - h-brid: Combines both approaches (rule-based ranking + LLM reasoning)
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        capabilities_path: Optional[Path] = None,
        system_prompt_path: Optional[Path] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        openai_base_url: Optional[str] = None,
    ):
        """Initialize recommendation engine.

        Args:
            config: Engine configuration (thresholds, fallbacks). Uses defaults if None.
            capabilities_path: Path to model_capabilities.json
            system_prompt_path: Path to SystemPrompt.md
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            openai_model: OpenAI model to use (default: gpt-4o)
            openai_base_url: Optional base URL for OpenAI-compatible API (e.g., VLLM)
        """
        # Configuration
        self.config = config or DEFAULT_ENGINE_CONFIG
        
        # Load capabilities registry
        if capabilities_path is None:
            capabilities_path = Path(__file__).parent / "model_capabilities.json"
        with open(capabilities_path, "r") as f:
            self.registry = json.load(f)
        self.models = self.registry["models"]

        # Load system prompt (for LLM mode)
        if system_prompt_path is None:
            # Try environment variable first
            import os
            env_path = os.getenv("SYNTHONY_SYSTEM_PROMPT")
            if env_path:
                system_prompt_path = Path(env_path)
            else:
                # Default to SystemPrompt_v3.md in docs/
                system_prompt_path = (
                    Path(__file__).parent.parent.parent.parent
                    / "docs"
                    / "SystemPrompt_v3.md"
                )

        self.system_prompt_path = system_prompt_path
        if system_prompt_path.exists():
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
                self.system_prompt_loaded = True
                # Print confirmation
                print(f"✓ SystemPrompt loaded from: {system_prompt_path}")
                print(f"  Size: {len(self.system_prompt)} characters")
        else:
            self.system_prompt = None
            self.system_prompt_loaded = False
            print(f"⚠ SystemPrompt not found at: {system_prompt_path}")
            print(f"  LLM mode will use default prompt")

        # OpenAI configuration
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url

        # Try to import OpenAI (optional dependency)
        try:
            from openai import OpenAI

            # Only create client if API key is provided
            if self.openai_api_key:
                # Support custom base URL (e.g., for VLLM)
                client_kwargs = {"api_key": self.openai_api_key}
                if self.openai_base_url:
                    client_kwargs["base_url"] = self.openai_base_url

                self.openai_client = OpenAI(**client_kwargs)
                self.llm_available = True
            else:
                self.openai_client = None
                self.llm_available = False
        except ImportError:
            self.openai_client = None
            self.llm_available = False
        except Exception as e:
            # Handle API key validation errors gracefully
            self.openai_client = None
            self.llm_available = False

    @property
    def model_capabilities(self) -> Dict[str, Any]:
        """Expose the model capabilities registry."""
        return self.registry

    def recommend(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_n: int = 3,
        method: str = "rule_based",
    ) -> RecommendationResult:
        """Generate model recommendations.

        Args:
            dataset_profile: Dataset profile from StochasticDataAnalyzer
            column_analysis: Optional column analysis from ColumnAnalyzer
            constraints: Optional user constraints (cpu_only, strict_dp, etc.)
            top_n: Number of alternative recommendations
            method: Recommendation method - 'rule_based', 'llm', or 'hybrid'

        Returns:
            RecommendationResult with recommendations
        """
        # Convert RecommendationConstraints Pydantic model to dict if needed
        from synthony.core.schemas import RecommendationConstraints
        if isinstance(constraints, RecommendationConstraints):
            constraints = constraints.model_dump()
        
        constraints = constraints or {}
        constraints["dataset_rows"] = dataset_profile.row_count

        if method == "llm":
            return self._recommend_llm(
                dataset_profile, column_analysis, constraints, top_n
            )
        elif method == "hybrid":
            return self._recommend_hybrid(
                dataset_profile, column_analysis, constraints, top_n
            )
        else:  # rule_based (default)
            return self._recommend_rule_based(
                dataset_profile, column_analysis, constraints, top_n
            )

    def _recommend_rule_based(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        top_n: int,
    ) -> RecommendationResult:
        """Rule-based recommendation with v2 Hard Problem detection.
        
        Flow:
        1. Apply hard filters (cpu_only, strict_dp, size constraints)
        2. Check for Hard Problem (skew AND cardinality AND zipfian)
           - If yes: Route to GReaT with safety checks
        3. Calculate weighted capability scores
        4. Apply tie-breaking rules
        5. Build and return recommendation
        """
        # Apply prefer_speed default if not specified
        if "prefer_speed" not in constraints:
            constraints["prefer_speed"] = self.config.prefer_speed_default
        
        # Step 1: Apply hard filters
        eligible_models, excluded_models = self._apply_hard_filters(constraints)
        
        if not eligible_models:
            raise ValueError(
                f"No eligible models after applying constraints. "
                f"Excluded: {excluded_models}"
            )

        # Step 2: Check for Hard Problem
        is_hard, hard_details = self._is_hard_problem(dataset_profile)
        
        if is_hard:
            # Hard Problem Path: Route to specialized models
            hard_recommended = self._handle_hard_problem(
                dataset_profile, eligible_models, excluded_models
            )
            if hard_recommended:
                # Build Hard Problem result
                return self._build_hard_problem_result(
                    hard_recommended,
                    dataset_profile,
                    column_analysis,
                    constraints,
                    eligible_models,
                    excluded_models,
                    hard_details,
                    top_n,
                )

        # Step 3: Normal Flow - Calculate required capabilities
        required_capabilities = self._calculate_required_capabilities(
            dataset_profile, column_analysis
        )

        # Step 4: Score models
        scored_models = self._score_models(eligible_models, required_capabilities)

        # Step 5: Sort by score
        sorted_models = sorted(
            scored_models, key=lambda x: x["total_score"], reverse=True
        )

        # Step 6: Apply tie-breaking
        primary_name = self._apply_tie_breaking(
            sorted_models, dataset_profile, constraints
        )
        
        # Re-order sorted_models to put primary first
        primary_model = next(
            m for m in sorted_models if m["model_name"] == primary_name
        )
        other_models = [
            m for m in sorted_models if m["model_name"] != primary_name
        ]

        # Step 7: Build recommendations
        primary = self._build_recommendation(
            primary_model, required_capabilities, constraints
        )

        alternatives = [
            self._build_recommendation(model, required_capabilities, constraints)
            for model in other_models[:top_n]
        ]

        # Step 8: Build difficulty summary
        difficulty_summary = {
            "max_column_difficulty": column_analysis.max_column_difficulty
            if column_analysis
            else 0,
            "stress_factors": dataset_profile.stress_factors.model_dump(),
            "required_capabilities": required_capabilities,
            "dataset_size": {
                "rows": dataset_profile.row_count,
                "columns": dataset_profile.column_count,
            },
            "is_hard_problem": is_hard,
            "hard_problem_details": hard_details,
        }

        return RecommendationResult(
            dataset_id=dataset_profile.dataset_id,
            method="rule_based_v2",
            recommended_model=primary,
            alternative_models=alternatives,
            constraints=constraints,
            difficulty_summary=difficulty_summary,
            excluded_models=excluded_models,
        )

    # =========================================================================
    # v2 Hard Problem Methods
    # =========================================================================

    def _is_hard_problem(
        self, dataset_profile: DatasetProfile
    ) -> Tuple[bool, Dict[str, bool]]:
        """Detect if dataset exhibits the 'Hard Problem' pattern.
        
        A Hard Problem is defined as ALL THREE conditions being true:
        1. Severe Skew (max |skewness| > threshold)
        2. High Cardinality (any column > threshold unique values)
        3. Zipfian Distribution (top 20% categories > threshold of data)
        
        Returns:
            Tuple of (is_hard_problem, details_dict)
        """
        stress = dataset_profile.stress_factors
        
        # Check Zipfian using the configured threshold (0.05 per v2 spec)
        zipfian_detected = False
        if dataset_profile.zipfian and dataset_profile.zipfian.top_20_percent_ratio:
            zipfian_detected = (
                dataset_profile.zipfian.top_20_percent_ratio > self.config.zipfian_threshold
            )
        
        details = {
            "severe_skew": stress.severe_skew,
            "high_cardinality": stress.high_cardinality,
            "zipfian": zipfian_detected,
        }
        
        is_hard = all([
            details["severe_skew"],
            details["high_cardinality"],
            details["zipfian"],
        ])
        
        return is_hard, details

    def _handle_hard_problem(
        self,
        dataset_profile: DatasetProfile,
        eligible_models: List[str],
        excluded_models: Dict[str, str],
    ) -> Optional[str]:
        """Handle Hard Problem routing with safety checks.
        
        Decision Logic:
        1. If rows > large_data_threshold → Recommend TabDDPM (GReaT too slow)
        2. If GReaT in candidate pool → Recommend GReaT
        3. Else → Recommend from fallback list (TabSyn > ARF > TabTree)
        
        Returns:
            Model name to recommend, or None if no suitable model found
        """
        row_count = dataset_profile.row_count
        
        # Large data: GReaT too slow
        if row_count > self.config.large_data_threshold:
            # Prefer TabDDPM for large hard problems
            if "TabDDPM" in eligible_models:
                return "TabDDPM"
            elif "TabSyn" in eligible_models:
                return "TabSyn"
        
        # Check if GReaT is available (best for hard problems)
        if "GReaT" in eligible_models:
            return "GReaT"
        
        # Fallback: TabSyn > ARF > TabTree (priority order from config)
        for backup in self.config.hard_problem_fallback:
            if backup in eligible_models:
                return backup
        
        return None  # No suitable model found

    def _apply_tie_breaking(
        self,
        sorted_models: List[Dict[str, Any]],
        dataset_profile: DatasetProfile,
        constraints: Dict[str, Any],
    ) -> str:
        """Apply tie-breaking rules when top models are within threshold.
        
        Priority Order:
        1. Small Data (<small_data_threshold rows) → ARF (prevents overfitting)
        2. Prefer Speed → TVAE or CTGAN
        3. Otherwise → TabDDPM (quality-focused)
        
        Returns:
            Model name after tie-breaking
        """
        if len(sorted_models) < 2:
            return sorted_models[0]["model_name"]
        
        top_score = sorted_models[0]["total_score"]
        second_score = sorted_models[1]["total_score"]
        
        # Check if within tie threshold
        threshold_fraction = self.config.tie_threshold_percent / 100.0
        score_diff = (top_score - second_score) / max(top_score, 0.01)
        
        if score_diff > threshold_fraction:
            return sorted_models[0]["model_name"]  # Clear winner
        
        # Tie detected - apply rules
        row_count = dataset_profile.row_count
        prefer_speed = constraints.get("prefer_speed", self.config.prefer_speed_default)
        
        candidates = [m["model_name"] for m in sorted_models[:3]]
        
        # Rule 1: Small data → ARF (prevents overfitting)
        if row_count < self.config.small_data_threshold:
            if "ARF" in candidates:
                return "ARF"
            if "GaussianCopula" in candidates:
                return "GaussianCopula"
        
        # Rule 2: Speed preference
        if prefer_speed:
            for fast_model in ["TVAE", "CTGAN", "ARF", "GaussianCopula"]:
                if fast_model in candidates:
                    return fast_model
        
        # Rule 3: Default to quality (diffusion models)
        for quality_model in ["TabDDPM", "TabSyn", "AutoDiff"]:
            if quality_model in candidates:
                return quality_model
        
        return sorted_models[0]["model_name"]

    def _build_hard_problem_result(
        self,
        recommended_model: str,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        eligible_models: List[str],
        excluded_models: Dict[str, str],
        hard_details: Dict[str, bool],
        top_n: int,
    ) -> RecommendationResult:
        """Build recommendation result for Hard Problem cases."""
        
        model_info = self.models.get(recommended_model, {})
        
        # Build reasoning
        reasoning = [
            "🔴 HARD PROBLEM DETECTED: Dataset exhibits all three critical stress factors",
            f"  • Severe Skew: {hard_details['severe_skew']}",
            f"  • High Cardinality: {hard_details['high_cardinality']}",
            f"  • Zipfian Distribution: {hard_details['zipfian']}",
        ]
        
        if recommended_model == "GReaT":
            reasoning.append(
                "✓ GReaT selected: Best-in-class for handling complex tail distributions"
            )
        elif recommended_model == "TabDDPM":
            reasoning.append(
                "✓ TabDDPM selected: GReaT too slow for large data, using diffusion fallback"
            )
        else:
            reasoning.append(
                f"✓ {recommended_model} selected: Best available backup for hard problems"
            )
        
        # Warnings
        warnings = []
        if dataset_profile.row_count > self.config.large_data_threshold:
            warnings.append(
                f"⚠ Large dataset ({dataset_profile.row_count} rows) may require significant compute time"
            )
        if recommended_model != "GReaT" and "GReaT" not in eligible_models:
            warnings.append(
                "⚠ GReaT (optimal for hard problems) was filtered out by constraints"
            )
        
        # Build primary recommendation
        primary = ModelRecommendation(
            model_name=recommended_model,
            confidence_score=0.95 if recommended_model == "GReaT" else 0.85,
            capability_match=model_info.get("capabilities", {}),
            reasoning=reasoning,
            warnings=warnings,
            model_info=model_info,
        )
        
        # Build alternatives from remaining eligible models
        alternatives = []
        alt_candidates = [m for m in eligible_models if m != recommended_model]
        for alt_name in alt_candidates[:top_n]:
            alt_info = self.models.get(alt_name, {})
            alternatives.append(
                ModelRecommendation(
                    model_name=alt_name,
                    confidence_score=0.70,
                    capability_match=alt_info.get("capabilities", {}),
                    reasoning=[f"Alternative to {recommended_model} for hard problem"],
                    warnings=[],
                    model_info=alt_info,
                )
            )
        
        # Difficulty summary
        difficulty_summary = {
            "max_column_difficulty": column_analysis.max_column_difficulty
            if column_analysis
            else 0,
            "stress_factors": dataset_profile.stress_factors.model_dump(),
            "dataset_size": {
                "rows": dataset_profile.row_count,
                "columns": dataset_profile.column_count,
            },
            "is_hard_problem": True,
            "hard_problem_details": hard_details,
            "hard_problem_routing": recommended_model,
        }
        
        return RecommendationResult(
            dataset_id=dataset_profile.dataset_id,
            method="rule_based_v2 (hard_problem_path)",
            recommended_model=primary,
            alternative_models=alternatives,
            constraints=constraints,
            difficulty_summary=difficulty_summary,
            excluded_models=excluded_models,
        )

    def _recommend_llm(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        top_n: int,
    ) -> RecommendationResult:
        """LLM-based recommendation using OpenAI API with SystemPrompt."""

        if not self.llm_available or not self.openai_client:
            raise RuntimeError(
                "LLM recommendation requires openai package and API key. "
                "Install with: pip install openai"
            )

        # Build LLM prompt
        prompt = self._build_llm_prompt(dataset_profile, column_analysis, constraints)

        # Prepare system message
        if self.system_prompt and self.system_prompt_loaded:
            system_message = self.system_prompt
            print(f"🤖 Using SystemPrompt from: {self.system_prompt_path.name}")
        else:
            system_message = """You are an expert in synthetic data generation models.

You have access to 12+ state-of-the-art synthesis models with different capabilities.
Your task is to analyze the dataset characteristics and recommend the best model.

Consider:
- Skewness (>2.0 requires specialized models)
- High cardinality (>500 unique values)
- Zipfian distributions (top 20% categories dominate)
- Data size (small <500 rows, large >50k rows)
- User constraints (cpu_only, differential privacy)

Return recommendations in JSON format with clear reasoning."""
            print(f"⚠ Using default prompt (SystemPrompt not loaded)")

        # Call OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic results
                response_format={"type": "json_object"},
            )

            llm_response = json.loads(response.choices[0].message.content)
            print(f"✓ LLM response received (model: {self.openai_model})")

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # Parse LLM response and build recommendations
        return self._parse_llm_response(
            llm_response, dataset_profile, column_analysis, constraints, top_n
        )

    def _recommend_hybrid(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        top_n: int,
    ) -> RecommendationResult:
        """Hybrid recommendation: Rule-based ranking + LLM reasoning."""
        
        # Get rule-based recommendations
        rule_based_result = self._recommend_rule_based(
            dataset_profile, column_analysis, constraints, top_n + 2
        )

        if not self.llm_available:
            # Fallback to rule-based if LLM not available
            rule_based_result.method = "hybrid (llm unavailable - used rule_based)"
            return rule_based_result

        # Get top 3-5 candidates from rule-based
        candidates = [rule_based_result.recommended_model.model_name] + [
            alt.model_name for alt in rule_based_result.alternative_models[:4]
        ]

        # Ask LLM to provide reasoning and re-rank
        prompt = self._build_hybrid_prompt(
            dataset_profile, column_analysis, constraints, candidates
        )

        # Prepare system message
        if self.system_prompt and self.system_prompt_loaded:
            system_message = self.system_prompt
            print(f"🤖 Hybrid mode: Using SystemPrompt from {self.system_prompt_path.name}")
        else:
            system_message = "You are an expert in synthetic data generation models."
            print(f"⚠ Hybrid mode: Using default prompt")

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            llm_response = json.loads(response.choices[0].message.content)
            print(f"✓ LLM reasoning added to hybrid recommendation")

            # Use LLM's ranking but keep rule-based structure
            primary_name = llm_response.get("recommended_model")
            llm_reasoning = llm_response.get("reasoning", "")

            # Find the recommended model in our candidates
            primary = next(
                (
                    rec
                    for rec in [rule_based_result.recommended_model]
                    + rule_based_result.alternative_models
                    if rec.model_name == primary_name
                ),
                rule_based_result.recommended_model,
            )

            # Add LLM reasoning to the recommendation
            primary.reasoning = [llm_reasoning] + primary.reasoning

            # Re-order alternatives
            alternative_names = llm_response.get("alternatives", [])
            alternatives = []
            for alt_name in alternative_names[:top_n]:
                alt = next(
                    (
                        rec
                        for rec in [rule_based_result.recommended_model]
                        + rule_based_result.alternative_models
                        if rec.model_name == alt_name
                    ),
                    None,
                )
                if alt and alt.model_name != primary_name:
                    alternatives.append(alt)

            rule_based_result.method = "hybrid"
            rule_based_result.recommended_model = primary
            rule_based_result.alternative_models = alternatives
            rule_based_result.llm_reasoning = llm_reasoning

            return rule_based_result

        except Exception as e:
            # Fallback to rule-based if LLM fails
            rule_based_result.method = f"hybrid (llm failed: {str(e)} - used rule_based)"
            return rule_based_result

    def _build_llm_prompt(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
    ) -> str:
        """Build prompt for LLM recommendation."""
        
        # Serialize dataset profile and column analysis
        profile_json = dataset_profile.model_dump_json(indent=2)
        column_json = (
            column_analysis.model_dump_json(indent=2) if column_analysis else "null"
        )

        # Get model capabilities
        models_json = json.dumps(self.models, indent=2)

        prompt = f"""You are an expert in synthetic data generation. Analyze this dataset profile and recommend the best model from the available options.

# Dataset Profile
{profile_json}

# Column-Level Analysis
{column_json}

# User Constraints
{json.dumps(constraints, indent=2)}

# Available Models
{models_json}

Please analyze the dataset characteristics and provide a JSON response with:
{{
  "recommended_model": "ModelName",
  "reasoning": "Detailed explanation of why this model is best for this dataset",
  "alternatives": ["Alternative1", "Alternative2", "Alternative3"],
  "warnings": ["Any potential issues or limitations"],
  "confidence": 0.0-1.0
}}

Consider:
1. Dataset size ({dataset_profile.row_count} rows × {dataset_profile.column_count} columns)
2. Active stress factors: {', '.join([k for k, v in dataset_profile.stress_factors.model_dump().items() if v])}
3. User constraints: {', '.join(f'{k}={v}' for k, v in constraints.items())}
4. Model capabilities vs. dataset requirements
5. Performance trade-offs (speed vs. quality)

Focus on matching model capabilities to dataset difficulty. Return ONLY valid JSON."""

        return prompt

    def _build_hybrid_prompt(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        candidates: List[str],
    ) -> str:
        """Build prompt for hybrid recommendation (re-ranking candidates)."""
        
        profile_summary = {
            "rows": dataset_profile.row_count,
            "columns": dataset_profile.column_count,
            "stress_factors": {
                k: v
                for k, v in dataset_profile.stress_factors.model_dump().items()
                if v
            },
            "max_column_difficulty": column_analysis.max_column_difficulty
            if column_analysis
            else "unknown",
        }

        candidate_info = {
            name: self.models[name] for name in candidates if name in self.models
        }

        prompt = f"""You are an expert in synthetic data generation. Review these pre-filtered candidate models and select the best one for this dataset.

# Dataset Summary
{json.dumps(profile_summary, indent=2)}

# Candidate Models (pre-filtered by rule-based system)
{json.dumps(candidate_info, indent=2)}

# User Constraints
{json.dumps(constraints, indent=2)}

Please provide a JSON response with:
{{
  "recommended_model": "BestCandidateName",
  "reasoning": "Clear explanation of why this model is best",
  "alternatives": ["Alternative1", "Alternative2"],
  "key_factors": ["Factor1", "Factor2", "Factor3"]
}}

Focus on:
1. Which model best handles the active stress factors
2. Performance vs. quality trade-offs
3. User constraints compatibility
4. Practical considerations (training time, memory, etc.)

Return ONLY valid JSON."""

        return prompt

    def _parse_llm_response(
        self,
        llm_response: Dict[str, Any],
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        top_n: int,
    ) -> RecommendationResult:
        """Parse LLM JSON response into RecommendationResult."""
        
        recommended_name = llm_response["recommended_model"]
        reasoning_text = llm_response["reasoning"]
        alternatives_list = llm_response.get("alternatives", [])
        warnings_list = llm_response.get("warnings", [])
        confidence = llm_response.get("confidence", 0.8)

        # Build primary recommendation
        model_info = self.models.get(recommended_name, {})
        primary = ModelRecommendation(
            model_name=recommended_name,
            confidence_score=confidence,
            capability_match=model_info.get("capabilities", {}),
            reasoning=[reasoning_text],
            warnings=warnings_list,
            model_info=model_info,
        )

        # Build alternatives
        alternatives = []
        for alt_name in alternatives_list[:top_n]:
            if alt_name in self.models:
                alt_info = self.models[alt_name]
                alternatives.append(
                    ModelRecommendation(
                        model_name=alt_name,
                        confidence_score=confidence * 0.8,  # Lower confidence for alternatives
                        capability_match=alt_info.get("capabilities", {}),
                        reasoning=[f"Alternative to {recommended_name}"],
                        warnings=[],
                        model_info=alt_info,
                    )
                )

        # Build difficulty summary
        difficulty_summary = {
            "max_column_difficulty": column_analysis.max_column_difficulty
            if column_analysis
            else 0,
            "stress_factors": dataset_profile.stress_factors.model_dump(),
            "dataset_size": {
                "rows": dataset_profile.row_count,
                "columns": dataset_profile.column_count,
            },
        }

        return RecommendationResult(
            dataset_id=dataset_profile.dataset_id,
            method="llm",
            recommended_model=primary,
            alternative_models=alternatives,
            llm_reasoning=reasoning_text,
            constraints=constraints,
            difficulty_summary=difficulty_summary,
            excluded_models={},
        )

    # === Rule-based helper methods ===

    def _apply_hard_filters(
        self, constraints: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Apply hard constraint filters."""
        eligible = []
        excluded = {}

        for model_name, model_info in self.models.items():
            model_constraints = model_info["constraints"]

            # CPU-only constraint
            if constraints.get("cpu_only", False):
                if not model_constraints.get("cpu_only_compatible", False):
                    excluded[model_name] = "Requires GPU (cpu_only constraint active)"
                    continue

            # Strict DP constraint
            if constraints.get("strict_dp", False):
                if model_info["capabilities"]["privacy_dp"] < 3:
                    excluded[model_name] = "Insufficient DP (strict_dp constraint)"
                    continue

            # Row constraints
            min_rows = model_constraints.get("min_rows", 0)
            max_rows = model_constraints.get("max_recommended_rows", float("inf"))
            dataset_rows = constraints.get("dataset_rows", None)

            if dataset_rows is not None:
                if dataset_rows < min_rows:
                    excluded[model_name] = f"Dataset too small ({dataset_rows} < {min_rows})"
                    continue
                if dataset_rows > max_rows:
                    excluded[model_name] = f"Dataset too large ({dataset_rows} > {max_rows})"
                    continue

            eligible.append(model_name)

        return eligible, excluded

    def _calculate_required_capabilities(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
    ) -> Dict[str, int]:
        """Calculate required capabilities from dataset."""
        required = {
            "skew_handling": 0,
            "cardinality_handling": 0,
            "zipfian_handling": 0,
            "small_data": 0,
            "correlation_handling": 0,
        }

        stress = dataset_profile.stress_factors

        # Skew
        if stress.severe_skew and dataset_profile.skewness:
            max_skew = dataset_profile.skewness.max_skewness
            required["skew_handling"] = 4 if max_skew >= 4.0 else 3

        # Cardinality
        if stress.high_cardinality and dataset_profile.cardinality:
            max_card = dataset_profile.cardinality.max_cardinality
            required["cardinality_handling"] = 4 if max_card >= 5000 else 3

        # Zipfian
        if stress.zipfian_distribution and dataset_profile.zipfian:
            if dataset_profile.zipfian.top_20_percent_ratio:
                ratio = dataset_profile.zipfian.top_20_percent_ratio
                required["zipfian_handling"] = 4 if ratio >= 0.9 else 3

        # Small data
        if stress.small_data:
            required["small_data"] = 4

        # Correlation
        if stress.higher_order_correlation:
            required["correlation_handling"] = 3

        return required

    def _score_models(
        self, eligible_models: List[str], required_capabilities: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Score models based on capability match."""
        scored = []

        for model_name in eligible_models:
            model_capabilities = self.models[model_name]["capabilities"]

            capability_scores = {}
            total_score = 0.0

            for capability, required in required_capabilities.items():
                model_score = model_capabilities.get(capability, 0)

                # Scoring
                if model_score >= required:
                    match_score = 1.0
                elif model_score == required - 1:
                    match_score = 0.7
                elif model_score == required - 2:
                    match_score = 0.4
                else:
                    match_score = 0.0

                weight = 1.0 if required > 0 else 0.1
                total_score += match_score * weight

                capability_scores[capability] = {
                    "required": required,
                    "model_score": model_score,
                    "match_score": match_score,
                }

            scored.append(
                {
                    "model_name": model_name,
                    "model_info": self.models[model_name],
                    "capability_scores": capability_scores,
                    "total_score": total_score,
                }
            )

        return scored

    def _build_recommendation(
        self,
        scored_model: Dict[str, Any],
        required_capabilities: Dict[str, int],
        constraints: Dict[str, Any],
    ) -> ModelRecommendation:
        """Build ModelRecommendation from scored model."""
        model_name = scored_model["model_name"]
        model_info = scored_model["model_info"]
        capability_scores = scored_model["capability_scores"]

        capability_match = {
            cap: scores["model_score"]
            for cap, scores in capability_scores.items()
            if scores["required"] > 0
        }

        reasoning = []
        warnings = []

        # Add strengths
        for strength in model_info["strengths"][:3]:
            reasoning.append(f"✓ {strength}")

        # Add matches
        for cap, scores in capability_scores.items():
            if scores["required"] > 0:
                if scores["match_score"] == 1.0:
                    reasoning.append(
                        f"✓ {cap.replace('_', ' ')}: {scores['model_score']}/{scores['required']}"
                    )
                elif scores["match_score"] < 0.7:
                    warnings.append(
                        f"⚠ {cap.replace('_', ' ')}: {scores['model_score']} < {scores['required']}"
                    )

        # Add limitations
        for limitation in model_info["limitations"][:2]:
            warnings.append(f"⚠ {limitation}")

        # Performance
        perf = model_info["performance"]
        reasoning.append(
            f"Performance: {perf['training_speed']} training, {perf['memory_usage']} memory"
        )

        # Confidence
        max_possible = sum(1.0 for req in required_capabilities.values() if req > 0)
        confidence = (
            scored_model["total_score"] / max_possible if max_possible > 0 else 1.0
        )
        confidence = min(1.0, max(0.0, confidence))

        return ModelRecommendation(
            model_name=model_name,
            confidence_score=confidence,
            capability_match=capability_match,
            reasoning=reasoning,
            warnings=warnings,
            model_info=model_info,
        )


# Convenience function
def recommend_model(
    dataset_profile: DatasetProfile,
    column_analysis: Optional[ColumnAnalysisResult] = None,
    constraints: Optional[Dict[str, Any]] = None,
    method: str = "rule_based",
    config: Optional[EngineConfig] = None,
    openai_api_key: Optional[str] = None,
) -> RecommendationResult:
    """Convenience function to get recommendations.

    Args:
        dataset_profile: Dataset profile from StochasticDataAnalyzer
        column_analysis: Optional column analysis from ColumnAnalyzer
        constraints: Optional constraints dict (cpu_only, strict_dp, prefer_speed)
        method: 'rule_based', 'llm', or 'hybrid'
        config: Optional EngineConfig for custom thresholds
        openai_api_key: OpenAI API key (required for 'llm' or 'hybrid')

    Returns:
        RecommendationResult
    """
    engine = ModelRecommendationEngine(
        config=config,
        openai_api_key=openai_api_key,
    )
    return engine.recommend(dataset_profile, column_analysis, constraints, method=method)

