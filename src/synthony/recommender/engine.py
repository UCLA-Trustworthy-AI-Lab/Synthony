"""
Hybrid recommendation engine: Rule-based + LLM-based recommendations.

Implements both:
1. Rule-based scoring from model_capabilities.json (v3 with Hard Problem detection)
2. LLM-based inference using OpenAI API with SystemPrompt
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from synthony.core.schemas import ColumnAnalysisResult, DatasetProfile
from synthony.recommender.focus_profiles import get_scale_factors


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

    # Empirical quality bonus weight (applied to spark avg_quality_score)
    quality_weight: float = 0.3

    # DP threshold: models with privacy_dp >= this value qualify under strict_dp.
    # Default loaded from registry metadata.dp_threshold; fallback = 3 (score > 2).
    dp_min_score: int = 3

    # Required capability thresholds (loaded from registry metadata.capability_thresholds)
    # Skew: if max_skewness >= skew_severe_boundary → require skew_high, else skew_moderate
    skew_severe_boundary: float = 4.0
    skew_high_required: int = 4
    skew_moderate_required: int = 3
    # Cardinality: if max_cardinality >= cardinality_severe_boundary → high, else moderate
    cardinality_severe_boundary: int = 5000
    cardinality_high_required: int = 4
    cardinality_moderate_required: int = 3
    # Zipfian: if top_20_percent_ratio >= zipfian_severe_ratio → high, else moderate
    zipfian_severe_ratio: float = 0.9
    zipfian_high_required: int = 4
    zipfian_moderate_required: int = 3
    # Small data and correlation required levels
    small_data_required: int = 4
    correlation_required: int = 3

    # Hard Problem routing (loaded from registry hard_problem_routing)
    hard_problem_primary: str = "GReaT"
    hard_problem_large_data_fallback: str = "TabDDPM"
    hard_problem_fallback: List[str] = field(
        default_factory=lambda: ["ARF", "TabSyn", "CART", "SMOTE", "BayesianNetwork"]
    )

    # Hard Problem confidence scores (loaded from registry metadata.hard_problem_confidence)
    hard_problem_confidence_primary: float = 0.95
    hard_problem_confidence_fallback: float = 0.85
    hard_problem_confidence_alternative: float = 0.70

    # Score decay curve (loaded from registry metadata.score_decay)
    # Applied in _score_models when model capability is below required level
    score_decay_exact: float = 1.0       # model_score >= required
    score_decay_near: float = 0.7        # model_score == required - 1
    score_decay_moderate: float = 0.4    # model_score == required - 2
    score_decay_poor: float = 0.0        # model_score < required - 2
    
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
            "quality_weight": self.quality_weight,
            "dp_min_score": self.dp_min_score,
            "hard_problem_primary": self.hard_problem_primary,
            "hard_problem_large_data_fallback": self.hard_problem_large_data_fallback,
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

        # Load data-driven config from registry metadata
        registry_dp = self.registry.get("metadata", {}).get("dp_threshold")
        if registry_dp is not None:
            self.config.dp_min_score = int(registry_dp)

        # Load capability thresholds from registry metadata
        cap_thresholds = self.registry.get("metadata", {}).get("capability_thresholds", {})
        for key, value in cap_thresholds.items():
            if hasattr(self.config, key):
                setattr(self.config, key, type(getattr(self.config, key))(value))

        # Load hard problem routing from registry
        hp_routing = self.registry.get("hard_problem_routing", {})
        if hp_routing:
            if "primary" in hp_routing:
                self.config.hard_problem_primary = hp_routing["primary"]
            if "large_data_fallback" in hp_routing:
                self.config.hard_problem_large_data_fallback = hp_routing["large_data_fallback"]
            if "fallback_priority" in hp_routing:
                self.config.hard_problem_fallback = hp_routing["fallback_priority"]

        # Load hard problem confidence scores from registry metadata
        hp_conf = self.registry.get("metadata", {}).get("hard_problem_confidence", {})
        for key, value in hp_conf.items():
            attr = f"hard_problem_confidence_{key}"
            if hasattr(self.config, attr):
                setattr(self.config, attr, float(value))

        # Load score decay curve from registry metadata
        decay = self.registry.get("metadata", {}).get("score_decay", {})
        for key, value in decay.items():
            attr = f"score_decay_{key}"
            if hasattr(self.config, attr):
                setattr(self.config, attr, float(value))

        # Load system prompt (for LLM mode)
        if system_prompt_path is None:
            # Try environment variable first
            import os
            env_path = os.getenv("SYNTHONY_SYSTEM_PROMPT")
            if env_path:
                system_prompt_path = Path(env_path)
            else:
                # Default to SystemPrompt.md in config/
                system_prompt_path = (
                    Path(__file__).parent.parent.parent.parent
                    / "config"
                    / "SystemPrompt.md"
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

                # Validate the API key with a lightweight call
                try:
                    self.openai_client.models.list()
                    self.llm_available = True
                except Exception as e:
                    print(f"⚠ OpenAI API key validation failed: {e}")
                    self.openai_client = None
                    self.llm_available = False
                    self._try_vllm_fallback()
            else:
                self.openai_client = None
                self.llm_available = False
                self._try_vllm_fallback()
        except ImportError:
            self.openai_client = None
            self.llm_available = False

    def _try_vllm_fallback(self):
        """Attempt to fall back to vLLM if OpenAI is unavailable."""
        vllm_url = os.getenv("VLLM_URL")
        if not vllm_url or self.openai_base_url == vllm_url:
            return  # No vLLM configured or already tried vLLM

        print(f"↻ Attempting vLLM fallback at {vllm_url}")
        try:
            from openai import OpenAI

            vllm_key = os.getenv("VLLM_API_KEY", "EMPTY")
            vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

            client = OpenAI(api_key=vllm_key, base_url=vllm_url)
            client.models.list()

            self.openai_client = client
            self.openai_api_key = vllm_key
            self.openai_base_url = vllm_url
            self.openai_model = vllm_model
            self.llm_available = True
            print(f"✓ vLLM fallback succeeded (model: {vllm_model})")
        except Exception as e:
            print(f"⚠ vLLM fallback also failed: {e}")

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
        focus: Optional[str] = None,
        scale_factors: Optional[Dict[str, float]] = None,
    ) -> RecommendationResult:
        """Generate model recommendations.

        Args:
            dataset_profile: Dataset profile from StochasticDataAnalyzer
            column_analysis: Optional column analysis from ColumnAnalyzer
            constraints: Optional user constraints (cpu_only, strict_dp, etc.)
            top_n: Number of alternative recommendations
            method: Recommendation method - 'rule_based', 'llm', or 'hybrid'
            focus: Optional focus name (e.g. "privacy", "fidelity", "latency").
                Looks up scale factors from the focus registry.
            scale_factors: Optional dict of capability->float scale factors.
                Overrides focus if both provided. When provided, the hard
                problem path is skipped so that scoring respects scale factors.

        Returns:
            RecommendationResult with recommendations
        """
        # Convert RecommendationConstraints Pydantic model to dict if needed
        from synthony.core.schemas import RecommendationConstraints
        if isinstance(constraints, RecommendationConstraints):
            constraints = constraints.model_dump()

        constraints = constraints or {}
        constraints["dataset_rows"] = dataset_profile.row_count

        # Resolve scale factors: explicit > focus > None
        resolved_sf = scale_factors
        if resolved_sf is None and focus is not None:
            resolved_sf = get_scale_factors(focus)

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
                dataset_profile, column_analysis, constraints, top_n,
                scale_factors=resolved_sf,
            )

    def _recommend_rule_based(
        self,
        dataset_profile: DatasetProfile,
        column_analysis: Optional[ColumnAnalysisResult],
        constraints: Dict[str, Any],
        top_n: int,
        scale_factors: Optional[Dict[str, float]] = None,
    ) -> RecommendationResult:
        """Rule-based recommendation with v2 Hard Problem detection.

        Flow:
        1. Apply hard filters (cpu_only, strict_dp, size constraints)
        2. Check for Hard Problem (skew AND cardinality AND zipfian)
           - If yes AND no scale_factors: Route to GReaT with safety checks
           - If yes AND scale_factors: Skip hard problem path (use normal scoring)
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

        # Skip the hard problem path when scale_factors are provided so that
        # scoring respects the scale factors instead of using fixed routing.
        if is_hard and scale_factors is None:
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
        scored_models = self._score_models(
            eligible_models, required_capabilities, scale_factors=scale_factors
        )

        # Step 5: Sort by score
        sorted_models = sorted(
            scored_models, key=lambda x: x["total_score"], reverse=True
        )

        # Step 6: Apply tie-breaking (skipped when scale_factors provided,
        # since scale factors already encode user preference)
        if scale_factors is None:
            primary_name = self._apply_tie_breaking(
                sorted_models, dataset_profile, constraints
            )
        else:
            primary_name = sorted_models[0]["model_name"]

        # Re-order sorted_models to put primary first
        primary_model = next(
            m for m in sorted_models if m["model_name"] == primary_name
        )
        other_models = [
            m for m in sorted_models if m["model_name"] != primary_name
        ]

        # Step 7: Build recommendations
        primary = self._build_recommendation(
            primary_model, required_capabilities, constraints,
            scale_factors=scale_factors,
        )

        alternatives = [
            self._build_recommendation(
                model, required_capabilities, constraints,
                scale_factors=scale_factors,
            )
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

        All model names are read from config (loaded from registry
        hard_problem_routing), not hardcoded.

        Decision Logic:
        1. If rows > large_data_threshold → Use large_data_fallback
        2. If primary model in eligible pool → Use primary
        3. Else → Walk fallback_priority list

        Returns:
            Model name to recommend, or None if no suitable model found
        """
        row_count = dataset_profile.row_count

        # Large data: primary model may be too slow
        if row_count > self.config.large_data_threshold:
            fallback = self.config.hard_problem_large_data_fallback
            if fallback in eligible_models:
                return fallback

        # Primary choice for hard problems
        primary = self.config.hard_problem_primary
        if primary in eligible_models:
            return primary

        # Fallback priority from registry
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

        Uses priority lists from model_capabilities.json tie_breaking_priority:
        1. Small Data (<small_data_threshold rows) → small_data_priority
        2. Prefer Speed → speed_priority
        3. Otherwise → quality_priority (empirically calibrated)

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

        # Tie detected - apply rules using registry priorities
        row_count = dataset_profile.row_count
        prefer_speed = constraints.get("prefer_speed", self.config.prefer_speed_default)

        candidates = [m["model_name"] for m in sorted_models[:5]]

        # Read priority lists from registry (with sensible defaults)
        registry_tb = self.registry.get("tie_breaking_priority", {})

        # Rule 1: Small data → prefer models robust to overfitting
        if row_count < self.config.small_data_threshold:
            priority = registry_tb.get(
                "small_data_priority", ["ARF", "CART", "BayesianNetwork", "SMOTE"]
            )
            for model in priority:
                if model in candidates:
                    return model

        # Rule 2: Speed preference
        if prefer_speed:
            priority = registry_tb.get(
                "speed_priority", ["CART", "ARF", "SMOTE", "TVAE", "DPCART"]
            )
            for model in priority:
                if model in candidates:
                    return model

        # Rule 3: Quality preference — GPU vs CPU path
        cpu_only = constraints.get("cpu_only", False)
        if not cpu_only:
            # GPU available: prefer Diffusion/LLM/Transformer models
            priority = registry_tb.get(
                "gpu_quality_priority",
                ["GReaT", "TabDDPM", "TabSyn", "AutoDiff", "TVAE", "PATECTGAN"],
            )
            for model in priority:
                if model in candidates:
                    return model

        # CPU-only or no GPU model in candidates: use CPU quality priority
        priority = registry_tb.get(
            "cpu_quality_priority",
            registry_tb.get(
                "quality_priority", ["CART", "SMOTE", "BayesianNetwork", "ARF", "NFlow"]
            ),
        )
        for model in priority:
            if model in candidates:
                return model

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
        
        primary = self.config.hard_problem_primary
        if recommended_model == primary:
            strengths = model_info.get("strengths", [primary + " model"])
            reasoning.append(
                f"✓ {recommended_model} selected: {strengths[0] if strengths else 'Primary hard problem model'}"
            )
        elif recommended_model == self.config.hard_problem_large_data_fallback:
            reasoning.append(
                f"✓ {recommended_model} selected: {primary} too slow for large data, using fallback"
            )
        else:
            reasoning.append(
                f"✓ {recommended_model} selected: Best available from fallback priority"
            )
        
        # Warnings
        warnings = []
        if dataset_profile.row_count > self.config.large_data_threshold:
            warnings.append(
                f"⚠ Large dataset ({dataset_profile.row_count} rows) may require significant compute time"
            )
        hp_primary = self.config.hard_problem_primary
        if recommended_model != hp_primary and hp_primary not in eligible_models:
            warnings.append(
                f"⚠ {hp_primary} (optimal for hard problems) was filtered out by constraints"
            )

        # Build primary recommendation
        primary_conf = (
            self.config.hard_problem_confidence_primary
            if recommended_model == hp_primary
            else self.config.hard_problem_confidence_fallback
        )
        primary = ModelRecommendation(
            model_name=recommended_model,
            confidence_score=primary_conf,
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
                    confidence_score=self.config.hard_problem_confidence_alternative,
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
            model_count = len(self.models)
            model_names = ", ".join(sorted(self.models.keys()))
            system_message = f"""You are an expert in synthetic data generation models.

You have access to {model_count} state-of-the-art synthesis models: {model_names}.
Your task is to analyze the dataset characteristics and recommend the best model.

Consider:
- Skewness (>{self.config.skew_threshold} requires specialized models)
- High cardinality (>{self.config.cardinality_threshold} unique values)
- Zipfian distributions (top 20% categories dominate)
- Data size (small <{self.config.small_data_threshold} rows, large >{self.config.large_data_threshold} rows)
- User constraints (cpu_only, differential privacy with dp score >= {self.config.dp_min_score})

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

        allowed_models = constraints.get("allowed_models")

        for model_name, model_info in self.models.items():
            # Skip excluded models (not benchmarked or baselines)
            if model_info.get("exclude", False):
                excluded[model_name] = "Model excluded from recommendations (exclude=true)"
                continue

            # Restrict to allowed_models if specified
            if allowed_models is not None and model_name not in allowed_models:
                excluded[model_name] = "Not in allowed_models list"
                continue

            model_constraints = model_info["constraints"]

            # CPU-only constraint
            if constraints.get("cpu_only", False):
                if not model_constraints.get("cpu_only_compatible", False):
                    excluded[model_name] = "Requires GPU (cpu_only constraint active)"
                    continue

            # Strict DP constraint (threshold from registry metadata.dp_threshold)
            if constraints.get("strict_dp", False):
                if model_info["capabilities"]["privacy_dp"] < self.config.dp_min_score:
                    excluded[model_name] = (
                        f"Insufficient DP: privacy_dp={model_info['capabilities']['privacy_dp']} "
                        f"< required {self.config.dp_min_score}"
                    )
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
        """Calculate required capabilities from dataset.

        All thresholds are read from self.config (loaded from registry
        metadata.capability_thresholds), not hardcoded.
        """
        required = {
            "skew_handling": 0,
            "cardinality_handling": 0,
            "zipfian_handling": 0,
            "small_data": 0,
            "correlation_handling": 0,
        }

        stress = dataset_profile.stress_factors
        cfg = self.config

        # Skew
        if stress.severe_skew and dataset_profile.skewness:
            max_skew = dataset_profile.skewness.max_skewness
            required["skew_handling"] = (
                cfg.skew_high_required if max_skew >= cfg.skew_severe_boundary
                else cfg.skew_moderate_required
            )

        # Cardinality
        if stress.high_cardinality and dataset_profile.cardinality:
            max_card = dataset_profile.cardinality.max_cardinality
            required["cardinality_handling"] = (
                cfg.cardinality_high_required if max_card >= cfg.cardinality_severe_boundary
                else cfg.cardinality_moderate_required
            )

        # Zipfian
        if stress.zipfian_distribution and dataset_profile.zipfian:
            if dataset_profile.zipfian.top_20_percent_ratio:
                ratio = dataset_profile.zipfian.top_20_percent_ratio
                required["zipfian_handling"] = (
                    cfg.zipfian_high_required if ratio >= cfg.zipfian_severe_ratio
                    else cfg.zipfian_moderate_required
                )

        # Small data
        if stress.small_data:
            required["small_data"] = cfg.small_data_required

        # Correlation
        if stress.higher_order_correlation:
            required["correlation_handling"] = cfg.correlation_required

        return required

    def _score_models(
        self,
        eligible_models: List[str],
        required_capabilities: Dict[str, int],
        scale_factors: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score models based on capability match.

        Args:
            eligible_models: List of model names that passed hard filters.
            required_capabilities: Dict of capability->required_level.
            scale_factors: Optional dict of capability->float multiplier.
                When provided, ``weight = base_weight * scale_factor``.
        """
        scored = []

        # When scale_factors are provided, also score capabilities not in
        # required_capabilities (e.g. privacy_dp) so that scale factors
        # can promote models with those strengths.
        if scale_factors:
            all_caps = set(required_capabilities) | set(scale_factors)
        else:
            all_caps = set(required_capabilities)

        for model_name in eligible_models:
            model_capabilities = self.models[model_name]["capabilities"]

            capability_scores = {}
            total_score = 0.0

            for capability in all_caps:
                required = required_capabilities.get(capability, 0)
                model_score = model_capabilities.get(capability, 0)

                # When the capability is not required by the dataset, use
                # model_score/4.0 so that higher raw capability differentiates
                # models (e.g. correlation=4 beats correlation=1). This prevents
                # all models from scoring identically on non-stressed datasets.
                if required == 0:
                    match_score = model_score / 4.0
                elif model_score >= required:
                    match_score = self.config.score_decay_exact
                elif model_score == required - 1:
                    match_score = self.config.score_decay_near
                elif model_score == required - 2:
                    match_score = self.config.score_decay_moderate
                else:
                    match_score = self.config.score_decay_poor

                base_weight = 1.0 if required > 0 else 0.1
                sf = scale_factors.get(capability, 1.0) if scale_factors else 1.0
                weight = base_weight * sf
                total_score += match_score * weight

                capability_scores[capability] = {
                    "required": required,
                    "model_score": model_score,
                    "match_score": match_score,
                    "scale_factor": sf,
                    "weight": weight,
                }

            # Empirical quality bonus from spark benchmarks
            spark = self.models[model_name].get("spark_empirical", {})
            quality = spark.get("avg_quality_score", 0.5)
            quality_bonus = quality * self.config.quality_weight
            total_score += quality_bonus

            scored.append(
                {
                    "model_name": model_name,
                    "model_info": self.models[model_name],
                    "capability_scores": capability_scores,
                    "total_score": total_score,
                    "quality_bonus": quality_bonus,
                }
            )

        return scored

    def _build_recommendation(
        self,
        scored_model: Dict[str, Any],
        required_capabilities: Dict[str, int],
        constraints: Dict[str, Any],
        scale_factors: Optional[Dict[str, float]] = None,
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
        for strength in model_info.get("strengths", model_info.get("best_for", []))[:3]:
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
        for limitation in model_info.get("limitations", [])[:2]:
            warnings.append(f"⚠ {limitation}")

        # Performance
        perf = model_info["performance"]
        reasoning.append(
            f"Performance: {perf['training_speed']} training, {perf['memory_usage']} memory"
        )

        # Confidence: denominator accounts for scale factors
        if scale_factors:
            max_possible = sum(
                scale_factors.get(cap, 1.0)
                for cap, req in required_capabilities.items()
                if req > 0
            )
        else:
            max_possible = sum(
                1.0 for req in required_capabilities.values() if req > 0
            )
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
    focus: Optional[str] = None,
    scale_factors: Optional[Dict[str, float]] = None,
) -> RecommendationResult:
    """Convenience function to get recommendations.

    Args:
        dataset_profile: Dataset profile from StochasticDataAnalyzer
        column_analysis: Optional column analysis from ColumnAnalyzer
        constraints: Optional constraints dict (cpu_only, strict_dp, prefer_speed)
        method: 'rule_based', 'llm', or 'hybrid'
        config: Optional EngineConfig for custom thresholds
        openai_api_key: OpenAI API key (required for 'llm' or 'hybrid')
        focus: Optional focus name (e.g. "privacy", "fidelity", "latency")
        scale_factors: Optional dict of capability->float scale factors

    Returns:
        RecommendationResult
    """
    engine = ModelRecommendationEngine(
        config=config,
        openai_api_key=openai_api_key,
    )
    return engine.recommend(
        dataset_profile, column_analysis, constraints, method=method,
        focus=focus, scale_factors=scale_factors,
    )

