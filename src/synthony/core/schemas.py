"""
Pydantic schemas for dataset profiling output.

Defines the structured output format for all data analysis results,
ensuring type safety and JSON serialization compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SkewnessMetrics(BaseModel):
    """Per-column skewness analysis results."""

    column_scores: Dict[str, float] = Field(
        default_factory=dict, description="Skewness coefficient for each numeric column"
    )
    max_skewness: float = Field(description="Maximum absolute skewness across all columns")
    severe_columns: List[str] = Field(
        default_factory=list, description="Columns with |skewness| > threshold (default 2.0)"
    )

    # model_config = {"json_schema_extra": {"example": {"column_scores": {"age": 3.2, "income": 4.1}, "max_skewness": 4.1, "severe_columns": ["income"]}}}


class CardinalityMetrics(BaseModel):
    """Per-column cardinality (unique value count) analysis."""

    column_counts: Dict[str, int] = Field(
        default_factory=dict, description="Number of unique values per column"
    )
    max_cardinality: int = Field(description="Maximum cardinality across all columns")
    high_cardinality_columns: List[str] = Field(
        default_factory=list, description="Columns with unique count > threshold (default 500)"
    )

    # model_config = {"json_schema_extra": {"example": {"column_counts": {"user_id": 10000, "category": 50}, "max_cardinality": 10000, "high_cardinality_columns": ["user_id"]}}}


class ZipfianMetrics(BaseModel):
    """Zipfian (power-law) distribution detection results."""

    detected: bool = Field(description="Whether Zipfian distribution was detected in any column")
    top_20_percent_ratio: Optional[float] = Field(
        default=None,
        description="Concentration ratio: fraction of data in top 20% of categories (max across columns)",
    )
    affected_columns: List[str] = Field(
        default_factory=list, description="Categorical columns exhibiting Zipfian behavior"
    )

    # model_config = {"json_schema_extra": {"example": {"detected": True, "top_20_percent_ratio": 0.92, "affected_columns": ["category"]}}}


class CorrelationMetrics(BaseModel):
    """Higher-order (non-linear) correlation detection results."""

    correlation_density: float = Field(
        description="Fraction of column pairs with |correlation| > threshold (default 0.1)"
    )
    mean_r_squared: float = Field(
        description="Average linear R² for densely correlated pairs (0 if not dense)"
    )
    has_higher_order: bool = Field(
        description="True if dense correlations but low linear R² (indicates non-linear relationships)"
  )
    correlation_matrix: Optional[Any] = Field(
        default=None,
        description="Pandas DataFrame containing the correlation matrix (optional)"
    )

    model_config = {"arbitrary_types_allowed": True}


class StressFactors(BaseModel):
    """Boolean flags indicating detected data stress factors.

    These flags identify statistical characteristics that present challenges
    for synthetic data generation models.
    """

    severe_skew: bool = Field(description="Any column has |skewness| > 2.0 (long-tailed distribution)")
    high_cardinality: bool = Field(
        description="Any column has >500 unique values (mode collapse risk)"
    )
    zipfian_distribution: bool = Field(
        description="Top 20% of categories contain >80% of data (power-law distribution)"
    )
    small_data: bool = Field(description="Row count < 500 (overfitting risk)")
    large_data: bool = Field(description="Row count > 50,000 (LLM context limitations)")
    higher_order_correlation: bool = Field(
        description="Dense correlation matrix with low linear R² (non-linear relationships)"
    )

    # model_config = {"json_schema_extra": {"example": {"severe_skew": True, "high_cardinality": False, "zipfian_distribution": True, "small_data": False, "large_data": False, "higher_order_correlation": False}}}


class DatasetProfile(BaseModel):
    """Complete statistical profile of a tabular dataset.

    This is the primary output of the StochasticDataAnalyzer, containing
    both high-level stress factor flags and detailed statistical metrics.
    """

    # Metadata
    dataset_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique identifier for this analysis"
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="When the analysis was performed"
    )
    row_count: int = Field(ge=0, description="Number of rows in the dataset")
    column_count: int = Field(ge=0, description="Number of columns in the dataset")

    # High-level stress detection
    stress_factors: StressFactors = Field(description="Boolean flags for detected stress factors")

    # Detailed metrics
    skewness: Optional[SkewnessMetrics] = Field(
        default=None, description="Detailed skewness analysis per column"
    )
    cardinality: Optional[CardinalityMetrics] = Field(
        default=None, description="Detailed cardinality analysis per column"
    )
    zipfian: Optional[ZipfianMetrics] = Field(
        default=None, description="Detailed Zipfian distribution analysis"
    )
    correlation: Optional[CorrelationMetrics] = Field(
        default=None, description="Detailed correlation complexity analysis"
    )

    # Data quality indicators
    null_percentage: Dict[str, float] = Field(
        default_factory=dict, description="Percentage of null values per column"
    )
    column_types: Dict[str, str] = Field(
        default_factory=dict, description="Data type of each column (numeric, categorical, etc.)"
    )

    # Configuration used
    thresholds_used: Dict[str, float] = Field(
        default_factory=dict, description="Threshold values used for stress detection"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "analysis_timestamp": "2026-01-09T12:00:00",
                "row_count": 10000,
                "column_count": 15,
                "stress_factors": {
                    "severe_skew": True,
                    "high_cardinality": False,
                    "zipfian_distribution": True,
                    "small_data": False,
                    "large_data": False,
                    "higher_order_correlation": False,
                },
                "skewness": {
                    "column_scores": {"income": 4.2},
                    "max_skewness": 4.2,
                    "severe_columns": ["income"],
                },
                "thresholds_used": {
                    "skewness_threshold": 2.0,
                    "cardinality_threshold": 500,
                    "zipfian_ratio": 0.80,
                },
            }
        }
    }

    def to_json(self) -> str:
        """Serialize profile to JSON string."""
        # Exclude correlation_matrix as it contains a pandas DataFrame
        return self.model_dump_json(indent=2, exclude={"correlation": {"correlation_matrix"}})

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetProfile":
        """Deserialize profile from JSON string."""
        return cls.model_validate_json(json_str)


class RecommendationConstraints(BaseModel):
    """User constraints for model recommendation.
    
    Defines hardware, privacy, and performance requirements that filter
    the available model pool before scoring.
    
    Scale Factors (SF):
    - SF = 1.0 (default) means no scaling (standard importance)
    - SF > 1.0 means "this capability is MORE important" (amplifies score)
    - SF < 1.0 (but > 0) means "this capability is LESS important" (reduces score)
    - SF = 0.0 means "ignore this capability entirely" (zero weight)
    """
    
    cpu_only: bool = Field(default=False, description="Require CPU-only models (no GPU dependency)")
    strict_dp: bool = Field(default=False, description="Require differential privacy guarantees")
    prefer_speed: bool = Field(default=False, description="Prioritize training speed over quality")
    dataset_rows: Optional[int] = Field(default=None, description="Number of rows in dataset (for size filtering)")
    
    # Scale Factors for Capability Scores (default 1.0 = no scaling)
    skew_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Skewness capability score")
    cardinality_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Cardinality capability score")
    zipfian_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Zipfian capability score")
    small_data_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Small Data capability score")
    correlation_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Correlation capability score")
    privacy_dp_sf: float = Field(default=1.0, ge=0.0, le=10.0, description="Scale factor for Privacy/Differential Privacy capability score")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_only": True,
                "strict_dp": False,
                "prefer_speed": False,
                "dataset_rows": 10000,
                "skew_sf": 1.0,
                "cardinality_sf": 1.0,
                "zipfian_sf": 1.0,
                "small_data_sf": 1.0,
                "correlation_sf": 1.0,
                "privacy_dp_sf": 1.0
            }
        }
    }


# ============================================================================
# Column-Level Analysis Schemas
# ============================================================================


class ColumnStressFactors(BaseModel):
    """Per-column stress factor indicators.

    Identifies which specific stress factors affect this column,
    enabling column-level model capability requirements.
    """

    severe_skew: bool = Field(
        default=False,
        description="Column has |skewness| > 2.0 (requires models with skew-handling capability ≥3)"
    )
    high_cardinality: bool = Field(
        default=False,
        description="Column has >500 unique values (requires models with cardinality capability ≥3)"
    )
    zipfian: bool = Field(
        default=False,
        description="Column exhibits Zipfian distribution (requires models with zipfian capability ≥3)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "severe_skew": True,
                "high_cardinality": True,
                "zipfian": False
            }
        }
    }


class ColumnDifficultyScore(BaseModel):
    """Difficulty score and required model capability for a column.

    Maps column characteristics to minimum model capability scores (0-4 scale)
    needed to properly synthesize this column.

    Scoring reference:
    - Score 0-1: Trivial (any model)
    - Score 2: Moderate (most models except basic GANs)
    - Score 3: Hard (requires specialized models)
    - Score 4: Very Hard (requires advanced models like GReaT, TabSyn)
    """

    overall_difficulty: int = Field(
        ge=0, le=4,
        description="Overall difficulty score (0-4), max of individual dimension scores"
    )
    skew_difficulty: int = Field(
        ge=0, le=4,
        description="Required model capability for skew handling (0=none, 4=severe)"
    )
    cardinality_difficulty: int = Field(
        ge=0, le=4,
        description="Required model capability for cardinality handling (0=low, 4=very high)"
    )
    zipfian_difficulty: int = Field(
        ge=0, le=4,
        description="Required model capability for Zipfian distribution (0=none, 4=extreme)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "overall_difficulty": 4,
                "skew_difficulty": 3,
                "cardinality_difficulty": 4,
                "zipfian_difficulty": 0
            }
        }
    }


class ColumnProfile(BaseModel):
    """Statistical profile for a single column.

    Contains metrics and difficulty scores only - no model recommendations.
    Model recommendations are handled by the Recommender component which
    considers the full picture (constraints, hardware, privacy, etc.).
    """

    column_name: str = Field(description="Name of the column")
    column_type: str = Field(description="Data type (numeric, categorical)")

    # Statistical metrics
    unique_count: int = Field(ge=0, description="Number of unique values")
    null_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of null values")
    skewness: Optional[float] = Field(
        default=None,
        description="Fisher-Pearson skewness coefficient (None for categorical)"
    )
    zipfian_ratio: Optional[float] = Field(
        default=None,
        description="Top 20% concentration ratio (None for numeric)"
    )

    # Stress indicators
    stress_factors: ColumnStressFactors = Field(
        description="Binary flags for detected stress factors"
    )

    # Difficulty scoring
    difficulty: ColumnDifficultyScore = Field(
        description="Difficulty scores (0-4 scale) per dimension"
    )

    # NOTE: Model recommendations are NOT included here.
    # Recommendations are computed by the Recommender, which considers:
    # - Dataset-level stress factors
    # - User constraints (hardware, privacy, latency)
    # - All columns together (not just one)




class ColumnAnalysisResult(BaseModel):
    """Aggregated column-level analysis results for entire dataset.

    Provides per-column profiles and overall dataset difficulty assessment.
    """

    # Metadata
    dataset_id: str = Field(description="References the parent DatasetProfile")
    column_count: int = Field(ge=0, description="Total number of columns analyzed")

    # Per-column profiles
    columns: Dict[str, ColumnProfile] = Field(
        default_factory=dict,
        description="Column name -> ColumnProfile mapping"
    )

    # Overall dataset difficulty
    max_column_difficulty: int = Field(
        ge=0, le=4,
        description="Maximum difficulty score across all columns"
    )
    difficult_columns: List[str] = Field(
        default_factory=list,
        description="Columns with difficulty ≥3 (requiring specialized models)"
    )

    # Summary statistics
    stress_factor_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of columns affected by each stress factor"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "column_count": 9,
                "max_column_difficulty": 4,
                "difficult_columns": ["Height", "Whole weight"],
                "stress_factor_summary": {
                    "severe_skew": 1,
                    "high_cardinality": 4,
                    "zipfian": 0
                }
            }
        }
    }

    def to_json(self) -> str:
        """Serialize column analysis to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ColumnAnalysisResult":
        """Deserialize column analysis from JSON string."""
        return cls.model_validate_json(json_str)
