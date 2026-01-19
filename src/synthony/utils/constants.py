"""
Configuration constants and thresholds for data stress detection.

All threshold values are based on empirical research and documented in:
- docs/architecture_v2.md
- docs/SystemPrompt_v2.md
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class AnalyzerConfig:
    """Configuration for StochasticDataAnalyzer.

    All thresholds can be customized by passing a modified config instance
    to the analyzer's constructor.
    """

    # Stress detection thresholds
    skewness_threshold: float = 2.0
    """Fisher-Pearson skewness > 2.0 indicates severe tail distribution."""

    cardinality_threshold: int = 500
    """Unique values > 500 risks mode collapse in generative models."""

    zipfian_ratio: float = 0.80
    """Top 20% categories > 80% of data indicates power-law distribution."""

    small_data_threshold: int = 500
    """< 500 rows risks overfitting without tree-based models."""

    large_data_threshold: int = 50000
    """> 50k rows makes LLM-based synthesis impractical."""

    # Correlation detection thresholds
    correlation_density_threshold: float = 0.5
    """Fraction of column pairs with |correlation| > correlation_strength_threshold to be considered 'dense'."""

    correlation_strength_threshold: float = 0.1
    """Minimum |correlation| to count a pair as 'correlated'."""

    r2_threshold: float = 0.3
    """Low R² with high correlation = non-linear relationship."""

    # Performance limits
    max_correlation_columns: int = 100
    """Sample if more than this many numeric columns (O(n²) complexity)."""

    @classmethod
    def from_dict(cls, config: Dict[str, float]) -> "AnalyzerConfig":
        """Create config from dictionary.

        Args:
            config: Dictionary with threshold values

        Returns:
            AnalyzerConfig instance with specified values
        """
        return cls(**config)

    def to_dict(self) -> Dict[str, float]:
        """Convert config to dictionary for JSON serialization."""
        return {
            "skewness_threshold": self.skewness_threshold,
            "cardinality_threshold": self.cardinality_threshold,
            "zipfian_ratio": self.zipfian_ratio,
            "small_data_threshold": self.small_data_threshold,
            "large_data_threshold": self.large_data_threshold,
            "correlation_density_threshold": self.correlation_density_threshold,
            "correlation_strength_threshold": self.correlation_strength_threshold,
            "r2_threshold": self.r2_threshold,
        }


# Default configuration instance
DEFAULT_CONFIG = AnalyzerConfig()


# Threshold explanations (for documentation and error messages)
THRESHOLD_DOCS = {
    "skewness_threshold": "Fisher-Pearson skewness > 2.0 indicates severe tail distribution that breaks basic GANs/VAEs",
    "cardinality_threshold": "Unique values > 500 risks mode collapse in generative models",
    "zipfian_ratio": "Top 20% categories > 80% of data indicates power-law distribution requiring specialized tokenization",
    "small_data_threshold": "< 500 rows risks overfitting without tree-based models like ARF",
    "large_data_threshold": "> 50k rows makes LLM-based synthesis impractical due to context window limitations",
    "correlation_density_threshold": "Fraction of pairs with strong correlation to consider matrix 'dense'",
    "correlation_strength_threshold": "Minimum correlation strength to count as 'correlated'",
    "r2_threshold": "Low R² combined with high correlation indicates non-linear (higher-order) relationships",
}
