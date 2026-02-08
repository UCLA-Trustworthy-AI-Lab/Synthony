"""
Column-level analyzer for per-column difficulty scoring.

Computes difficulty scores (0-4 scale) for each column based on
statistical characteristics. Does NOT make model recommendations -
that responsibility belongs to the Recommender component.
"""


import numpy as np
import pandas as pd

from synthony.core.schemas import (
    ColumnAnalysisResult,
    ColumnDifficultyScore,
    ColumnProfile,
    ColumnStressFactors,
    DatasetProfile,
)


class ColumnAnalyzer:
    """Analyze individual columns and assign difficulty scores.

    Computes difficulty metrics using the 0-4 scoring scale. Does NOT
    make model recommendations - the Recommender handles that after
    considering constraints, hardware, privacy, and all columns together.

    Scoring Logic:
    - Skew: Score 0 (none), 2 (moderate <2.0), 3 (severe 2.0-4.0), 4 (extreme >4.0)
    - Cardinality: Score 0 (<50), 1 (<500), 3 (500-5000), 4 (>5000)
    - Zipfian: Score 0 (ratio <0.5), 1 (<0.8), 3 (0.8-0.9), 4 (>0.9)
    """

    def __init__(self):
        """Initialize column analyzer."""
        pass

    def analyze(
        self, df: pd.DataFrame, dataset_profile: DatasetProfile
    ) -> ColumnAnalysisResult:
        """Analyze all columns and generate difficulty profiles.

        Args:
            df: Original DataFrame (needed for per-column statistics)
            dataset_profile: Output from StochasticDataAnalyzer

        Returns:
            ColumnAnalysisResult with per-column profiles and difficulty scores
        """
        columns: dict[str, ColumnProfile] = {}

        for col in df.columns:
            column_profile = self._analyze_column(
                col, df[col], dataset_profile
            )
            columns[col] = column_profile

        # Aggregate statistics
        max_difficulty = max(
            (c.difficulty.overall_difficulty for c in columns.values()),
            default=0
        )
        difficult_columns = [
            name for name, prof in columns.items()
            if prof.difficulty.overall_difficulty >= 3
        ]

        stress_summary = {
            "severe_skew": sum(
                1 for c in columns.values() if c.stress_factors.severe_skew
            ),
            "high_cardinality": sum(
                1 for c in columns.values() if c.stress_factors.high_cardinality
            ),
            "zipfian": sum(
                1 for c in columns.values() if c.stress_factors.zipfian
            ),
        }

        return ColumnAnalysisResult(
            dataset_id=dataset_profile.dataset_id,
            column_count=len(columns),
            columns=columns,
            max_column_difficulty=max_difficulty,
            difficult_columns=difficult_columns,
            stress_factor_summary=stress_summary,
        )

    def _analyze_column(
        self, col_name: str, series: pd.Series, profile: DatasetProfile
    ) -> ColumnProfile:
        """Analyze a single column and generate its profile.

        Args:
            col_name: Name of the column
            series: Pandas Series for this column
            profile: Parent DatasetProfile

        Returns:
            ColumnProfile with difficulty scores and recommendations
        """
        # Determine column type
        is_numeric = pd.api.types.is_numeric_dtype(series)
        col_type = "numeric" if is_numeric else "categorical"

        # Basic statistics
        unique_count = series.nunique()
        null_pct = (series.isna().sum() / len(series)) * 100

        # Get skewness if available
        skewness = None
        if profile.skewness and col_name in profile.skewness.column_scores:
            skewness = profile.skewness.column_scores[col_name]

        # Get Zipfian ratio if available (categorical columns only)
        zipfian_ratio = None
        if not is_numeric and profile.cardinality:
            # Calculate Zipfian ratio for this column
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                n_top = max(1, int(np.ceil(0.2 * len(value_counts))))
                zipfian_ratio = value_counts.head(n_top).sum() / len(series)

        # Detect stress factors
        stress_factors = self._detect_stress_factors(
            col_name, unique_count, skewness, zipfian_ratio, profile
        )

        # Calculate difficulty scores
        difficulty = self._calculate_difficulty(
            unique_count, skewness, zipfian_ratio
        )

        # Generate recommendations
        # min_capability = difficulty.overall_difficulty
        # recommended_models = self._recommend_models(difficulty)

        return ColumnProfile(
            column_name=col_name,
            column_type=col_type,
            unique_count=unique_count,
            null_percentage=null_pct,
            skewness=skewness,
            zipfian_ratio=zipfian_ratio,
            stress_factors=stress_factors,
            difficulty=difficulty,
            # min_model_capability=min_capability,
            # recommended_model_types=recommended_models,
        )

    def _detect_stress_factors(
        self,
        col_name: str,
        unique_count: int,
        skewness: float | None,
        zipfian_ratio: float | None,
        profile: DatasetProfile,
    ) -> ColumnStressFactors:
        """Detect which stress factors affect this column.

        Args:
            col_name: Column name
            unique_count: Number of unique values
            skewness: Skewness coefficient (None for categorical)
            zipfian_ratio: Top 20% concentration (None for numeric)
            profile: Parent DatasetProfile

        Returns:
            ColumnStressFactors with boolean flags
        """
        severe_skew = False
        if skewness is not None and not np.isnan(skewness):
            severe_skew = abs(skewness) > 2.0

        high_cardinality = unique_count > 500

        zipfian = False
        if zipfian_ratio is not None:
            zipfian = zipfian_ratio > 0.80

        return ColumnStressFactors(
            severe_skew=severe_skew,
            high_cardinality=high_cardinality,
            zipfian=zipfian,
        )

    def _calculate_difficulty(
        self,
        unique_count: int,
        skewness: float | None,
        zipfian_ratio: float | None,
    ) -> ColumnDifficultyScore:
        """Calculate difficulty scores across dimensions.

        Scoring Logic:
        - Skewness: 0 (none), 2 (<2.0), 3 (2.0-4.0), 4 (>4.0)
        - Cardinality: 0 (<50), 1 (<500), 3 (500-5000), 4 (>5000)
        - Zipfian: 0 (<0.5), 1 (<0.8), 3 (0.8-0.9), 4 (>0.9)

        Args:
            unique_count: Number of unique values
            skewness: Skewness coefficient
            zipfian_ratio: Top 20% concentration

        Returns:
            ColumnDifficultyScore with per-dimension scores
        """
        # Skewness difficulty
        skew_diff = 0
        if skewness is not None and not np.isnan(skewness):
            abs_skew = abs(skewness)
            if abs_skew < 0.5:
                skew_diff = 0  # Nearly normal
            elif abs_skew < 2.0:
                skew_diff = 2  # Moderate skew
            elif abs_skew < 4.0:
                skew_diff = 3  # Severe skew (requires specialized models)
            else:
                skew_diff = 4  # Extreme skew (requires GReaT/Diffusion)

        # Cardinality difficulty
        if unique_count < 50:
            card_diff = 0  # Low cardinality
        elif unique_count < 500:
            card_diff = 1  # Moderate cardinality
        elif unique_count < 5000:
            card_diff = 3  # High cardinality (mode collapse risk)
        else:
            card_diff = 4  # Very high cardinality (requires advanced models)

        # Zipfian difficulty (categorical columns only)
        zipf_diff = 0
        if zipfian_ratio is not None:
            if zipfian_ratio < 0.5:
                zipf_diff = 0  # Balanced distribution
            elif zipfian_ratio < 0.8:
                zipf_diff = 1  # Slight imbalance
            elif zipfian_ratio < 0.9:
                zipf_diff = 3  # Zipfian (requires specialized tokenization)
            else:
                zipf_diff = 4  # Extreme Zipfian (needle in haystack)

        # Overall difficulty is the maximum across dimensions
        overall = max(skew_diff, card_diff, zipf_diff)

        return ColumnDifficultyScore(
            overall_difficulty=overall,
            skew_difficulty=skew_diff,
            cardinality_difficulty=card_diff,
            zipfian_difficulty=zipf_diff,
        )

