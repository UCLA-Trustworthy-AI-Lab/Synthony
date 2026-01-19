"""
Cardinality analysis and Zipfian distribution detection.

Identifies high-cardinality columns and power-law (Zipfian) distributions
where a small number of categories dominate the data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthony.core.schemas import CardinalityMetrics, ZipfianMetrics


class CardinalityDetector:
    """Unique value counting and Zipfian distribution detection.

    Cardinality refers to the number of unique values in a column.
    High cardinality (>500 unique values) can cause mode collapse in
    generative models.

    Zipfian distribution (power-law) occurs when a small percentage of
    categories account for the majority of the data. This is detected by
    checking if the top 20% of categories contain >80% of the data.
    """

    def __init__(
        self, cardinality_threshold: int = 500, zipfian_ratio: float = 0.80
    ):
        """Initialize cardinality and Zipfian detector.

        Args:
            cardinality_threshold: Unique count above which to flag as high cardinality.
                                  Default 500 from architecture docs.
            zipfian_ratio: Concentration threshold for Zipfian detection.
                          Default 0.80 means top 20% contains >80% of data.
        """
        self.cardinality_threshold = cardinality_threshold
        self.zipfian_ratio = zipfian_ratio

    def analyze(
        self, df: pd.DataFrame
    ) -> Tuple[CardinalityMetrics, ZipfianMetrics]:
        """Analyze cardinality and Zipfian patterns.

        Cardinality Algorithm:
        1. Count unique values per column
        2. Flag if > threshold (default 500)

        Zipfian Detection Algorithm:
        1. For categorical/object columns:
           a. Count value frequencies using value_counts()
           b. Sort counts descending
           c. Calculate top 20% index: n_top = ceil(0.2 * num_unique)
           d. Calculate ratio: sum(top_n_counts) / total_rows
           e. Flag if ratio > 0.80 (concentration in few categories)

        Edge Cases:
        - Empty column: cardinality = 0, skip Zipfian
        - Single category: ratio = 1.0 (not Zipfian by definition)
        - All unique values: ratio = 0.2 (uniform, not Zipfian)
        - Numeric columns: included in cardinality, excluded from Zipfian

        Args:
            df: DataFrame to analyze

        Returns:
            Tuple of (CardinalityMetrics, ZipfianMetrics)
        """
        cardinality_results: Dict[str, int] = {}
        high_card_cols: List[str] = []
        zipfian_cols: List[str] = []
        zipfian_ratio_overall: Optional[float] = None

        # Analyze each column
        for col in df.columns:
            # Cardinality: count unique values (including NaN as unique)
            unique_count = df[col].nunique(dropna=False)
            cardinality_results[col] = unique_count

            # High cardinality check
            if unique_count > self.cardinality_threshold:
                high_card_cols.append(col)

            # Zipfian check: only for categorical/object columns
            # Numeric columns can have high cardinality but aren't categorical
            if df[col].dtype in ["object", "category", "string"]:
                # Need at least 2 unique categories for Zipfian to be meaningful
                if unique_count > 1:
                    # Get value counts (sorted descending by default)
                    value_counts = df[col].value_counts()

                    # Calculate top 20% index
                    n_top = max(1, int(np.ceil(0.2 * len(value_counts))))

                    # Calculate concentration ratio
                    top_sum = value_counts.head(n_top).sum()
                    total = len(df)
                    ratio = top_sum / total if total > 0 else 0.0

                    # Check if Zipfian (top 20% > threshold)
                    if ratio > self.zipfian_ratio:
                        zipfian_cols.append(col)

                        # Track maximum Zipfian ratio across columns
                        if zipfian_ratio_overall is None or ratio > zipfian_ratio_overall:
                            zipfian_ratio_overall = ratio

        # Build metrics
        cardinality_metrics = CardinalityMetrics(
            column_counts=cardinality_results,
            max_cardinality=max(cardinality_results.values()) if cardinality_results else 0,
            high_cardinality_columns=high_card_cols,
        )

        zipfian_metrics = ZipfianMetrics(
            detected=len(zipfian_cols) > 0,
            top_20_percent_ratio=zipfian_ratio_overall,
            affected_columns=zipfian_cols,
        )

        return cardinality_metrics, zipfian_metrics
