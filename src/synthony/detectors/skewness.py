"""
Skewness detection using Fisher-Pearson coefficient.

Identifies columns with severe tail distributions that challenge
basic generative models like GANs and VAEs.
"""


import numpy as np
import pandas as pd
from scipy.stats import skew

from synthony.core.schemas import SkewnessMetrics


class SkewnessDetector:
    """Fisher-Pearson skewness calculation and detection.

    Uses scipy.stats.skew to calculate the sample skewness coefficient
    for each numeric column. Flags columns with |skewness| > threshold
    as "severely skewed".

    The default threshold of 2.0 is based on empirical evidence that
    skewness beyond this value breaks basic GANs/VAEs, requiring diffusion
    models or LLMs for proper tail distribution capture.
    """

    def __init__(self, threshold: float = 2.0):
        """Initialize skewness detector.

        Args:
            threshold: Absolute skewness threshold for "severe" classification.
                      Default 2.0 aligns with architecture docs.
        """
        self.threshold = threshold

    def analyze(self, df: pd.DataFrame) -> SkewnessMetrics:
        """Calculate skewness for all numeric columns.

        Algorithm:
        1. Filter to numeric columns only
        2. For each column:
           - Remove NaN values
           - Calculate Fisher-Pearson coefficient using scipy.stats.skew
           - Flag if |skewness| > threshold
        3. Return aggregated metrics

        Edge Cases:
        - Single value column: skewness = 0 (no distribution)
        - All same values: skewness = 0 (no variance)
        - < 3 values: returns NaN (skewness undefined)
        - Non-numeric columns: skipped

        Args:
            df: DataFrame to analyze

        Returns:
            SkewnessMetrics with per-column scores and severe column list
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            # No numeric columns - return empty metrics
            return SkewnessMetrics(
                column_scores={}, max_skewness=0.0, severe_columns=[]
            )

        results: dict[str, float] = {}
        severe_columns: list[str] = []

        for col in numeric_cols:
            # Remove NaN values
            clean_data = df[col].dropna()

            # Check if we have enough data
            if len(clean_data) < 3:
                # Skewness is undefined for < 3 values
                results[col] = np.nan
                continue

            # Check if all values are the same
            if clean_data.nunique() == 1:
                # No variance = no skew
                results[col] = 0.0
                continue

            # Calculate Fisher-Pearson skewness
            # bias=False gives sample skewness (corrected for sample size)
            skewness = skew(clean_data, bias=False)
            results[col] = float(skewness)

            # Check if severe
            if abs(skewness) > self.threshold:
                severe_columns.append(col)

        # Calculate max absolute skewness
        valid_skewness = [s for s in results.values() if not np.isnan(s)]
        max_skewness = max(abs(s) for s in valid_skewness) if valid_skewness else 0.0

        return SkewnessMetrics(
            column_scores=results,
            max_skewness=float(max_skewness),
            severe_columns=severe_columns,
        )
