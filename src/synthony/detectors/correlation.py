"""
Higher-order correlation detection.

Identifies dense correlation matrices with low linear R², indicating
non-linear relationships that challenge linear models.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from synthony.core.schemas import CorrelationMetrics


class CorrelationDetector:
    """Detect dense but non-linear correlation matrices.

    This detector identifies datasets where many features are correlated,
    but the relationships are not well-captured by linear models. This
    "higher-order correlation" indicates non-linear dependencies that
    require advanced models.

    The detection works by:
    1. Computing Pearson correlation matrix
    2. Checking if many pairs are correlated (density)
    3. Fitting linear regression to correlated pairs
    4. Checking if linear R² is low despite high correlation
    """

    def __init__(
        self,
        density_threshold: float = 0.5,
        correlation_threshold: float = 0.1,
        r_squared_threshold: float = 0.3,
        max_columns: int = 100,
    ):
        """Initialize correlation detector.

        Args:
            density_threshold: Fraction of pairs with |corr| > correlation_threshold
                             to be considered "dense". Default 0.5.
            correlation_threshold: Minimum |correlation| to count a pair as correlated.
                          Default 0.1.
            r_squared_threshold: Maximum R² below which to flag as non-linear.
                        Default 0.3 (low linear fit despite correlation).
            max_columns: Maximum columns to analyze before sampling.
                        Default 100 (O(n²) complexity management).
        """
        self.density_threshold = density_threshold
        self.correlation_threshold = correlation_threshold
        self.r_squared_threshold = r_squared_threshold
        self.max_columns = max_columns

    def analyze(self, df: pd.DataFrame) -> CorrelationMetrics:
        """Detect higher-order (non-linear) correlations.

        Algorithm:
        1. Compute Pearson correlation matrix for numeric columns
        2. Calculate matrix density:
           - Count pairs with |corr| > threshold (default 0.1)
           - Density = dense_pairs / total_pairs
        3. If density > 0.5 (dense matrix):
           a. For each dense pair (i, j):
              - Fit linear regression: y = mx + b
              - Calculate R² score
           b. Calculate mean R² across all dense pairs
           c. If mean_R² < 0.3: Flag as higher-order correlation
              (Dense correlation but poor linear fit = non-linear)

        Edge Cases:
        - Single numeric column: No correlation, return zeros
        - All correlations weak: density = 0, not higher-order
        - Perfect linear: High R², not flagged as higher-order
        - >100 columns: Sample to reduce O(n²) cost

        Performance: O(n²) for n numeric columns, O(n) with sampling

        Args:
            df: DataFrame to analyze

        Returns:
            CorrelationMetrics with density, R², and detection flag
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Check if we have enough columns for correlation
        if numeric_df.shape[1] < 2:
            return CorrelationMetrics(
                correlation_density=0.0,
                mean_r_squared=0.0,
                has_higher_order=False,
                correlation_matrix=None,
            )

        # Sample columns if too many (performance optimization)
        sampled = False
        if numeric_df.shape[1] > self.max_columns:
            sampled = True
            warnings.warn(
                f"Sampling {self.max_columns} of {numeric_df.shape[1]} numeric columns for correlation analysis (performance optimization)"
            )
            # Use random_state for reproducibility
            numeric_df = numeric_df.sample(n=self.max_columns, axis=1, random_state=42)

        # Compute correlation matrix
        corr_matrix = numeric_df.corr()

        # Calculate matrix density
        # Create boolean mask for correlations above threshold
        mask = np.abs(corr_matrix) > self.correlation_threshold
        # Exclude diagonal (self-correlation) - make a copy to avoid read-only error
        mask_array = mask.values.copy()
        np.fill_diagonal(mask_array, False)
        mask = pd.DataFrame(mask_array, index=mask.index, columns=mask.columns)

        # Count dense pairs (divide by 2 since matrix is symmetric)
        total_pairs = (len(corr_matrix) * (len(corr_matrix) - 1)) / 2
        dense_pairs = mask.sum().sum() / 2
        density = dense_pairs / total_pairs if total_pairs > 0 else 0.0

        # Check for higher-order correlation if matrix is dense
        mean_r2 = 0.0
        higher_order = False

        if density > self.density_threshold:
            r2_scores = []
            cols = numeric_df.columns

            # Iterate through upper triangle of correlation matrix
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    # Only process pairs above correlation threshold
                    if mask.iloc[i, j]:
                        # Get column data, removing NaN pairs
                        col_i = numeric_df[cols[i]]
                        col_j = numeric_df[cols[j]]

                        # Create DataFrame and drop NaN rows
                        pair_df = pd.DataFrame({"x": col_i, "y": col_j}).dropna()

                        # Need sufficient data for regression
                        if len(pair_df) > 10:
                            X = pair_df["x"].values.reshape(-1, 1)
                            y = pair_df["y"].values

                            # Fit linear regression
                            model = LinearRegression()
                            model.fit(X, y)

                            # Calculate R²
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)
                            r2_scores.append(r2)

           # Calculate mean R² if we have scores
            if r2_scores:
                mean_r2 = float(np.mean(r2_scores))
                # Flag as higher-order if R² is low despite high correlation
                higher_order = mean_r2 < self.r_squared_threshold

        return CorrelationMetrics(
            correlation_density=float(density),
            mean_r_squared=float(mean_r2),
            has_higher_order=higher_order,
            correlation_matrix=corr_matrix,
        )
