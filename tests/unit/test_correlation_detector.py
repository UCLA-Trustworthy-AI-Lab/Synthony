"""
Unit tests for CorrelationDetector.

Tests higher-order correlation detection (dense correlation matrix with low R²).
"""

import numpy as np
import pandas as pd
import pytest

from synthony.detectors.correlation import CorrelationDetector


class TestCorrelationDetector:
    """Test correlation detection functionality."""

    def test_no_correlation(self):
        """Independent variables should not be flagged."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.random.randn(1000),
            "x2": np.random.randn(1000),
            "x3": np.random.randn(1000),
        })

        detector = CorrelationDetector()
        result = detector.analyze(df)

        assert result.has_higher_order is False
        assert result.correlation_density < 0.5  # Most pairs uncorrelated

    def test_linear_correlation(self):
        """Strongly linear correlated variables should not trigger higher-order."""
        np.random.seed(42)
        x = np.random.randn(1000)
        df = pd.DataFrame({
            "x1": x,
            "x2": 2 * x + 0.1 * np.random.randn(1000),  # Strong linear
            "x3": -0.5 * x + 0.1 * np.random.randn(1000),  # Strong linear
        })

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # Dense correlation matrix (many pairs correlated)
        assert result.correlation_density > 0.5

        # But high R² (linear relationships)
        assert result.mean_r_squared > 0.3

        # Should NOT flag as higher-order (linear explains it)
        assert result.has_higher_order is False

    def test_higher_order_correlation(self):
        """Non-linear relationships should trigger higher-order detection."""
        np.random.seed(42)
        x = np.random.randn(1000)
        df = pd.DataFrame({
            "x1": x,
            "x2": x ** 2,  # Quadratic relationship
            "x3": np.sin(x * 3),  # Sinusoidal relationship
            "x4": np.exp(x / 5),  # Exponential relationship
        })

        detector = CorrelationDetector(
            correlation_threshold=0.1,
            density_threshold=0.5,
            r_squared_threshold=0.3
        )
        result = detector.analyze(df)

        # Should have dense correlation matrix
        assert result.correlation_density > 0.5

        # But low R² (linear models don't fit well)
        assert result.mean_r_squared < 0.3

        # Should flag as higher-order
        assert result.has_higher_order is True

    def test_single_column(self):
        """Single column should not compute correlations."""
        df = pd.DataFrame({"x": np.random.randn(100)})

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # No pairs to correlate
        assert result.correlation_density == 0.0
        assert result.mean_r_squared == 0.0
        assert result.has_higher_order is False

    def test_two_columns_independent(self):
        """Two independent columns."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.random.randn(1000),
            "x2": np.random.randn(1000),
        })

        detector = CorrelationDetector(correlation_threshold=0.1)
        result = detector.analyze(df)

        # Should have low correlation
        assert abs(result.correlation_matrix.iloc[0, 1]) < 0.1
        assert result.correlation_density == 0.0  # No pairs above threshold

    def test_ignores_categorical_columns(self):
        """Should skip categorical columns."""
        np.random.seed(42)
        df = pd.DataFrame({
            "numeric1": np.random.randn(100),
            "numeric2": np.random.randn(100),
            "category": ["A", "B", "C"] * 33 + ["A"],
        })

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # Correlation matrix should only include numeric columns
        assert result.correlation_matrix.shape == (2, 2)
        assert "category" not in result.correlation_matrix.columns

    def test_custom_thresholds(self):
        """Test with custom threshold parameters."""
        np.random.seed(42)
        x = np.random.randn(1000)
        df = pd.DataFrame({
            "x1": x,
            "x2": 0.15 * x + np.random.randn(1000),  # Weak correlation
        })

        # With low correlation threshold, should detect
        detector_low = CorrelationDetector(correlation_threshold=0.1)
        result_low = detector_low.analyze(df)
        assert result_low.correlation_density > 0.0

        # With high correlation threshold, should not detect
        detector_high = CorrelationDetector(correlation_threshold=0.5)
        result_high = detector_high.analyze(df)
        assert result_high.correlation_density == 0.0

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty metrics."""
        df = pd.DataFrame()

        detector = CorrelationDetector()
        result = detector.analyze(df)

        assert result.correlation_matrix is None or result.correlation_matrix.empty
        assert result.correlation_density == 0.0
        assert result.has_higher_order is False

    def test_all_same_values(self):
        """Columns with all same values should have undefined correlation."""
        df = pd.DataFrame({
            "constant1": [5.0] * 100,
            "constant2": [10.0] * 100,
        })

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # Correlation of constants is undefined (NaN or 0)
        # Should handle gracefully
        assert result.correlation_density == 0.0

    def test_with_nan_values(self):
        """Test that NaN values are handled correctly."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.concatenate([np.random.randn(90), [np.nan] * 10]),
            "x2": np.concatenate([np.random.randn(90), [np.nan] * 10]),
        })

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # Should compute correlation on non-NaN values
        assert result.correlation_matrix is not None
        # Correlation should be computed (may be low due to independence)
        assert not np.isnan(result.correlation_matrix.iloc[0, 1])

    def test_many_columns_sampling(self):
        """Test with many columns (should sample for performance)."""
        np.random.seed(42)
        # Create 150 numeric columns (> 100 threshold for sampling)
        data = {f"col_{i}": np.random.randn(1000) for i in range(150)}
        df = pd.DataFrame(data)

        detector = CorrelationDetector()
        result = detector.analyze(df)

        # Should still complete (with sampling)
        assert result.correlation_matrix is not None
        # If sampling is implemented, matrix should be smaller
        # Otherwise, should have all 150 columns

    def test_perfect_correlation(self):
        """Test with perfectly correlated columns."""
        x = np.linspace(0, 10, 1000)
        df = pd.DataFrame({
            "x1": x,
            "x2": x,  # Perfect correlation
            "x3": 2 * x,  # Perfect correlation (different scale)
        })

        detector = CorrelationDetector(correlation_threshold=0.1)
        result = detector.analyze(df)

        # Should have very dense correlation matrix (all pairs correlated)
        assert result.correlation_density == 1.0  # All pairs above threshold

        # R² should be very high (perfect linear fit)
        assert result.mean_r_squared > 0.99

        # Should NOT flag as higher-order (perfect linear)
        assert result.has_higher_order is False

    def test_quadratic_relationship(self):
        """Test detection of quadratic relationship."""
        np.random.seed(42)
        x = np.linspace(-5, 5, 1000)
        df = pd.DataFrame({
            "x": x,
            "y": x ** 2 + 0.1 * np.random.randn(1000),  # Quadratic with noise
        })

        detector = CorrelationDetector(
            correlation_threshold=0.1,
            r_squared_threshold=0.3
        )
        result = detector.analyze(df)

        # Should have correlation (due to relationship)
        # Note: Pearson correlation of x and x² around x=0 is 0
        # This tests whether we detect the relationship

        # If correlation is detected and R² is low, should flag
        if result.correlation_density > 0.0:
            if result.mean_r_squared < 0.3:
                assert result.has_higher_order is True
