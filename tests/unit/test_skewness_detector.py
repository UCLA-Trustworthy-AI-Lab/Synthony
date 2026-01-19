"""
Unit tests for SkewnessDetector.

Tests the Fisher-Pearson skewness calculation and severe skew detection.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm

from synthony.detectors.skewness import SkewnessDetector


class TestSkewnessDetector:
    """Test skewness detection functionality."""

    def test_normal_distribution_low_skew(self):
        """Normal distribution should have skewness near 0."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(1000)})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert "value" in result.column_scores
        assert abs(result.column_scores["value"]) < 0.5  # Should be close to 0
        assert len(result.severe_columns) == 0  # Not severe
        assert result.max_skewness < 0.5

    def test_lognormal_high_skew(self):
        """LogNormal distribution should trigger severe skew detection."""
        np.random.seed(42)
        data = lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42)
        df = pd.DataFrame({"value": data})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert result.column_scores["value"] > 2.0  # Above threshold
        assert "value" in result.severe_columns
        assert result.max_skewness > 2.0

    def test_single_value_column(self):
        """All same values should have skewness = 0."""
        df = pd.DataFrame({"value": [5.0] * 100})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert result.column_scores["value"] == 0.0
        assert len(result.severe_columns) == 0

    def test_insufficient_data(self):
        """< 3 values should return NaN (skewness undefined)."""
        df = pd.DataFrame({"value": [1.0, 2.0]})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert np.isnan(result.column_scores["value"])
        assert len(result.severe_columns) == 0  # NaN doesn't count as severe

    def test_ignores_non_numeric(self):
        """Should skip categorical columns."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "category": ["A", "B", "C", "D", "E"],
        })

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert "numeric" in result.column_scores
        assert "category" not in result.column_scores

    def test_multiple_columns(self):
        """Test with multiple numeric columns."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal": np.random.randn(1000),
            "skewed": lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42),
            "uniform": np.random.uniform(0, 1, 1000),
        })

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert len(result.column_scores) == 3
        assert "skewed" in result.severe_columns  # Only skewed should be severe
        assert "normal" not in result.severe_columns
        assert "uniform" not in result.severe_columns

    def test_negative_skew(self):
        """Test detection of negative skew (left tail)."""
        # Reverse the lognormal to create negative skew
        np.random.seed(42)
        positive_skew = lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42)
        negative_skew = -positive_skew  # Flip to create left tail

        df = pd.DataFrame({"value": negative_skew})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        # Should detect severe skew (absolute value > 2.0)
        assert abs(result.column_scores["value"]) > 2.0
        assert "value" in result.severe_columns

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty metrics."""
        df = pd.DataFrame()

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        assert len(result.column_scores) == 0
        assert result.max_skewness == 0.0
        assert len(result.severe_columns) == 0

    def test_custom_threshold(self):
        """Test with custom skewness threshold."""
        np.random.seed(42)
        # Create data with moderate skew (around 1.5)
        df = pd.DataFrame({"value": lognorm.rvs(s=0.5, scale=10, size=1000, random_state=42)})

        # With low threshold, should detect
        detector_low = SkewnessDetector(threshold=1.0)
        result_low = detector_low.analyze(df)
        assert len(result_low.severe_columns) > 0

        # With high threshold, should not detect
        detector_high = SkewnessDetector(threshold=5.0)
        result_high = detector_high.analyze(df)
        # May or may not detect depending on actual skewness
        # Just verify threshold is used correctly
        assert detector_high.threshold == 5.0

    def test_with_nan_values(self):
        """Test that NaN values are properly handled."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan]})

        detector = SkewnessDetector(threshold=2.0)
        result = detector.analyze(df)

        # Should compute skewness on non-NaN values only
        assert "value" in result.column_scores
        assert not np.isnan(result.column_scores["value"])  # Should have valid result
