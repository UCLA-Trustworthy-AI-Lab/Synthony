"""
Unit tests for CardinalityDetector.

Tests high cardinality and Zipfian distribution detection.
"""

import numpy as np
import pandas as pd

from synthony.detectors.cardinality import CardinalityDetector


class TestCardinalityDetector:
    """Test cardinality detection functionality."""

    def test_low_cardinality(self):
        """Low cardinality columns should not be flagged."""
        df = pd.DataFrame({
            "category": ["A", "B", "C"] * 100  # Only 3 unique values
        })

        detector = CardinalityDetector(cardinality_threshold=500)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert "category" in cardinality_metrics.column_counts
        assert cardinality_metrics.column_counts["category"] == 3
        assert "category" not in cardinality_metrics.high_cardinality_columns
        assert cardinality_metrics.max_cardinality == 3

    def test_high_cardinality(self):
        """High cardinality columns should be flagged."""
        # Create column with 600 unique values (> 500 threshold)
        df = pd.DataFrame({
            "user_id": range(600)
        })

        detector = CardinalityDetector(cardinality_threshold=500)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert cardinality_metrics.column_counts["user_id"] == 600
        assert "user_id" in cardinality_metrics.high_cardinality_columns
        assert cardinality_metrics.max_cardinality == 600

    def test_zipfian_distribution(self):
        """Zipfian distribution (top 20% > 80%) should be detected."""
        # Create Zipfian: top 2 categories (20% of 10) = 450+450=900 = 90% of 1000
        categories = (
            ["top1"] * 450 +
            ["top2"] * 450 +
            ["rare_" + str(i) for i in range(100)]
        )
        df = pd.DataFrame({"category": categories})

        detector = CardinalityDetector(zipfian_ratio=0.80)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert "category" in zipfian_metrics.affected_columns
        assert zipfian_metrics.top_20_percent_ratio > 0.80
        assert zipfian_metrics.detected is True

    def test_uniform_distribution_not_zipfian(self):
        """Uniform distribution should not be Zipfian."""
        # Perfectly uniform: 10 categories, 100 each = 1000 total
        categories = []
        for i in range(10):
            categories.extend([f"cat_{i}"] * 100)

        df = pd.DataFrame({"category": categories})

        detector = CardinalityDetector(zipfian_ratio=0.80)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert "category" not in zipfian_metrics.affected_columns
        assert zipfian_metrics.detected is False
        # Top 20% (2 categories) = 200/1000 = 0.2 (not > 0.80)
        # zipfian_metrics.top_20_percent_ratio should be None or < 0.80

    def test_single_category(self):
        """Single category should have cardinality = 1."""
        df = pd.DataFrame({"category": ["A"] * 100})

        detector = CardinalityDetector(cardinality_threshold=500)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert cardinality_metrics.column_counts["category"] == 1
        assert "category" not in cardinality_metrics.high_cardinality_columns

    def test_ignores_numeric_columns(self):
        """Should analyze numeric columns for cardinality only."""
        df = pd.DataFrame({
            "id": range(100),  # Numeric
            "category": ["A", "B", "C"] * 33 + ["A"],
        })

        detector = CardinalityDetector()
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Should analyze both columns for cardinality
        assert "category" in cardinality_metrics.column_counts
        assert "id" in cardinality_metrics.column_counts

        # But only categorical columns for Zipfian
        # (numeric columns won't be in zipfian analysis)

    def test_multiple_columns(self):
        """Test with multiple categorical columns."""
        df = pd.DataFrame({
            "low_card": ["A", "B"] * 500,  # 2 unique values, 1000 rows
            "high_card": [f"val_{i}" for i in range(600)] + [f"val_{i % 100}" for i in range(400)],  # 600 unique
            "zipfian": ["top"] * 900 + [f"rare_{i}" for i in range(100)],
        })

        detector = CardinalityDetector(cardinality_threshold=500, zipfian_ratio=0.80)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Check high cardinality detection
        assert "high_card" in cardinality_metrics.high_cardinality_columns
        assert "low_card" not in cardinality_metrics.high_cardinality_columns

        # Check Zipfian detection (zipfian column should be detected)
        assert "zipfian" in zipfian_metrics.affected_columns

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty metrics."""
        df = pd.DataFrame()

        detector = CardinalityDetector()
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        assert len(cardinality_metrics.column_counts) == 0
        assert len(cardinality_metrics.high_cardinality_columns) == 0
        assert len(zipfian_metrics.affected_columns) == 0
        assert zipfian_metrics.detected is False

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        df = pd.DataFrame({
            "category": range(300)  # 300 unique values
        })

        # With low threshold, should detect as high cardinality
        detector_low = CardinalityDetector(cardinality_threshold=100)
        cardinality_metrics_low, _ = detector_low.analyze(df)
        assert "category" in cardinality_metrics_low.high_cardinality_columns

        # With high threshold, should not detect
        detector_high = CardinalityDetector(cardinality_threshold=1000)
        cardinality_metrics_high, _ = detector_high.analyze(df)
        assert "category" not in cardinality_metrics_high.high_cardinality_columns

    def test_zipfian_edge_case_exactly_threshold(self):
        """Test Zipfian detection at exactly the threshold."""
        # Create distribution where top 20% = exactly 80%
        # 10 categories, top 2 = 400 each, remaining 8 = 25 each
        # Total = 800 + 200 = 1000, ratio = 0.80
        categories = (
            ["top1"] * 400 +
            ["top2"] * 400 +
            sum([[f"cat_{i}"] * 25 for i in range(8)], [])
        )
        df = pd.DataFrame({"category": categories})

        detector = CardinalityDetector(zipfian_ratio=0.80)
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Implementation uses > threshold, not >=, so exactly 0.80 won't trigger
        # (This is correct behavior - need to exceed the threshold)
        assert "category" not in zipfian_metrics.affected_columns

        # But slightly above should trigger
        categories_above = (
            ["top1"] * 405 +
            ["top2"] * 405 +
            sum([[f"cat_{i}"] * 19 for i in range(10)], [])
        )
        df_above = pd.DataFrame({"category": categories_above})
        _, zipfian_metrics_above = detector.analyze(df_above)
        assert "category" in zipfian_metrics_above.affected_columns

    def test_with_nan_values(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame({
            "category": ["A", "B", "C", np.nan, np.nan, np.nan] * 100
        })

        detector = CardinalityDetector()
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Should count unique values (pandas nunique handles NaN)
        assert "category" in cardinality_metrics.column_counts

    def test_all_nan_column(self):
        """Column with all NaN should be handled gracefully."""
        df = pd.DataFrame({
            "all_nan": [np.nan] * 100
        })

        detector = CardinalityDetector()
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Should handle gracefully
        assert "all_nan" in cardinality_metrics.column_counts

    def test_mixed_types_column(self):
        """Test column with mixed types (numeric and string)."""
        df = pd.DataFrame({
            "mixed": ["A", "B", 1, 2, 3.5, "A", "B", 1]
        })

        detector = CardinalityDetector()
        cardinality_metrics, zipfian_metrics = detector.analyze(df)

        # Should count all unique values regardless of type
        assert "mixed" in cardinality_metrics.column_counts
        # Should have A, B, 1, 2, 3.5 = 5 unique values
        assert cardinality_metrics.column_counts["mixed"] == 5
