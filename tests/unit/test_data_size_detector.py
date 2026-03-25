"""
Unit tests for DataSizeClassifier.

Tests small data (<500 rows) and large data (>50k rows) detection.
"""


from synthony.detectors.data_size import DataSizeClassifier


class TestDataSizeClassifier:
    """Test data size classification functionality."""

    def test_small_data_detection(self):
        """Data with <500 rows should be flagged as small."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(200)

        assert result["small_data"] is True
        assert result["large_data"] is False

    def test_large_data_detection(self):
        """Data with >50k rows should be flagged as large."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(60000)

        assert result["small_data"] is False
        assert result["large_data"] is True

    def test_medium_data(self):
        """Data between thresholds should not be flagged."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(5000)

        assert result["small_data"] is False
        assert result["large_data"] is False

    def test_exact_small_threshold(self):
        """Data with exactly 500 rows should not be small."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(500)

        # 500 is not < 500, so not small
        assert result["small_data"] is False

    def test_exact_large_threshold(self):
        """Data with exactly 50000 rows should not be large."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(50000)

        # 50000 is not > 50000, so not large
        assert result["large_data"] is False

    def test_very_small_data(self):
        """Test with very small dataset (< 10 rows)."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(5)

        assert result["small_data"] is True

    def test_empty_dataframe(self):
        """Empty DataFrame (0 rows) should be flagged as small."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)
        result = classifier.classify(0)

        assert result["small_data"] is True
        assert result["large_data"] is False

    def test_custom_thresholds(self):
        """Test with custom threshold parameters."""
        row_count = 300

        # With low small threshold, should not be small
        classifier_low = DataSizeClassifier(small_threshold=100, large_threshold=50000)
        result_low = classifier_low.classify(row_count)
        assert result_low["small_data"] is False

        # With high small threshold, should be small
        classifier_high = DataSizeClassifier(small_threshold=1000, large_threshold=50000)
        result_high = classifier_high.classify(row_count)
        assert result_high["small_data"] is True

    def test_classification_edge_cases(self):
        """Test edge cases around threshold boundaries."""
        classifier = DataSizeClassifier(small_threshold=500, large_threshold=50000)

        # Just below small threshold
        result_499 = classifier.classify(499)
        assert result_499["small_data"] is True

        # Just at small threshold
        result_500 = classifier.classify(500)
        assert result_500["small_data"] is False

        # Just above large threshold
        result_50001 = classifier.classify(50001)
        assert result_50001["large_data"] is True

        # Just at large threshold
        result_50000 = classifier.classify(50000)
        assert result_50000["large_data"] is False
