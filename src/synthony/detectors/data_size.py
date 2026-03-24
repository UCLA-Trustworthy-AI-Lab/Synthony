"""
Data size classification.

Classifies datasets as small or large based on row count thresholds.
"""



class DataSizeClassifier:
    """Classify dataset size as small/normal/large.

    Small datasets (<1,000 rows) risk overfitting without tree-based models.
    Large datasets (>50k rows) make LLM-based synthesis impractical due to
    context window limitations and computational cost.
    """

    def __init__(
            self,
            small_threshold: int = 1000,
            large_threshold: int = 50000
            ):
        """Initialize data size classifier.

        Args:
            small_threshold: Row count below which to flag as "small data".
                           Default 1,000 from architecture docs.
            large_threshold: Row count above which to flag as "large data".
                           Default 50,000 from architecture docs.
        """
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold

    def classify(self, row_count: int) -> dict[str, bool]:
        """Simple threshold-based classification.

        Args:
            row_count: Number of rows in the dataset

        Returns:
            Dictionary with boolean flags:
            - small_data: True if < small_threshold (default 1,000)
            - large_data: True if > large_threshold (default 50,000)

        Note:
            Both flags can be False (normal-sized dataset).
            Both flags cannot be True simultaneously.
        """
        small_data = row_count < self.small_threshold
        large_data = row_count > self.large_threshold

        return {"small_data": small_data, "large_data": large_data}
