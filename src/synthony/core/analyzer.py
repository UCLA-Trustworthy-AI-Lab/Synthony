from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from synthony.core.loaders import DataLoader
from synthony.core.schemas import DatasetProfile, StressFactors
from synthony.detectors.cardinality import CardinalityDetector
from synthony.detectors.correlation import CorrelationDetector
from synthony.detectors.data_size import DataSizeClassifier
from synthony.detectors.skewness import SkewnessDetector
from synthony.utils.constants import AnalyzerConfig, DEFAULT_CONFIG
from synthony.core.errors import ValidationError

class StochasticDataAnalyzer:
    """
    Analyzes tabular data to detect statistical "stress factors" that
    indicate difficulty for synthesis models. Integrates all detectors
    and produces a comprehensive DatasetProfile.

    Example:
        ```python
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze_from_file("data.csv")
        print(f"Severe skew: {profile.stress_factors.severe_skew}")
        print(f"Zipfian: {profile.stress_factors.zipfian_distribution}")

        profile_json = profile.to_json()        
        ```
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize analyzer with optional threshold overrides.

        Args:
            config: Custom configuration with threshold values.
                   If None, uses DEFAULT_CONFIG from constants.py.
        """
        self.config = config or DEFAULT_CONFIG

        # Initialize detectors with config thresholds
        self.skewness_detector = SkewnessDetector(
            threshold=self.config.skewness_threshold
        )
        self.cardinality_detector = CardinalityDetector(
            cardinality_threshold=self.config.cardinality_threshold,
            zipfian_ratio=self.config.zipfian_ratio,
        )
        self.correlation_detector = CorrelationDetector(
            density_threshold=self.config.correlation_density_threshold,
            correlation_threshold=self.config.correlation_strength_threshold,
            r_squared_threshold=self.config.r2_threshold,
            max_columns=self.config.max_correlation_columns,
        )
        self.size_classifier = DataSizeClassifier(
            small_threshold=self.config.small_data_threshold,
            large_threshold=self.config.large_data_threshold,
        )

    def analyze(self, data: Union[pd.DataFrame, Path, str]) -> DatasetProfile:
        """Comprehensive analysis returning structured profile.

        This is the main entry point. Accepts either a DataFrame or
        a file path (CSV/Parquet), runs all detectors, and returns
        a complete DatasetProfile.

        Args:
            data: DataFrame, file path to CSV/Parquet, or path string

        Returns:
            DatasetProfile with stress factors and detailed metrics

        Raises:
            ValidationError: If data is malformed 
            WrongFormatError: If not in supported format (csv or parquet)
            FileNotFoundError: file not found at path
            ValueError: file is a supported format but not readable or empty
            TypeError: if data is not a DataFrame or file path

        Examples:
            ```python
            # From DataFrame
            df = pd.read_csv("data.csv")
            analyzer = StochasticDataAnalyzer()
            profile = analyzer.analyze(df)

            # From file
            profile = analyzer.analyze("data.csv")
            profile = analyzer.analyze(Path("data.parquet"))

            # With custom config (pass to constructor, not analyze())
            analyzer = StochasticDataAnalyzer(config=AnalyzerConfig(
                skewness_threshold=2.0,
                cardinality_threshold=10,
                zipfian_ratio=0.5,
                small_data_threshold=500,
                large_data_threshold=10000,
            ))
            profile = analyzer.analyze("data.csv")
            ```
        """
        # Handle different input types
        if isinstance(data, (str, Path)):
            df = DataLoader.load(data, validate=True)
        elif isinstance(data, pd.DataFrame):
            # Validate DataFrame
            validation_result = DataLoader.validate_dataframe(data)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"DataFrame validation failed: {', '.join(validation_result.errors)}"
                )
            df = data
        else:
            raise TypeError(
                f"Expected DataFrame, Path, or str, got {type(data).__name__}"
            )

        # Extract basic metadata
        row_count = len(df)
        column_count = len(df.columns)

        # Run all detectors
        skewness_metrics = self.skewness_detector.analyze(df)
        cardinality_metrics, zipfian_metrics = self.cardinality_detector.analyze(df)
        correlation_metrics = self.correlation_detector.analyze(df)
        size_metrics = self.size_classifier.classify(row_count)

        # Stress factor summary
        stress_factors = StressFactors(
            severe_skew=skewness_metrics.max_skewness > self.config.skewness_threshold,
            high_cardinality=cardinality_metrics.max_cardinality > self.config.cardinality_threshold,
            zipfian_distribution=zipfian_metrics.detected,
            small_data=size_metrics["small_data"],
            large_data=size_metrics["large_data"],
            higher_order_correlation=correlation_metrics.has_higher_order,
        )

        # Calculate data quality metrics
        null_percentage: Dict[str, float] = {}
        column_types: Dict[str, str] = {}

        for col in df.columns:
            # Null percentage
            null_pct = df[col].isna().sum() / len(df) if len(df) > 0 else 0.0
            null_percentage[col] = float(null_pct)

            # Column type (simplified)
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                column_types[col] = "numeric"
            elif pd.api.types.is_categorical_dtype(dtype):
                column_types[col] = "categorical"
            elif dtype == "object":
                column_types[col] = "object"
            else:
                column_types[col] = str(dtype)

        # Build complete profile
        profile = DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            stress_factors=stress_factors,
            skewness=skewness_metrics,
            cardinality=cardinality_metrics,
            zipfian=zipfian_metrics,
            correlation=correlation_metrics,
            null_percentage=null_percentage,
            column_types=column_types,
            thresholds_used=self.config.to_dict(),
        )

        return profile

    def analyze_from_file(
        self, path: Union[Path, str], file_format: Optional[str] = None
    ) -> DatasetProfile:
        """Load file and analyze (auto-detects format).

        Convenience method that wraps DataLoader.load() and analyze().

        Args:
            path: Path to CSV or Parquet file
            file_format: Optional format override ('csv' or 'parquet').
                        If None, auto-detects from extension.

        Returns:
            DatasetProfile with complete analysis

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If unsupported format
            ValidationError: If data validation fails
        """
        df = DataLoader.load(path, file_format=file_format, validate=True)
        return self.analyze(df)

    @staticmethod
    def from_json(json_str: str) -> DatasetProfile:
        """Deserialize profile from JSON string.

        Args:
            json_str: JSON string or path to JSON file

        Returns:
            DatasetProfile instance

        Example:
            ```python
            # From string
            profile = StochasticDataAnalyzer.from_json(json_str)

            # From file
            json_content = Path("profile.json").read_text()
            profile = StochasticDataAnalyzer.from_json(json_content)
            ```
        """
        return DatasetProfile.model_validate_json(json_str)

    def to_json(self, profile: DatasetProfile, path: Optional[Union[Path, str]] = None) -> str:
        """Serialize profile to JSON (optionally save to file).

        Args:
            profile: DatasetProfile to serialize
            path: Optional file path to save JSON.
                 If None, returns JSON string without saving.

        Returns:
            JSON string representation of profile

        Example:
            ```python
            # Get JSON string
            json_str = analyzer.to_json(profile)

            # Save to file
            analyzer.to_json(profile, "profile.json")
            ```
        """
        # Use the profile's built-in serialization which handles all types correctly
        json_str = profile.model_dump_json(indent=2, exclude={"correlation": {"correlation_matrix"}})

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)

        return json_str

