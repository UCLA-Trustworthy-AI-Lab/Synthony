"""
Data loading and validation utilities.

Handles CSV and Parquet file loading with comprehensive validation
to ensure data quality before analysis.
"""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from synthony.core.errors import ValidationError


class ValidationResult:
    """Results of dataframe validation checks."""

    def __init__(
        self,
        is_valid: bool,
        errors: list[str] | None = None,
        warnings_list: list[str] | None = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings_list = warnings_list or []

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        if self.is_valid:
            return "Validation passed"
        return f"Validation failed: {', '.join(self.errors)}"


class DataLoader:
    """Handles file I/O with validation for tabular data.

    Supports CSV and Parquet formats with automatic format detection
    based on file extension.
    """

    @staticmethod
    def auto_detect_format(path: Path) -> str:
        """Detect file format from extension.

        Args:
            path: Path to data file

        Returns:
            Format string: 'csv' or 'parquet'

        Raises:
            ValueError: If format is not supported
        """
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix in [".parquet", ".pq"]:
            return "parquet"
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported formats: .csv, .parquet, .pq"
            )

    @staticmethod
    def load_csv(
        path: Path, validate: bool = True, **kwargs: Any
    ) -> pd.DataFrame:
        """Load CSV file with validation.

        Args:
            path: Path to CSV file
            validate: Whether to validate the dataframe after loading
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            Loaded and optionally validated DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If validation fails
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Load CSV
        df = pd.read_csv(path, **kwargs)

        if validate:
            validation_result = DataLoader.validate_dataframe(df)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"CSV validation failed: {', '.join(validation_result.errors)}"
                )

            # Show warnings
            for warning in validation_result.warnings_list:
                warnings.warn(warning)

        return df

    @staticmethod
    def load_parquet(
        path: Path, validate: bool = True, **kwargs: Any
    ) -> pd.DataFrame:
        """Load Parquet file with validation.

        Args:
            path: Path to Parquet file
            validate: Whether to validate the dataframe after loading
            **kwargs: Additional arguments passed to pd.read_parquet

        Returns:
            Loaded and optionally validated DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If validation fails
        """
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        # Use pyarrow engine explicitly for consistency
        kwargs.setdefault("engine", "pyarrow")

        # Load Parquet
        df = pd.read_parquet(path, **kwargs)

        if validate:
            validation_result = DataLoader.validate_dataframe(df)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Parquet validation failed: {', '.join(validation_result.errors)}"
                )

            # Show warnings
            for warning in validation_result.warnings_list:
                warnings.warn(warning)

        return df

    @staticmethod
    def load(
        path: str | Path,
        file_format: str | None = None,
        validate: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data file with automatic format detection.

        Args:
            path: Path to data file (CSV or Parquet)
            file_format: Optional format override ('csv' or 'parquet')
                        If None, auto-detects from file extension
            validate: Whether to validate the dataframe after loading
            **kwargs: Additional arguments passed to the loader

        Returns:
            Loaded and optionally validated DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
            ValidationError: If validation fails
        """
        path = Path(path)

        # Determine format
        if file_format is None:
            file_format = DataLoader.auto_detect_format(path)
        else:
            file_format = file_format.lower()

        # Load based on format
        if file_format == "csv":
            return DataLoader.load_csv(path, validate=validate, **kwargs)
        elif file_format == "parquet":
            return DataLoader.load_parquet(path, validate=validate, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: {file_format}. Use 'csv' or 'parquet'"
            )

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
        """Validate dataframe for analysis.

        Checks for common data quality issues:
        - Empty dataframe
        - All-NaN columns
        - Null column names
        - Inconsistent types

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        # Check if empty
        if df.empty or df.shape[0] == 0:
            errors.append("DataFrame is empty (zero rows)")
            return ValidationResult(is_valid=False, errors=errors)

        if df.shape[1] == 0:
            errors.append("DataFrame has no columns")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for null column names
        null_col_mask = df.columns.isna()
        if null_col_mask.any():
            errors.append(f"Found {null_col_mask.sum()} columns with null names")

        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"Duplicate column names found: {duplicate_cols}")

        # Check for all-NaN columns
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            warnings_list.append(
                f"Columns with all NaN values will be excluded from analysis: {all_nan_cols}"
            )

        # Check for mostly-null columns (>95% null)
        mostly_null_cols = []
        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > 0.95:
                mostly_null_cols.append(f"{col} ({null_pct*100:.1f}% null)")

        if mostly_null_cols:
            warnings_list.append(
                f"Columns with >95% null values: {mostly_null_cols}"
            )

        # Validation passed if no errors
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings_list=warnings_list
        )
