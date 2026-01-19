"""
Custom exceptions for synthony.

Note: Do NOT redefine Python built-in exceptions (TypeError, ValueError, 
FileNotFoundError). Use them directly or create domain-specific subclasses.
"""


class SynthonyError(Exception):
    """Base exception for all synthony errors."""
    pass


class ValidationError(SynthonyError):
    """Raised when dataset validation fails (empty, malformed, etc)."""
    pass


class UnsupportedFormatError(SynthonyError):
    """Raised when file format is not CSV or Parquet."""
    pass


class ProfileError(SynthonyError):
    """Raised when dataset profiling encounters an error."""
    pass


class ConfigurationError(SynthonyError):
    """Raised when analyzer configuration is invalid."""
    pass