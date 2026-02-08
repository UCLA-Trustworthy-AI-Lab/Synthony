"""
Pytest configuration and fixtures for Synthony tests.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import lognorm


@pytest.fixture
def sample_normal_df():
    """Normal distribution DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "value1": np.random.randn(1000),
        "value2": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000)
    })


@pytest.fixture
def sample_skewed_df():
    """Heavily skewed DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "skewed_value": lognorm.rvs(s=0.95, scale=np.exp(5), size=1000, random_state=42)
    })


@pytest.fixture
def sample_zipfian_df():
    """Zipfian distribution DataFrame for testing."""
    # Top 2 categories (20% of 10) = 90% of data
    categories = ["top1"] * 450 + ["top2"] * 450 + [f"rare_{i}" for i in range(100)]
    return pd.DataFrame({"category": categories})


@pytest.fixture
def sample_small_df():
    """Small dataset (< 500 rows) for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "value": np.random.randn(200),
        "category": np.random.choice(["A", "B"], 200)
    })


@pytest.fixture
def temp_csv_file(tmp_path, sample_normal_df):
    """Create temporary CSV file for testing."""
    csv_path = tmp_path / "test.csv"
    sample_normal_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_parquet_file(tmp_path, sample_normal_df):
    """Create temporary Parquet file for testing."""
    parquet_path = tmp_path / "test.parquet"
    sample_normal_df.to_parquet(parquet_path, index=False)
    return parquet_path


# ============================================================
# Real Dataset Fixtures (with .env support)
# ============================================================

def get_test_data_paths():
    """
    Get input/output data paths from .env or use hardcoded fallbacks.
    
    Priority:
    1. Environment variables (INPUT_TESTDATA, OUTPUT_TESTDATA)
    2. Hardcoded fallback paths (./dataset/input_data/)
    """
    import os
    from dotenv import load_dotenv
    
    # Load .env file from project root
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    
    # Get paths with fallbacks (strip whitespace from env values)
    input_env = os.getenv("INPUT_TESTDATA", "").strip().strip("'\"")
    output_env = os.getenv("OUTPUT_TESTDATA", "").strip().strip("'\"")
    
    # Default fallback paths
    default_input = project_root / "dataset" / "input_data"
    default_output = project_root / "output" / "tests"
    
    # Resolve paths - if env var is set and path exists, use it; otherwise fallback
    if input_env:
        input_path = Path(input_env)
        # Make relative paths relative to project root
        if not input_path.is_absolute():
            input_path = project_root / input_path
        # If env path doesn't exist, fall back to default
        if not input_path.exists():
            input_path = default_input
    else:
        input_path = default_input
    
    if output_env:
        output_path = Path(output_env)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = default_output
    
    return input_path, output_path


@pytest.fixture(scope="session")
def test_data_paths():
    """Get test data input/output paths."""
    input_path, output_path = get_test_data_paths()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return {"input": input_path, "output": output_path}


@pytest.fixture(scope="session")
def real_csv_small(test_data_paths):
    """
    Load a small real CSV file for testing.
    Uses IndianLiverPatient.csv (~28KB, 583 rows).
    """
    csv_path = test_data_paths["input"] / "IndianLiverPatient.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def real_csv_medium(test_data_paths):
    """
    Load a medium real CSV file for testing.
    Uses insurance.csv (~55KB, 1338 rows).
    """
    csv_path = test_data_paths["input"] / "insurance.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def real_csv_path_small(test_data_paths):
    """Return path to small real CSV file (IndianLiverPatient.csv)."""
    csv_path = test_data_paths["input"] / "IndianLiverPatient.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return csv_path


@pytest.fixture(scope="session")
def real_csv_path_medium(test_data_paths):
    """Return path to medium real CSV file (insurance.csv)."""
    csv_path = test_data_paths["input"] / "insurance.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return csv_path


@pytest.fixture
def test_output_dir(test_data_paths, tmp_path):
    """Get test output directory (uses tmp_path for isolation)."""
    return tmp_path
