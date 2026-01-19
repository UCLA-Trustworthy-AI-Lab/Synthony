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
