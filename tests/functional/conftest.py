"""
Shared fixtures for functional tests.

Some files in this directory are manual live-server scripts (not pytest tests).
They require a running server at localhost:8000 and local dataset files.
They are excluded from automated CI collection via collect_ignore.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from synthony.api.server import app

collect_ignore = [
    "test_api_advanced.py",
    "test_llm_with_systemprompt.py",
    "test_api_simple.py",
]


@pytest.fixture
def client():
    """Create test client for API with startup events triggered."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for upload."""
    df = pd.DataFrame({
        "id": range(1000),
        "value": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return ("test.csv", csv_buffer, "text/csv")
