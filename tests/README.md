# Synthony Test Suite

Comprehensive test suite for the Synthony Data Analysis & Model Recommendation API.

## Quick Start

### Install Test Dependencies

```bash
pip install -e ".[api,llm]"
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests (individual components)
pytest tests/unit/ -v

# Functional tests (API workflows)
pytest tests/functional/ -v

# Integration tests (end-to-end)
pytest tests/integration/ -v

# Regression tests (baseline consistency)
pytest tests/regression/ -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=synthony --cov-report=term-missing --cov-report=html
open htmlcov/index.html  # View HTML report
```

## Test Organization

### Unit Tests (`unit/`)
Tests for individual components in isolation:
- **Detectors**: Skewness, Cardinality, Zipfian, Correlation, Data Size
- **Recommendation Engine**: Scoring logic, constraint filtering, SystemPrompt loading
- **API Components**: Request validation, response formatting

**Run**: `pytest tests/unit/ -v`

### Functional Tests (`functional/`)
Tests for complete user workflows through the API:
- **API Workflows**: Health checks, analysis, recommendations
- **Recommendation Methods**: Rule-based, LLM, Hybrid comparisons
- **Constraint Scenarios**: CPU-only, DP requirements

**Run**: `pytest tests/functional/ -v`

### Integration Tests (`integration/`)
Tests for end-to-end scenarios:
- **Benchmark Datasets**: Long Tail, Needle in Haystack, Small Data Trap
- **Real-world Scenarios**: Production workflows, exploratory analysis
- **Edge Cases**: Very small datasets, single columns, all categorical

**Run**: `pytest tests/integration/ -v`

### Regression Tests (`regression/`)
Tests to prevent regressions:
- **Baseline Recommendations**: Consistent results for known datasets
- **Schema Consistency**: API response format stability
- **Backward Compatibility**: No breaking changes

**Run**: `pytest tests/regression/ -v`

## Test Markers

Use pytest markers to filter tests:

```bash
# Run only unit tests
pytest -m unit

# Run only slow tests
pytest -m slow

# Run tests that don't require LLM
pytest -m "not requires_llm"

# Run specific combinations
pytest -m "unit and not slow"
```

Available markers:
- `unit`: Unit tests
- `functional`: Functional tests
- `integration`: Integration tests
- `regression`: Regression tests
- `slow`: Tests that take longer to run
- `requires_llm`: Tests that need LLM API access

## Parallel Testing

Run tests in parallel for faster execution:

```bash
pytest tests/ -n auto  # Auto-detect CPU cores
pytest tests/ -n 4     # Use 4 workers
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
# Use in your tests
def test_example(sample_normal_df, sample_skewed_df):
    # sample_normal_df: Normal distribution DataFrame
    # sample_skewed_df: Heavily skewed DataFrame
    pass
```

Available fixtures:
- `sample_normal_df`: Normal distribution (1000 rows)
- `sample_skewed_df`: LogNormal distribution (high skew)
- `sample_zipfian_df`: Zipfian distribution (top 20% = 90%)
- `sample_small_df`: Small dataset (200 rows)
- `temp_csv_file`: Temporary CSV file
- `temp_parquet_file`: Temporary Parquet file

## Writing New Tests

### Unit Test Example

```python
# tests/unit/test_my_component.py
import pytest
from synthony.my_module import MyComponent

class TestMyComponent:
    def test_basic_functionality(self):
        """Test basic component behavior."""
        component = MyComponent(threshold=10)
        result = component.process(data)

        assert result.status == "success"
        assert result.value > 0
```

### Functional Test Example

```python
# tests/functional/test_my_workflow.py
from fastapi.testclient import TestClient
from synthony.api.server import app

def test_complete_workflow():
    """Test complete user workflow."""
    client = TestClient(app)

    # Step 1: Upload data
    response = client.post("/analyze", files={"file": ...})
    assert response.status_code == 200

    # Step 2: Get recommendation
    response = client.post("/recommend", json={...})
    assert response.status_code == 200
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[api,llm]"
          pip install -r tests/requirements-test.txt

      - name: Run tests
        run: pytest tests/ --cov=synthony --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Detectors | >90% |
| Recommendation Engine | >90% |
| API Endpoints | >95% |
| Error Handling | >80% |
| **Overall** | **>90%** |

## Troubleshooting

### Tests Fail Due to Missing API Server

Some integration/functional tests require the API server to be running:

```bash
# Terminal 1: Start API server
python start_api.py

# Terminal 2: Run tests
pytest tests/functional/ -v
```

### Tests Fail Due to Missing LLM

Tests marked with `requires_llm` need LLM API access:

```bash
# Skip LLM tests
pytest -m "not requires_llm"

# Or set API keys
export VLLM_API_KEY="your-key"
export VLLM_URL="http://your-server:8000/v1/"
pytest tests/
```

### Import Errors

Ensure the package is installed in development mode:

```bash
pip install -e .
```

## Additional Resources

- **Test Summary**: See `TEST_SUITE_SUMMARY.md` for detailed overview
- **API Documentation**: See `docs/API_USAGE.md`
- **Architecture**: See `docs/architecture_v2.md`

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure >90% coverage for new code
3. Run full test suite before submitting PR
4. Update this README if adding new test categories

---

**Last Updated**: 2026-01-16
**Test Suite Version**: 0.1.0
