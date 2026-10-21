"""
Pytest configuration and fixtures for Synthony tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm

from synthony.llm import LLMClient, LLMResponse, ToolCall
from synthony.runners.base import SynthesisJobResult, SynthesizerRunner


def submit_recommendation_call(call_id: str = "submit", model_name: str = "CTGAN") -> ToolCall:
    """A valid `submit_recommendation` tool call, for scripting FakeLLMClient
    in agent/orchestrator/manager tests. Centralized since the same shape is
    needed across test_recommender_agent.py, test_orchestrator.py,
    test_run_manager.py, and the functional /runs tests.
    """
    return ToolCall(
        id=call_id,
        name="submit_recommendation",
        arguments={
            "ranked_candidates": [
                {
                    "model_name": model_name,
                    "confidence": 0.8,
                    "hyperparameter_hints": {},
                    "rationale": "ok",
                }
            ],
            "reasoning": "test",
        },
    )


def happy_path_llm_responses() -> list[LLMResponse]:
    """A scripted 3-turn happy path: profile -> rule-based anchor -> submit."""
    return [
        LLMResponse(
            tool_calls=[ToolCall(id="c1", name="get_profile_overview", arguments={})],
            stop_reason="tool_use",
        ),
        LLMResponse(
            tool_calls=[ToolCall(id="c2", name="get_rule_based_scores", arguments={})],
            stop_reason="tool_use",
        ),
        LLMResponse(tool_calls=[submit_recommendation_call()], stop_reason="tool_use"),
    ]


class FakeLLMClient(LLMClient):
    """Scripted LLMClient for tests: returns responses from `responses` in
    order, one per `complete()` call, repeating the last one once the
    script is exhausted (so a test with an imprecise turn count degrades to
    "model won't stop" instead of crashing with an IndexError).

    Backward compatible with the original single-fixed-response usage:
    `FakeLLMClient(LLMResponse(...))` still works via the isinstance branch.

    Centralized here (was previously defined only in test_llm_clients.py)
    since Phase 3's agent/orchestrator tests need a multi-turn scripted
    fake, not just a single fixed response.
    """

    provider = "fake"
    model = "fake-model"

    def __init__(self, responses: LLMResponse | list[LLMResponse]):
        self._responses: list[LLMResponse] = (
            [responses] if isinstance(responses, LLMResponse) else list(responses)
        )
        self.calls: list[tuple[list, dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeRunner(SynthesizerRunner):
    """Scripted SynthesizerRunner for Phase 4 tests -- no real subprocess,
    no real synthcity. ``outcomes`` is a list popped in call order
    (repeating the last once exhausted, mirroring ``FakeLLMClient``), each
    one of:

    - ``"success"``     -- writes the training CSV back out verbatim as the
                            "synthetic" parquet, so a real
                            ``DataQualityBenchmark.compare`` scores it near
                            1.0 (comfortably above any realistic threshold).
    - ``"low_quality"``  -- writes a heavily-corrupted copy (numeric columns
                             replaced with unrelated noise), so a real
                             compare scores it well below a 0.75 threshold.
    - ``"fail"``         -- returns ``ok=False``, no output written.

    Using the *real* ``DataQualityBenchmark`` against a controlled-quality
    fake synthetic file (rather than mocking the benchmark too) keeps these
    tests honest about what "met_threshold" actually depends on.
    """

    def __init__(self, outcomes: str | list[str] = "success"):
        self._outcomes: list[str] = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        self.calls: list = []
        self.cancel_events_seen: list = []

    def supported_models(self) -> frozenset:
        return frozenset({"CTGAN", "TVAE", "ARF"})

    async def train_and_generate(self, job, cancel_event=None):
        self.calls.append(job)
        self.cancel_events_seen.append(cancel_event)
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]

        if outcome == "fail":
            return SynthesisJobResult(
                ok=False,
                error="synthetic training failure (scripted)",
                train_seconds=0.0,
                generate_seconds=0.0,
                output_path=None,
            )

        df = pd.read_csv(job.train_csv_path)
        if outcome == "low_quality":
            df = df.copy()
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = np.random.RandomState(0).uniform(
                -1000, 1000, size=df[numeric_cols].shape
            )
        df.to_parquet(job.output_path)
        return SynthesisJobResult(
            ok=True,
            error=None,
            train_seconds=0.01,
            generate_seconds=0.01,
            output_path=job.output_path,
        )


@pytest.fixture
def isolated_db():
    """Point database.py's global connection at a fresh temp SQLite file,
    restoring the prior state afterward.

    Centralized here (was previously copy-pasted identically across
    test_database_sessions.py, test_run_persistence.py, and
    test_step_recorder.py) since Phase 3 adds enough new test files that
    need it to make a 4th+ copy not worth it.
    """
    from synthony.api import database as db_module
    from synthony.api.database import init_database

    prior_engine, prior_session_local = db_module._engine, db_module._SessionLocal
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        init_database(f"sqlite:///{db_path}")
        yield
    db_module._engine, db_module._SessionLocal = prior_engine, prior_session_local


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
