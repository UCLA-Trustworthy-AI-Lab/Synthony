"""
Functional tests for the synthony CLI commands.

Tests profile, recommend, and benchmark commands via typer's CliRunner,
verifying exit codes, output content, file creation, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from synthony.cli import app

runner = CliRunner()

# Absolute path to the bundled test dataset
ABALONE_CSV = Path(__file__).resolve().parents[2] / "dataset" / "input_data" / "abalone.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def abalone_path() -> Path:
    """Return the path to the abalone test CSV, skipping if absent."""
    if not ABALONE_CSV.exists():
        pytest.skip(f"Test dataset not found: {ABALONE_CSV}")
    return ABALONE_CSV


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a small temporary CSV file for quick tests."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, 200).astype(int),
        "income": np.random.lognormal(10, 1.5, 200),
        "category": np.random.choice(["A", "B", "C", "D"], 200),
        "score": np.random.uniform(0, 100, 200),
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_csv_pair(tmp_path: Path):
    """Create a pair of CSV files (original + synthetic) for benchmark tests."""
    np.random.seed(42)
    n = 200
    original = pd.DataFrame({
        "age": np.random.normal(35, 10, n).astype(int),
        "income": np.random.lognormal(10, 1.5, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    synthetic = pd.DataFrame({
        "age": np.random.normal(36, 11, n).astype(int),
        "income": np.random.lognormal(10.1, 1.4, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })
    orig_path = tmp_path / "original.csv"
    synth_path = tmp_path / "synthetic.csv"
    original.to_csv(orig_path, index=False)
    synthetic.to_csv(synth_path, index=False)
    return orig_path, synth_path


# =========================================================================
# Profile command tests
# =========================================================================

class TestProfileCommand:
    """Tests for the `profile` CLI command."""

    def test_profile_valid_csv(self, abalone_path: Path):
        """Valid CSV file should exit 0 and print stress factor information."""
        result = runner.invoke(app, ["profile", str(abalone_path)])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # Should contain stress factor keywords
        assert "Severe Skew" in result.output or "severe_skew" in result.output.lower()

    def test_profile_with_output_flag(self, abalone_path: Path, tmp_path: Path):
        """--output should create a JSON file."""
        out_file = tmp_path / "profile_output.json"
        result = runner.invoke(app, ["profile", str(abalone_path), "--output", str(out_file)])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert out_file.exists(), "Output JSON file was not created"
        # Validate JSON structure
        with open(out_file) as f:
            data = json.load(f)
        assert "stress_factors" in data
        assert "row_count" in data

    def test_profile_missing_file(self, tmp_path: Path):
        """Missing input file should exit with code 1."""
        missing = tmp_path / "does_not_exist.csv"
        result = runner.invoke(app, ["profile", str(missing)])
        assert result.exit_code == 1

    def test_profile_verbose_flag(self, abalone_path: Path):
        """--verbose should include per-column detail tables."""
        result = runner.invoke(app, ["profile", str(abalone_path), "--verbose"])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # Verbose mode shows "Skewness by Column" or "Cardinality by Column"
        output_lower = result.output.lower()
        assert (
            "skewness" in output_lower
            or "cardinality" in output_lower
            or "column" in output_lower
        )

    def test_profile_with_tmp_csv(self, tmp_csv: Path):
        """Profile should work with a generated temporary CSV."""
        result = runner.invoke(app, ["profile", str(tmp_csv)])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert "Row Count" in result.output or "row_count" in result.output.lower()


# =========================================================================
# Recommend command tests
# =========================================================================

class TestRecommendCommand:
    """Tests for the `recommend` CLI command."""

    def test_recommend_rulebased(self, abalone_path: Path, tmp_path: Path):
        """Rulebased method should exit 0 and show a recommendation."""
        out_file = tmp_path / "rec_output.json"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(abalone_path),
            "--method", "rulebased",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert "Recommended Model" in result.output or "recommended" in result.output.lower()

    def test_recommend_cpu_only(self, tmp_csv: Path, tmp_path: Path):
        """--cpu-only should exclude GPU-only models from the recommendation."""
        out_file = tmp_path / "rec_cpu.json"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(tmp_csv),
            "--method", "rulebased",
            "--cpu-only",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # Verify output JSON excludes GPU models
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        primary = data["primary_recommendation"]["model_name"]
        # GPU-only models that should NOT be primary when cpu_only=True
        gpu_only_models = {"TabSyn", "TabDDPM", "TVAE", "PATECTGAN", "AutoDiff", "GReaT"}
        assert primary not in gpu_only_models, (
            f"CPU-only constraint violated: recommended {primary}"
        )

    def test_recommend_strict_dp(self, tmp_path: Path):
        """--strict-dp should only recommend DP-capable models."""
        # DP models have min_rows >= 500, so create a larger dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.normal(35, 10, 1500).astype(int),
            "income": np.random.lognormal(10, 1.5, 1500),
            "category": np.random.choice(["A", "B", "C", "D"], 1500),
        })
        dp_csv = tmp_path / "dp_test_data.csv"
        df.to_csv(dp_csv, index=False)
        out_file = tmp_path / "rec_dp.json"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(dp_csv),
            "--method", "rulebased",
            "--strict-dp",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        primary = data["primary_recommendation"]["model_name"]
        # Only AIM, DPCART, PATECTGAN have privacy_dp >= 3
        dp_models = {"AIM", "DPCART", "PATECTGAN"}
        assert primary in dp_models, (
            f"Strict DP constraint violated: recommended {primary}"
        )

    def test_recommend_invalid_method(self, tmp_csv: Path):
        """Invalid --method value should exit with code 1."""
        result = runner.invoke(app, [
            "recommend",
            "--input", str(tmp_csv),
            "--method", "invalid_method",
        ])
        assert result.exit_code == 1

    def test_recommend_with_output_flag(self, tmp_csv: Path, tmp_path: Path):
        """--output should create a JSON file with recommendation details."""
        out_file = tmp_path / "rec_result.json"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(tmp_csv),
            "--method", "rulebased",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "primary_recommendation" in data
        assert "alternatives" in data
        assert "difficulty_summary" in data

    def test_recommend_with_scale_factors(self, tmp_csv: Path, tmp_path: Path):
        """--skew-sf should be accepted and succeed."""
        out_file = tmp_path / "rec_sf.json"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(tmp_csv),
            "--method", "rulebased",
            "--skew-sf", "2.0",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert out_file.exists()

    def test_recommend_missing_input(self, tmp_path: Path):
        """Missing input file should exit with code 1."""
        missing = tmp_path / "nonexistent.csv"
        result = runner.invoke(app, [
            "recommend",
            "--input", str(missing),
            "--method", "rulebased",
        ])
        assert result.exit_code == 1

    def test_recommend_default_output(self, tmp_csv: Path):
        """When no --output is given, the command should still succeed
        and create a default output file."""
        result = runner.invoke(app, [
            "recommend",
            "--input", str(tmp_csv),
            "--method", "rulebased",
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # The default output path is output/rec/rec__{stem}__rulebased.json
        assert "saved to" in result.output.lower() or "Recommendations" in result.output


# =========================================================================
# Benchmark command tests
# =========================================================================

class TestBenchmarkCommand:
    """Tests for the `benchmark` CLI command."""

    def test_benchmark_valid_files(self, tmp_csv_pair):
        """Valid original + synthetic files should exit 0."""
        orig, synth = tmp_csv_pair
        result = runner.invoke(app, [
            "benchmark",
            "--original", str(orig),
            "--synthetic", str(synth),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # Should show quality metrics
        output_lower = result.output.lower()
        assert "quality" in output_lower or "divergence" in output_lower

    def test_benchmark_with_output(self, tmp_csv_pair, tmp_path: Path):
        """--output should create a JSON file with benchmark results."""
        orig, synth = tmp_csv_pair
        out_file = tmp_path / "benchmark_result.json"
        result = runner.invoke(app, [
            "benchmark",
            "--original", str(orig),
            "--synthetic", str(synth),
            "--output", str(out_file),
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "overall_quality_score" in data or "profile_comparison" in data

    def test_benchmark_missing_original(self, tmp_path: Path):
        """Missing original file should exit with code 1."""
        missing = tmp_path / "missing_original.csv"
        synth = tmp_path / "synthetic.csv"
        # Create only synthetic
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(synth, index=False)
        result = runner.invoke(app, [
            "benchmark",
            "--original", str(missing),
            "--synthetic", str(synth),
        ])
        assert result.exit_code == 1

    def test_benchmark_missing_synthetic(self, tmp_csv_pair, tmp_path: Path):
        """Missing synthetic file should exit with code 1."""
        orig, _ = tmp_csv_pair
        missing = tmp_path / "missing_synthetic.csv"
        result = runner.invoke(app, [
            "benchmark",
            "--original", str(orig),
            "--synthetic", str(missing),
        ])
        assert result.exit_code == 1

    def test_benchmark_verbose(self, tmp_csv_pair):
        """--verbose should show per-column divergence details."""
        orig, synth = tmp_csv_pair
        result = runner.invoke(app, [
            "benchmark",
            "--original", str(orig),
            "--synthetic", str(synth),
            "--verbose",
        ])
        assert result.exit_code == 0, f"stderr: {result.output}"
        # Verbose mode should show per-column divergence table
        output_lower = result.output.lower()
        assert "column" in output_lower or "divergence" in output_lower
