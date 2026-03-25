"""
Integration tests for the full pipeline: CSV → Profile → JSON.

Tests the complete workflow from file loading through analysis to JSON output.
"""

import json

import pandas as pd

from pathlib import Path
from synthony import StochasticDataAnalyzer
from synthony.benchmark.generators import BenchmarkDatasetGenerator


class TestFullPipeline:
    """Test complete analysis pipeline."""

    # pytest will NOT collect tests from classes that define an __init__.
    # Define `test_data_path` as a class attribute instead so methods can access it.
    test_data_path = Path('../data')


    def test_csv_to_profile_to_json(self, tmp_path):
        """Full pipeline: CSV → Analysis → JSON output."""
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": range(100),
            "value": [1, 2, 3] * 33 + [1],  # Low cardinality
            "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,  # Slight imbalance
        })
        df.to_csv(csv_path, index=False)

        # Analyze
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze_from_file(csv_path)

        # Validate basic structure
        assert profile.row_count == 100
        assert profile.column_count == 3
        assert isinstance(profile.dataset_id, str)
        assert len(profile.dataset_id) > 0

        # Check stress factors exist
        assert hasattr(profile, "stress_factors")
        assert hasattr(profile.stress_factors, "severe_skew")
        assert hasattr(profile.stress_factors, "small_data")

        # Export JSON
        json_path = tmp_path / "profile.json"
        analyzer.to_json(profile, json_path)

        assert json_path.exists()

        # Validate JSON can be loaded and parsed
        with open(json_path) as f:
            data = json.load(f)
            assert data["row_count"] == 100
            assert data["column_count"] == 3
            assert "stress_factors" in data
            assert "dataset_id" in data


    def test_titanic_sample_csv_to_profile_to_json(self, temp_parquet_file):
        """Full pipeline: CSV → Analysis → JSON output."""
        if temp_parquet_file is None:
            temp_parquet_file = self.test_data_path / 'titanic_sample.csv'

        # Analyze
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze_from_file(temp_parquet_file)

        # Validate basic structure
        assert isinstance(profile.dataset_id, str)
        assert len(profile.dataset_id) > 0

        # Check stress factors exist
        assert hasattr(profile, "stress_factors")
        assert hasattr(profile.stress_factors, "severe_skew")
        assert hasattr(profile.stress_factors, "small_data")

        # Export JSON
        json_path = self.test_data_path / "titatic_profile.json"

        analyzer.to_json(profile, json_path)

        assert json_path.exists()

        # Validate JSON can be loaded and parsed
        with open(json_path) as f:
            data = json.load(f)
            assert data["row_count"] == 1000
            assert data["column_count"] == 3
            assert "stress_factors" in data
            assert "dataset_id" in data

    def test_titanic_parquet_loading(self, temp_parquet_file):
        """Test loading and analyzing Parquet files."""

        if temp_parquet_file is None:
            temp_parquet_file = self.test_data_path / 'titanic_sample.parquet'

        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze_from_file(temp_parquet_file)

        assert profile.row_count == 1000
        assert profile.column_count == 3

    def test_dataframe_analysis(self, sample_normal_df):
        """Analyze DataFrame directly without file I/O."""
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze(sample_normal_df)

        assert profile.row_count == 1000
        assert profile.column_count == 3
        assert profile.stress_factors.small_data is False  # 1000 rows > 500 threshold
        assert profile.stress_factors.large_data is False  # 1000 rows < 50k threshold



    def test_json_serialization_roundtrip(self, sample_normal_df):
        """Test JSON serialization and deserialization."""
        analyzer = StochasticDataAnalyzer()
        original_profile = analyzer.analyze(sample_normal_df)

        # Serialize to JSON
        json_str = analyzer.to_json(original_profile)

        # Deserialize
        loaded_profile = analyzer.from_json(json_str)

        # Compare key fields
        assert loaded_profile.row_count == original_profile.row_count
        assert loaded_profile.column_count == original_profile.column_count
        assert (
            loaded_profile.stress_factors.severe_skew
            == original_profile.stress_factors.severe_skew
        )


class TestBenchmarkDatasets:
    """Test benchmark dataset generation and profiling."""

    def test_long_tail_dataset_profiling(self):
        """Profile 'Long Tail' dataset and verify severe skew detection."""
        df = BenchmarkDatasetGenerator.generate_long_tail(n_rows=1000, seed=42)

        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze(df)

        # Should detect severe skew
        assert profile.stress_factors.severe_skew is True
        assert profile.skewness is not None
        assert profile.skewness.max_skewness > 2.0  # Above threshold

    def test_needle_in_haystack_profiling(self):
        """Profile 'Needle in Haystack' dataset and verify Zipfian detection."""
        df = BenchmarkDatasetGenerator.generate_needle_in_haystack(n_rows=1000, seed=42)

        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze(df)

        # Should detect Zipfian distribution
        assert profile.stress_factors.zipfian_distribution is True
        assert profile.zipfian is not None
        assert profile.zipfian.detected is True
        assert profile.zipfian.top_20_percent_ratio is not None
        assert profile.zipfian.top_20_percent_ratio > 0.80  # Above threshold

    def test_small_data_trap_profiling(self):
        """Profile 'Small Data Trap' dataset and verify small data detection."""
        df = BenchmarkDatasetGenerator.generate_small_data_trap(n_rows=200, seed=42)

        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze(df)

        # Should detect small data
        assert profile.stress_factors.small_data is True
        assert profile.row_count == 200
        assert profile.row_count < 500  # Below threshold

    def test_save_benchmarks(self, tmp_path):
        """Test saving all benchmark datasets to disk."""
        output_dir = tmp_path / "benchmarks"

        BenchmarkDatasetGenerator.save_benchmarks(output_dir)

        # Check all three files were created
        assert (output_dir / "dataset_a_long_tail.csv").exists()
        assert (output_dir / "dataset_b_needle_haystack.csv").exists()
        assert (output_dir / "dataset_c_small_data.csv").exists()

        # Verify files can be loaded
        df_a = pd.read_csv(output_dir / "dataset_a_long_tail.csv")
        assert len(df_a) == 10000  # Default size


class TestCustomConfiguration:
    """Test custom threshold configuration."""

    def test_custom_thresholds(self, sample_skewed_df):
        """Test analyzer with custom threshold configuration."""
        from synthony import AnalyzerConfig

        # Create config with very high skewness threshold
        config = AnalyzerConfig(skewness_threshold=10.0)  # Much higher than default 2.0

        analyzer = StochasticDataAnalyzer(config=config)
        profile = analyzer.analyze(sample_skewed_df)

        # With high threshold, may not detect as severe
        # (depends on actual skewness of sample_skewed_df)
        assert profile.thresholds_used["skewness_threshold"] == 10.0
