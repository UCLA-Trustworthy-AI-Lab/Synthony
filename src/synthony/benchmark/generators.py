"""
Benchmark dataset generators for validation.

Creates synthetic control datasets with known stress factors to validate
model recommendations and stress detection algorithms.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm, skew


class BenchmarkDatasetGenerator:
    """Generate synthetic control datasets for validation.

    Creates three benchmark datasets documented in:
    - docs/validation_plan_knowledge_base_v2.md

    Each dataset is designed to test specific stress detection capabilities:
    - Dataset A: Severe skewness (long tail)
    - Dataset B: Zipfian distribution (mode collapse risk)
    - Dataset C: Small data (overfitting risk)
    """

    @staticmethod
    def generate_long_tail(
        n_rows: int = 10000, target_skewness: float = 4.5, seed: int = 42
    ) -> pd.DataFrame:
        """Dataset A: "The Long Tail" - Severe skewness test.

        Generates data with LogNormal distribution to create severe
        positive skewness (long right tail). This tests whether models
        can capture tail distributions that break basic GANs/VAEs.

        Generation:
        - Use scipy.stats.lognorm(s=0.95, scale=exp(5))
        - Verify resulting skewness ≈ 4.5

        Purpose: Test skew-handling capabilities (Score 3-4 should pass)

        Args:
            n_rows: Number of rows to generate. Default 10,000.
            target_skewness: Target skewness value. Default 4.5.
            seed: Random seed for reproducibility. Default 42.

        Returns:
            DataFrame with columns: value (skewed), category, id
        """
        np.random.seed(seed)

        # LogNormal distribution parameters
        # s = shape parameter, scale = exp(mu) where mu is mean of log
        s = 0.95  # Shape parameter
        scale = np.exp(5)  # Scale parameter

        # Generate skewed data
        skewed_values = lognorm.rvs(s=s, scale=scale, size=n_rows)

        # Add some additional columns for realism
        categories = np.random.choice(["A", "B", "C"], size=n_rows)

        df = pd.DataFrame({
            "value": skewed_values,
            "category": categories,
            "id": range(n_rows),
        })

        # Verify skewness
        actual_skew = skew(skewed_values, bias=False)
        print(
            f"Generated 'Long Tail' dataset: "
            f"target_skew={target_skewness}, actual_skew={actual_skew:.2f}, "
            f"rows={n_rows}"
        )

        if abs(actual_skew - target_skewness) > 1.0:
            print(
                f"Warning: Actual skewness ({actual_skew:.2f}) differs from target ({target_skewness}) by >1.0"
            )

        return df

    @staticmethod
    def generate_needle_in_haystack(
        n_rows: int = 10000,
        n_categories: int = 1000,
        top_n: int = 10,
        top_concentration: float = 0.90,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Dataset B: "The Needle in Haystack" - Zipfian distribution test.

        Generates data with power-law distribution where a small number
        of categories dominate. This tests mode collapse resistance.

        Generation:
        1. Create 1000 unique categories
        2. Top 10 categories = 90% of rows (9000 rows)
        3. Remaining 990 categories = 10% of rows (1000 rows)

        Purpose: Test Zipfian distribution detection and rare category coverage

        Args:
            n_rows: Total number of rows. Default 10,000.
            n_categories: Total number of unique categories. Default 1000.
            top_n: Number of "top" categories. Default 10.
            top_concentration: Fraction of data in top categories. Default 0.90.
            seed: Random seed for reproducibility. Default 42.

        Returns:
            DataFrame with columns: category (Zipfian), value (random), id
        """
        np.random.seed(seed)

        # Allocate rows to top vs. tail categories
        top_rows = int(n_rows * top_concentration)
        tail_rows = n_rows - top_rows

        # Generate top categories (evenly distributed within top_n)
        top_categories = [f"top_{i}" for i in range(top_n)]
        top_data = np.random.choice(top_categories, size=top_rows)

        # Generate tail categories (rare)
        tail_categories = [f"rare_{i}" for i in range(n_categories - top_n)]
        tail_data = np.random.choice(tail_categories, size=tail_rows)

        # Combine and shuffle
        all_categories = np.concatenate([top_data, tail_data])
        np.random.shuffle(all_categories)

        df = pd.DataFrame({
            "category": all_categories,
            "value": np.random.randn(n_rows),  # Random numeric column
            "id": range(n_rows),
        })

        # Verify Zipfian ratio
        value_counts = df["category"].value_counts()
        n_top_20pct = int(np.ceil(0.2 * len(value_counts)))
        ratio = value_counts.head(n_top_20pct).sum() / len(df)

        print(
            f"Generated 'Needle in Haystack' dataset: "
            f"zipfian_ratio={ratio:.3f}, unique_categories={df['category'].nunique()}, "
            f"rows={n_rows}"
        )

        if ratio < 0.80:
            print(
                f"Warning: Zipfian ratio ({ratio:.3f}) is below threshold (0.80)"
            )

        return df

    @staticmethod
    def generate_small_data_trap(
        n_rows: int = 200, n_features: int = 10, seed: int = 42
    ) -> pd.DataFrame:
        """Dataset C: "The Small Data Trap" - Overfitting risk test.

        Generates a small dataset with correlated features to test
        overfitting prevention in models. Small datasets require
        tree-based models like ARF or GaussianCopula.

        Generation:
        - 200 rows (default), multivariate data
        - Mix of numeric and categorical features
        - Include some correlations to test overfitting detection

        Purpose: Test overfitting prevention (Row count < 500)

        Args:
            n_rows: Number of rows. Default 200 (< 500 threshold).
            n_features: Number of features. Default 10.
            seed: Random seed for reproducibility. Default 42.

        Returns:
            DataFrame with mixed numeric/categorical features
        """
        np.random.seed(seed)

        data = {}

        # Generate correlated numeric features
        # Base signal that other features will correlate with
        base = np.random.randn(n_rows)

        for i in range(n_features // 2):
            # Add noise to create imperfect correlation
            noise = np.random.randn(n_rows) * 0.3
            data[f"num_{i}"] = base + noise

        # Generate categorical features
        for i in range(n_features // 2):
            n_cats = np.random.randint(3, 8)  # 3-7 categories per feature
            data[f"cat_{i}"] = np.random.choice(
                [f"c{j}" for j in range(n_cats)], size=n_rows
            )

        df = pd.DataFrame(data)
        df["id"] = range(n_rows)

        print(
            f"Generated 'Small Data Trap' dataset: "
            f"rows={n_rows}, features={n_features}"
        )

        return df

    @staticmethod
    def save_benchmarks(output_dir: Path | str) -> None:
        """Generate and save all three benchmark datasets to disk.

        Creates all benchmark datasets and saves them as CSV files
        in the specified directory.

        Args:
            output_dir: Directory to save datasets. Created if doesn't exist.

        Example:
            ```python
            BenchmarkDatasetGenerator.save_benchmarks("./benchmarks")
            # Creates:
            # - ./benchmarks/dataset_a_long_tail.csv
            # - ./benchmarks/dataset_b_needle_haystack.csv
            # - ./benchmarks/dataset_c_small_data.csv
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating benchmark datasets in: {output_dir}")
        print("=" * 60)

        # Generate all datasets
        datasets = {
            "dataset_a_long_tail.csv": BenchmarkDatasetGenerator.generate_long_tail(),
            "dataset_b_needle_haystack.csv": BenchmarkDatasetGenerator.generate_needle_in_haystack(),
            "dataset_c_small_data.csv": BenchmarkDatasetGenerator.generate_small_data_trap(),
        }

        # Save to CSV
        for filename, df in datasets.items():
            path = output_dir / filename
            df.to_csv(path, index=False)
            print(f"✓ Saved: {path} ({len(df)} rows, {len(df.columns)} columns)")

        print("=" * 60)
        print(f"All benchmark datasets saved to: {output_dir}\n")


# Backwards compatibility
