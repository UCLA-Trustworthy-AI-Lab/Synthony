# Scripts Documentation

## Batch Dataset Analysis

### Overview

The `batch_analyze_datasets.py` script performs comprehensive analysis on all CSV files in a directory, generating individual reports and cross-dataset comparisons.

### Usage

#### Basic Usage

```bash
python scripts/batch_analyze_datasets.py
```

This will:
1. Find all `*.csv` files in `dataset/input_data/`
2. Run `StochasticDataAnalyzer` and `ColumnAnalyzer` on each
3. Save individual JSON reports to `dataset/analysis_results/`
4. Generate a comparison report across all datasets

#### Expected Output

```
dataset/analysis_results/
├── abalone_analysis.json          # Individual dataset reports
├── Bean_analysis.json
├── ...
├── comparison_report.json         # Cross-dataset comparison
└── ANALYSIS_SUMMARY.md            # Human-readable summary
```

### Output Format

#### Individual Dataset Reports

Each `{dataset}_analysis.json` contains:

```json
{
  "metadata": {
    "dataset_name": "abalone.csv",
    "file_size_mb": 0.19,
    "analysis_time_seconds": 0.12,
    "analyzed_at": "2026-01-10T19:45:14"
  },
  "dataset_profile": {
    "stress_factors": { ... },
    "skewness": { ... },
    "cardinality": { ... }
  },
  "column_analysis": {
    "columns": {
      "Height": {
        "difficulty": {
          "overall_difficulty": 3,
          "skew_difficulty": 3,
          "cardinality_difficulty": 1
        },
        "recommended_model_types": [
          "Skew handling: TabDDPM, TabSyn, AutoDiff, GReaT, TabTree (score 3-4)"
        ]
      }
    }
  }
}
```

#### Comparison Report

`comparison_report.json` contains:

```json
{
  "total_datasets": 11,
  "summary_statistics": {
    "total_rows": 99182,
    "processing_speed_rows_per_second": 28748.14
  },
  "stress_factor_distribution": {
    "severe_skew": 9,
    "high_cardinality": 9,
    "zipfian_distribution": 4
  },
  "datasets": [
    {
      "dataset_name": "Bean.csv",
      "max_column_difficulty": 4,
      "difficult_columns_count": 16,
      "difficult_columns": ["Area", "Perimeter", ...]
    }
  ]
}
```

### Performance

**Benchmark Results** (11 datasets, 99,182 total rows):
- Total time: 3.45 seconds
- Average per dataset: 0.31 seconds
- Processing speed: 28,748 rows/second

### Customization

#### Using Custom Directories

Edit the `main()` function in `batch_analyze_datasets.py`:

```python
def main():
    input_dir = Path("path/to/your/csvs")
    output_dir = Path("path/to/results")

    batch_analyzer = DatasetBatchAnalyzer(input_dir, output_dir)
    batch_analyzer.analyze_all()
    batch_analyzer.generate_comparison_report()
```

#### Using Custom Thresholds

```python
from synthony import AnalyzerConfig

config = AnalyzerConfig(
    skewness_threshold=3.0,      # Default: 2.0
    cardinality_threshold=1000,  # Default: 500
    zipfian_ratio=0.85          # Default: 0.80
)

analyzer = StochasticDataAnalyzer(config=config)
```

### Error Handling

The script handles errors gracefully:

```python
# Errors are logged but don't stop the batch
try:
    result = self._analyze_dataset(csv_path)
    self.results.append(result)
except Exception as e:
    self.errors.append({
        "dataset": csv_path.name,
        "error": str(e),
        "error_type": type(e).__name__
    })
```

Errors are reported in the comparison report:

```json
{
  "errors": [
    {
      "dataset": "corrupted.csv",
      "error": "Unable to parse CSV",
      "error_type": "ParserError"
    }
  ]
}
```

### Integration with Workflow

#### 1. Data Profiling Stage

```bash
# Analyze all datasets
python scripts/batch_analyze_datasets.py
```

#### 2. Review Results

```bash
# View summary
cat dataset/analysis_results/ANALYSIS_SUMMARY.md

# View comparison report
cat dataset/analysis_results/comparison_report.json | jq
```

#### 3. Access Individual Reports

```python
import json

# Load specific dataset analysis
with open('dataset/analysis_results/abalone_analysis.json', 'r') as f:
    analysis = json.load(f)

# Access difficulty scores
for col_name, col_profile in analysis['column_analysis']['columns'].items():
    if col_profile['difficulty']['overall_difficulty'] >= 3:
        print(f"{col_name}: {col_profile['recommended_model_types']}")
```

### Interpreting Results

#### Difficulty Scores (0-4 scale)

| Score | Meaning | Models Required |
|-------|---------|-----------------|
| 0-1 | Trivial | Any model (TVAE, GaussianCopula, etc.) |
| 2 | Moderate | Most models (TVAE, CTGAN, ARF) |
| 3 | Hard | Specialized models (TabDDPM, TabSyn, GReaT, TabTree) |
| 4 | Very Hard | Advanced only (GReaT, TabTree, TabDDPM) |

#### Stress Factors

| Factor | Threshold | Impact |
|--------|-----------|--------|
| Severe Skew | \|skewness\| > 2.0 | Breaks basic GANs/VAEs, requires diffusion or LLMs |
| High Cardinality | Unique values > 500 | Mode collapse risk, requires advanced embedding |
| Zipfian Distribution | Top 20% > 80% | Power-law distribution, rare category challenges |
| Higher-Order Correlation | Dense + low R² | Non-linear relationships, requires deep models |

### Troubleshooting

#### Large Files (>100MB)

For very large CSV files, consider:

1. **Sampling** before analysis:
   ```python
   df = pd.read_csv(csv_path, nrows=100000)  # Sample first 100k rows
   ```

2. **Chunked processing**:
   ```python
   chunks = pd.read_csv(csv_path, chunksize=10000)
   # Process in chunks
   ```

#### Memory Issues

If analysis fails due to memory:

```bash
# Monitor memory usage
python -m memory_profiler scripts/batch_analyze_datasets.py
```

Optimize by:
- Reducing correlation matrix size (samples >100 columns automatically)
- Processing subsets of datasets
- Using Polars for large files

#### Timeout on Large Datasets

The script has a 300-second (5 minute) timeout. For larger datasets:

```python
# In batch_analyze_datasets.py, increase timeout
@timeout(600)  # 10 minutes
def _analyze_dataset(self, csv_path: Path):
    ...
```

### Advanced Usage

#### Parallel Processing

For very large batches, process datasets in parallel:

```python
from multiprocessing import Pool

def analyze_dataset_parallel(csv_path):
    analyzer = DatasetBatchAnalyzer(input_dir, output_dir)
    return analyzer._analyze_dataset(csv_path)

with Pool(4) as pool:  # 4 parallel workers
    results = pool.map(analyze_dataset_parallel, csv_files)
```

#### Custom Metrics

Add custom metrics to the analysis:

```python
# In _analyze_dataset method
result["custom_metrics"] = {
    "mean_column_difficulty": sum(
        col.difficulty.overall_difficulty
        for col in column_analysis.columns.values()
    ) / len(column_analysis.columns)
}
```

### See Also

- [Column Analyzer Documentation](../docs/column_analyzer_summary.md)
- [Architecture Overview](../docs/architecture_v2.md)
- [Validation Plan](../docs/validation_plan_knowledge_base_v2.md)
