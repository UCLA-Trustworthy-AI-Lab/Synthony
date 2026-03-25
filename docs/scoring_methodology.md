# Model Capability Scoring Methodology

This document defines how model capability scores (0-4) are derived from benchmark metrics.

## 1. Capability Dimensions

| Capability | Description | Evaluated On |
|------------|-------------|--------------|
| `skew_handling` | Ability to preserve skewed distributions | Columns with \|skew\| > 2.0 |
| `cardinality_handling` | Ability to handle high cardinality | Columns with >500 unique values |
| `zipfian_handling` | Ability to capture long-tail/rare values | Zipfian distribution datasets |
| `small_data` | Performance on small datasets | Datasets with <500 rows |
| `correlation_handling` | Ability to preserve inter-column correlations | All numeric columns |
| `privacy_dp` | Differential privacy guarantees | Static flag (model property) |

---

## 2. Score Thresholds

Scores are on a 0-4 scale based on metric performance:

| Score | Label | Threshold | Interpretation |
|-------|-------|-----------|----------------|
| **4** | Excellent | metric ≥ 0.90 | Best-in-class performance |
| **3** | Good | 0.75 ≤ metric < 0.90 | Reliable performance |
| **2** | Moderate | 0.50 ≤ metric < 0.75 | Acceptable with caveats |
| **1** | Poor | 0.25 ≤ metric < 0.50 | Significant limitations |
| **0** | Fails | metric < 0.25 | Not suitable |

---

## 3. Metric Formulas

### 3.1 Skew Handling

**Goal**: Measure how well the model preserves skewness in heavily skewed columns.

```python
def calculate_skew_score(benchmark):
    """Calculate skew handling capability (0-4)."""
    original_skew = benchmark['profile_comparison']['skewness']['original']
    synthetic_skew = benchmark['profile_comparison']['skewness']['synthetic']
    
    # Only evaluate columns with |original_skew| > 2.0
    skew_scores = []
    for col, orig_val in original_skew.items():
        if abs(orig_val) > 2.0:
            synth_val = synthetic_skew.get(col, 0)
            # Preservation ratio (1.0 = perfect, 0.0 = lost all skew)
            preservation = 1 - abs(orig_val - synth_val) / abs(orig_val)
            skew_scores.append(max(0, preservation))
    
    if not skew_scores:
        return None  # No skewed columns to evaluate
    
    avg_preservation = sum(skew_scores) / len(skew_scores)
    return metric_to_score(avg_preservation)
```

### 3.2 Cardinality Handling

**Goal**: Measure how well the model preserves unique value density (proportion of unique values).

> **v7.0.0 Change**: The cardinality formula was updated from raw ratio (`synth_unique / orig_unique`) to **density-normalized ratio** to eliminate bias when synthetic datasets have fewer rows than the original. See `docs/analysis_summary_spark.md` Section 2 for details.

```python
def calculate_cardinality_score(benchmark):
    """Calculate cardinality handling capability (0-4).

    Uses density-normalized formula to avoid bias when
    synth_rows << orig_rows (e.g., spark benchmarks generate
    ~1000 rows regardless of original size).
    """
    original_card = benchmark['profile_comparison']['cardinality']['original']
    synthetic_card = benchmark['profile_comparison']['cardinality']['synthetic']
    orig_rows = benchmark.get('original_rows', 1)
    synth_rows = benchmark.get('synthetic_rows', 1)

    # Only evaluate columns with original cardinality > 500
    card_scores = []
    for col, orig_val in original_card.items():
        if orig_val > 500:
            synth_val = synthetic_card.get(col, 0)
            # Density-normalized: measures proportion of unique values
            orig_density = orig_val / max(orig_rows, 1)
            synth_density = synth_val / max(synth_rows, 1)
            ratio = min(synth_density / max(orig_density, 1e-10), 1.0)
            card_scores.append(ratio)

    if not card_scores:
        return None  # No high-cardinality columns

    avg_ratio = sum(card_scores) / len(card_scores)
    return metric_to_score(avg_ratio)
```

### 3.3 Correlation Handling

**Goal**: Measure preservation of inter-column relationships.

```python
def calculate_correlation_score(benchmark):
    """Calculate correlation handling capability (0-4)."""
    orig_corr = benchmark['profile_comparison']['correlation']['original']
    synth_corr = benchmark['profile_comparison']['correlation']['synthetic']
    
    # Use mean R-squared preservation
    orig_r2 = orig_corr.get('mean_r_squared', 0)
    synth_r2 = synth_corr.get('mean_r_squared', 0)
    
    if orig_r2 == 0:
        return None
    
    preservation = synth_r2 / orig_r2
    
    # Also factor in fidelity.correlation_preservation if available
    fidelity_corr = benchmark.get('fidelity', {}).get('correlation_preservation')
    if fidelity_corr:
        preservation = (preservation + fidelity_corr) / 2
    
    return metric_to_score(min(preservation, 1.0))
```

### 3.4 Small Data

**Goal**: Measure quality when training on limited data.

```python
def calculate_small_data_score(benchmark):
    """Calculate small data handling capability (0-4)."""
    rows = benchmark.get('original_rows', 0)
    
    if rows >= 1000:
        return None  # Not a small data benchmark
    
    # Use overall quality score directly
    quality = benchmark.get('overall_quality_score', 0)
    return metric_to_score(quality)
```

### 3.5 Helper Function

```python
def metric_to_score(metric_value):
    """Convert 0-1 metric to 0-4 score."""
    if metric_value >= 0.90:
        return 4
    elif metric_value >= 0.75:
        return 3
    elif metric_value >= 0.50:
        return 2
    elif metric_value >= 0.25:
        return 1
    else:
        return 0
```

---

## 4. Aggregation Across Datasets

When a model has benchmarks on multiple datasets:

```python
def aggregate_capability(scores):
    """Aggregate scores from multiple benchmarks."""
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return 2  # Default to moderate if no data
    return round(sum(valid_scores) / len(valid_scores))
```

---

## 5. Static Capabilities

Some capabilities are not derived from benchmarks:

| Capability | Source | Values |
|------------|--------|--------|
| `privacy_dp` | Model documentation | 0 (none), 3 (moderate), 4 (strong) |
| `requires_gpu` | Model class inspection | true/false |
| `min_rows` | Known constraints | Integer threshold |

---

## 6. Example Calculation

Given benchmark for ARF on abalone:

```json
{
  "overall_quality_score": 0.9553,
  "fidelity": {
    "correlation_preservation": 0.9734
  },
  "profile_comparison": {
    "skewness": {
      "original": {"Height": 3.128},
      "synthetic": {"Height": -0.006}
    },
    "cardinality": {
      "original": {"Whole weight": 2429},
      "synthetic": {"Whole weight": 100}
    }
  }
}
```

**Calculations**:

- `skew_handling`: Height preservation = 1 - |3.128 - (-0.006)| / 3.128 = 0.0 → **Score: 0**
- `cardinality_handling`: Whole weight = 100/2429 = 0.04 → **Score: 0**
- `correlation_handling`: 0.9734 → **Score: 4**
- `overall_quality`: 0.9553 → For small_data context → **Score: 4**

---

## 7. Engine Scoring Pipeline (v7.0.0)

The recommendation engine uses capabilities from this methodology in a multi-stage pipeline:

### 7.1 Required Capability Calculation

Based on dataset stress factors, the engine determines required capability levels. All thresholds are configurable via `config/model_capabilities.json` → `metadata.capability_thresholds`:

| Stress Factor | Severe Condition | Required Level |
|---------------|-----------------|----------------|
| Skew | max_skewness >= 4.0 | 4 (high) |
| Skew | max_skewness < 4.0 | 3 (moderate) |
| Cardinality | max_cardinality >= 5000 | 4 (high) |
| Cardinality | max_cardinality < 5000 | 3 (moderate) |
| Zipfian | top_20_percent_ratio >= 0.9 | 4 (high) |
| Zipfian | top_20_percent_ratio < 0.9 | 3 (moderate) |
| Small Data | rows < 999 | 4 |
| Correlation | higher_order detected | 3 |

### 7.2 Score Decay Curve

When a model's capability falls below the required level, the match score decays. Configurable via `metadata.score_decay`:

| Condition | Match Score |
|-----------|------------|
| model_score >= required | 1.0 (exact) |
| model_score == required - 1 | 0.7 (near) |
| model_score == required - 2 | 0.4 (moderate) |
| model_score < required - 2 | 0.0 (poor) |

For non-required capabilities (required = 0), the match score is `model_score / 4.0`, allowing differentiation even when stress factors are not active.

### 7.3 Empirical Quality Bonus

After capability scoring, an empirical quality bonus is added:

```
quality_bonus = avg_quality_score * quality_weight (default 0.3)
total_score = capability_score + quality_bonus
```

This gives models like CART (quality=0.981, bonus=+0.294) a meaningful advantage over lower-quality models like TabDDPM (quality=0.697, bonus=+0.209).

### 7.4 Tie-Breaking

When top models score within 5% of each other, deterministic tie-breaking applies:

1. **Small data** (rows < 999): Prefer ARF > CART > BayesianNetwork > SMOTE
2. **Speed preference**: CART > ARF > SMOTE > TVAE > DPCART
3. **GPU available** (cpu_only=false): GReaT > TabDDPM > TabSyn > AutoDiff > TVAE
4. **CPU only**: CART > SMOTE > BayesianNetwork > ARF > NFlow

All priority lists are configurable via `tie_breaking_priority` in the registry.

### 7.5 Hard Problem Confidence

When the Hard Problem path is triggered (skew + cardinality + zipfian), confidence scores are assigned from `metadata.hard_problem_confidence`:

| Condition | Confidence |
|-----------|-----------|
| Primary model selected | 0.95 |
| Fallback model selected | 0.85 |
| Alternative models | 0.70 |

### 7.6 Data-Driven Configuration

All engine parameters are loaded from `config/model_capabilities.json` at initialization:

| Registry Section | Engine Config Fields |
|-----------------|---------------------|
| `metadata.dp_threshold` | `dp_min_score` |
| `metadata.capability_thresholds` | 11 threshold fields |
| `metadata.hard_problem_confidence` | 3 confidence fields |
| `metadata.score_decay` | 4 decay curve fields |
| `hard_problem_routing` | primary, fallback, priority |
| `tie_breaking_priority` | GPU/CPU/speed/small_data priorities |

---

## 8. Validation

After generating capabilities:

1. Compare to manually-created v4 capabilities
2. Verify recommendations still sensible via `test_accuracy_regression.py`
3. Document significant discrepancies
4. Adjust thresholds if needed — only requires JSON edits to `config/model_capabilities.json`
