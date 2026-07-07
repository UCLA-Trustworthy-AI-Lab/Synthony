# Validation Plan: Benchmarking & Knowledge Base Refinement

**Objective**: Ensure the 0-4 scores in `SystemPrompt.md` reflect reality.

## 1. Synthetic Control Datasets (The Exam Questions)

We generate 3 specific datasets to test every model version.

### Dataset A: "The Long Tail" (Skew Test)

* **Generation**: `scipy.stats.lognorm(s=0.95, scale=exp(5))`
* **Profile**: Skewness = 4.5.
* **Pass Criteria**: Wasserstein Distance < 0.1.
* **Failure**: Histogram is Gaussian (bell curve) instead of skewed.

### Dataset B: "The Needle in Haystack" (Zipfian Test)

* **Generation**: 10,000 rows. 1,000 Unique Categories. Top 10 categories = 90% of rows.
* **Profile**: Zipfian Ratio = 0.09.
* **Pass Criteria**: The generated data must contain at least 80% of the "Rare" categories (the tail).
* **Failure**: Generated data only contains the top 10 categories (Mode Collapse).

### Dataset C: "The Small Data Trap"

* **Generation**: 200 rows of multivariate data.
* **Profile**: Small Data.
* **Pass Criteria**: Train on 200, Test on real holdout. R² > 0.6.
* **Failure**: R² < 0 (Overfitting/Memorization).

## 2. Execution Strategy

1. **Grid Search**: Run all models on Datasets A, B, C. (Note: written when the
   registry had 13 models incl. the now-removed `GaussianCopula`; the current
   registry has 15 models — see `config/model_capabilities.json`.)
2. **Score Update**:
    * If **ARF** outperforms **CART** on Dataset C, update ARF's "Small Data" score to 4 and CART's accordingly.
    * If **GReaT** fails Dataset B (due to token limits), downgrade "High Cardinality" score.

## 3. Implementation Phases

### Phase A: Infrastructure (Part of Package 1)

- Create `BenchmarkGenerator` class to produce these specific distributions on demand.
* Implement `Evaluator` class with WD, TVD, and MIA metrics.

### Phase B: Execution (Part of Verification Support)

- Run a "Grid Search" of Models x StressFactors.
* This is computationally expensive; run on a subset first (e.g., CTGAN vs. GReaT vs. TabDDPM).

### Phase C: Knowledge Base Refinement

- Update `SystemPrompt.md` if empirical scores differ from theoretical/paper claims.
* *Example*: If CTGAN actually captures Skew=3.0 well in our tests, upgrade its score from 1 to 2.

## 4. Specific Validation Scenarios (for User Verification)

### Scenario 1: The "GReaT vs. Skew" Test

- **Input**: 10k rows, 1 col, LogNormal skew=5.0.
* **Run**: GReaT (Score 4) vs. CTGAN (Score 1).
* **Expectation**: GReaT histogram overlaps tail; CTGAN histogram cuts off or is Gaussian-forced.

### Scenario 2: The "Zipfian Tail" Test

- **Input**: 10k rows, 1 col, Zipfian categories. Top 10 categories = 80% of data. Tail = 1000 categories.
* **Run**: GReaT (Score 4) vs. TVAE (Score 1).
* **Expectation**: GReaT generates valid rare tokens. TVAE generates mostly top-10 or noise.

## 5. Actionable Next Step

Create `src/benchmark/datasets.py` in Package 1 to generate these 5 control datasets.
