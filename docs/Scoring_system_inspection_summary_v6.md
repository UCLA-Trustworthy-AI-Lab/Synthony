# Scoring System Inspection Summary - v6

**Date**: 2026-02-04
**Benchmark Source**: `output/benchmark/trial4/analysis_comparison.json`
**Methodology Reference**: `docs/scoring_methodology.md`
**Scope**: 11 models evaluated across 8 datasets (abalone, Bean, faults, IndianLiverPatient, insurance, Obesity, Shoppers, wilt)

---

## 1. Overview

This document summarizes the inspection of model capability scores used by the Synthony recommender engine. The inspection compared scores in the active configuration files against empirically-derived scores from trial4 benchmark results, applying the scoring methodology defined in `docs/scoring_methodology.md`.

### Files Inspected

| File | Version Before | Role |
|------|---------------|------|
| `config/model_capabilities.json` | 6.0.0 | Active config with exclude flags |
| `src/synthony/recommender/model_capabilities.json` | 1.0.0 | Runtime recommender registry |
| `docs/model_capabilities_v5.json` | 5.0.0 | Auto-generated (sparse benchmarks) |

### Outcome

All three files contained scores inconsistent with trial4 empirical data. The v6 update recalibrates scores for 11 benchmarked models using the `derived_capabilities` from `analysis_comparison.json`.

---

## 2. Scoring Methodology Verification

The methodology in `docs/scoring_methodology.md` was reviewed and confirmed internally consistent:

```
metric_to_score(value):
    >= 0.90 -> 4 (Excellent)
    >= 0.75 -> 3 (Good)
    >= 0.50 -> 2 (Moderate)
    >= 0.25 -> 1 (Poor)
    <  0.25 -> 0 (Fails)
```

**Verification**: All 11 models' `derived_capabilities` in `analysis_comparison.json` were cross-checked against the raw `empirical` preservation rates. All scores match the `metric_to_score` thresholds correctly:

| Model | skew_pres | -> score | card_pres | -> score | corr_pres | -> score |
|-------|-----------|----------|-----------|----------|-----------|----------|
| DPCART | 0.875 | 3 | 0.500 | 2 | 0.250 | 1 |
| BayesianNetwork | 0.875 | 3 | 0.875 | 3 | 0.500 | 2 |
| ARF | 0.875 | 3 | 0.875 | 3 | 0.375 | 1 |
| CART | 0.875 | 3 | 1.000 | 4 | 0.625 | 2 |
| NFlow | 1.000 | 4 | 0.857 | 3 | 0.000 | 0 |
| TabDDPM | 0.875 | 3 | 0.625 | 2 | 0.500 | 2 |
| AutoDiff | 0.500 | 2 | 0.875 | 3 | 0.125 | 0 |
| TVAE | 0.500 | 2 | 0.875 | 3 | 0.125 | 0 |
| Identity | 0.875 | 3 | 1.000 | 4 | 0.500 | 2 |
| AIM | 0.875 | 3 | 0.125 | 0 | 0.375 | 1 |
| SMOTE | 0.857 | 3 | 0.857 | 3 | 0.714 | 2 |

---

## 3. Discrepancy Analysis: Config vs Empirical

### 3.1 Systematic Patterns

**correlation_handling is overestimated across the board.**
The config file assumed high correlation handling (avg ~2.5 across models) but empirical results show much lower performance (avg ~1.1). The worst offenders:

| Model | Config corr | Empirical corr | Delta |
|-------|------------|---------------|-------|
| AutoDiff | 3 | 0 | -3 |
| TVAE | 2 | 0 | -2 |
| TabDDPM | 4 | 2 | -2 |
| ARF | 2 | 1 | -1 |

**skew_handling is underestimated across the board.**
Most models achieve skew_preservation >= 0.875 (score 3), but the config often listed them at 1-2:

| Model | Config skew | Empirical skew | Delta |
|-------|------------|---------------|-------|
| DPCART | 1 | 3 | +2 |
| AIM | 1 | 3 | +2 |
| TVAE | 1 | 2 | +1 |
| TabDDPM | 4 | 3 | -1 |
| AutoDiff | 3 | 2 | -1 |

### 3.2 Per-Model Detailed Comparison

#### TabDDPM (most inflated)
| Capability | Config | Empirical | Gap |
|-----------|--------|-----------|-----|
| skew_handling | 4 | 3 | -1 |
| cardinality_handling | 3 | 2 | -1 |
| correlation_handling | 4 | 2 | **-2** |
| avg_quality_score | — | 0.685 | Low |
| avg_fidelity | — | 0.386 | Very low |

Config treated TabDDPM as a top-tier model. Benchmarks show it has the second-worst fidelity (0.386) among all tested models, and only moderate quality (0.685).

#### AutoDiff (mixed direction)
| Capability | Config | Empirical | Gap |
|-----------|--------|-----------|-----|
| skew_handling | 3 | 2 | -1 |
| cardinality_handling | 2 | 3 | +1 |
| correlation_handling | 3 | 0 | **-3** |

Correlation handling was severely overestimated. Cardinality was underestimated.

#### TVAE (correlation collapse)
| Capability | Config | Empirical | Gap |
|-----------|--------|-----------|-----|
| skew_handling | 1 | 2 | +1 |
| cardinality_handling | 2 | 3 | +1 |
| correlation_handling | 2 | 0 | **-2** |

#### AIM (skew surprise)
| Capability | Config | Empirical | Gap |
|-----------|--------|-----------|-----|
| skew_handling | 1 | 3 | **+2** |
| cardinality_handling | 1 | 0 | -1 |
| correlation_handling | 1 | 1 | 0 |

AIM preserves skew well despite being a DP model, but completely fails on cardinality.

### 3.3 Models Missing from Config

Five models had benchmark data but were absent from the config registry:

| Model | Quality | Fidelity | Utility | Notable |
|-------|---------|----------|---------|---------|
| BayesianNetwork | 0.974 | 0.956 | 0.985 | Top-3 overall quality |
| CART | 0.989 | 0.978 | 0.996 | Highest non-baseline quality |
| NFlow | 0.924 | 0.845 | 0.983 | Perfect skew preservation (1.0) |
| SMOTE | 0.979 | 0.968 | 0.981 | Best correlation preservation |
| Identity | 0.989 | 0.982 | 0.992 | Baseline reference |

These models are now included in v6.

---

## 4. Quality Ranking from Trial4

Overall model quality ranking based on `avg_quality_score`:

| Rank | Model | Quality | Fidelity | Utility | Privacy |
|------|-------|---------|----------|---------|---------|
| 1 | Identity* | 0.989 | 0.982 | 0.992 | 0.058 |
| 2 | CART | 0.989 | 0.978 | 0.996 | 0.057 |
| 3 | SMOTE | 0.979 | 0.968 | 0.981 | 0.090 |
| 4 | BayesianNetwork | 0.974 | 0.956 | 0.985 | 0.132 |
| 5 | ARF | 0.971 | 0.950 | 0.984 | 0.209 |
| 6 | NFlow | 0.924 | 0.845 | 0.983 | 0.320 |
| 7 | TVAE | 0.796 | 0.710 | 0.869 | 0.218 |
| 8 | DPCART | 0.763 | 0.718 | 0.851 | 0.241 |
| 9 | TabDDPM | 0.685 | 0.386 | 0.910 | 0.548 |
| 10 | AutoDiff | 0.559 | 0.360 | 0.702 | 0.387 |
| 11 | AIM | 0.537 | 0.323 | 0.781 | 1.000 |

*Identity is the baseline (copy of original data).

---

## 5. v6 Capability Score Changes

### Benchmarked Models (scores updated from trial4)

| Model | skew | card | zipfian | small | corr | dp | Source |
|-------|------|------|---------|-------|------|----|--------|
| DPCART | 3 | 2 | 2 | 3 | 1 | 3 | trial4 derived |
| BayesianNetwork | 3 | 3 | 2 | 2 | 2 | 0 | trial4 derived (NEW) |
| ARF | 3 | 3 | 3 | 4 | 1 | 0 | trial4 derived |
| CART | 3 | 4 | 2 | 2 | 2 | 0 | trial4 derived (NEW) |
| NFlow | 4 | 3 | 2 | 2 | 0 | 0 | trial4 derived (NEW) |
| TabDDPM | 3 | 2 | 2 | 2 | 2 | 0 | trial4 derived |
| AutoDiff | 2 | 3 | 2 | 2 | 0 | 0 | trial4 derived |
| TVAE | 2 | 3 | 1 | 3 | 0 | 0 | trial4 derived |
| Identity | 3 | 4 | 2 | 2 | 2 | 0 | trial4 derived (NEW) |
| AIM | 3 | 0 | 1 | 2 | 1 | 4 | trial4 derived |
| SMOTE | 3 | 3 | 2 | 2 | 2 | 0 | trial4 derived (NEW) |

### Non-Benchmarked Models (scores retained from previous config)

| Model | skew | card | zipfian | small | corr | dp | Source |
|-------|------|------|---------|-------|------|----|--------|
| GReaT | 4 | 4 | 4 | 2 | 3 | 0 | literature (no trial4 data) |
| TabTree | 3 | 4 | 4 | 4 | 3 | 0 | literature (no trial4 data) |
| TabSyn | 3 | 3 | 3 | 2 | 3 | 0 | literature (no trial4 data) |
| CTGAN | 1 | 3 | 2 | 2 | 2 | 0 | literature (no trial4 data) |
| GaussianCopula | 1 | 1 | 1 | 4 | 2 | 0 | literature (no trial4 data) |
| PATE-CTGAN | 1 | 2 | 2 | 1 | 2 | 4 | literature (no trial4 data) |

### Notes on Non-Derived Scores

- **zipfian_handling**: No Zipfian-specific benchmarks in trial4. Retained from previous config where available; defaulted to 2 for newly added models.
- **small_data**: All trial4 datasets appear to be >= 1000 rows, so no small-data scores were derivable. Retained from previous config.
- **privacy_dp**: Static model property, unchanged.

---

## 6. Impact on Rankings

### Updated `model_ranking_by_capability`

Rankings (score >= 3 to qualify):

| Capability | v6 Ranking |
|-----------|-----------|
| skew_handling | GReaT(4), NFlow(4), TabTree(3), TabDDPM(3), TabSyn(3), DPCART(3), BayesianNetwork(3), ARF(3), CART(3), AIM(3), SMOTE(3), Identity(3) |
| cardinality_handling | GReaT(4), TabTree(4), CART(4), Identity(4), TabSyn(3), CTGAN(3), ARF(3), BayesianNetwork(3), NFlow(3), AutoDiff(3), TVAE(3), SMOTE(3) |
| zipfian_handling | GReaT(4), TabTree(4), TabSyn(3), ARF(3) |
| small_data | GaussianCopula(4), TabTree(4), ARF(4), TVAE(3), DPCART(3), TabSyn(3-lit) |
| correlation_handling | GReaT(3), TabTree(3), TabSyn(3) |
| privacy_dp | PATE-CTGAN(4), AIM(4), DPCART(3) |

**Major ranking shifts:**
- correlation_handling: Previously 5+ models at score 3-4. Now only 3 non-benchmarked models qualify (GReaT, TabTree, TabSyn). All benchmarked models scored <= 2.
- skew_handling: Most benchmarked models now qualify at score 3 (previously many at 1).
- cardinality_handling: More models qualify due to empirical data showing better performance than assumed.

---

## 7. Recommendations

1. **Prioritize trial4-calibrated scores** for all recommendation decisions involving benchmarked models.
2. **Benchmark GReaT, TabTree, TabSyn, CTGAN, GaussianCopula, PATE-CTGAN** on the same 8 datasets to replace their literature-based scores.
3. **Investigate correlation_handling** further - the universally low scores may indicate a measurement issue in the benchmark pipeline, or may genuinely reflect that most models struggle with inter-column correlations.
4. **Add Zipfian-specific benchmarks** to the evaluation suite to replace the default score of 2.
5. **Add small-data benchmarks** (datasets < 500 rows) to validate the small_data scores.


## 8. Must have
*1. weighted average of current model* (engine)  1 x a + 1 x b +  1 x c + 1 x d = weighted factor (1)
2. intention aware - best *correlation handling* (scoring system), best DP, Utilty, and Fiedelity.  weight = 1, (interface)
3. *calculate DP* - normalize the dataset normalize the DP score = (small(min) - large(max)) / level 
4. missing data (optional)