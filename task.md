# Task Tracker

## Phase 1: Model Capability Scoring v6 Calibration (DONE)

### Context
The model capability scores in `config/model_capabilities.json` and `src/synthony/recommender/model_capabilities.json` were based on literature assumptions and limited benchmarks. Trial4 benchmark results (`output/benchmark/trial4/analysis_comparison.json`) across 8 datasets and 11 models revealed significant discrepancies between assumed and empirically-derived capability scores.

### Completed Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Inspect model_capabilities against trial4 benchmarks | DONE | Identified systematic overestimation of correlation_handling and underestimation of skew_handling |
| 2 | Review scoring methodology (docs/scoring_methodology.md) | DONE | Methodology is internally consistent; thresholds verified |
| 3 | Write Scoring_system_inspection_summary_v6.md | DONE | docs/Scoring_system_inspection_summary_v6.md |
| 4 | Update config/model_capabilities.json to v6 | DONE | 17 models, benchmark-calibrated scores |
| 5 | Update src/synthony/recommender/model_capabilities.json to v6 | DONE | Runtime copy synced |
| 6 | Verify model_ranking_by_capability reflects new scores | DONE | Rankings regenerated |

---

## Phase 2: Engine + SystemPrompt Alignment (DONE)

### Context
Cross-inspection of `engine.py`, `config/SystemPrompt.md`, and `constants.py` against the updated v6 model_capabilities revealed 13 misalignment issues. The SystemPrompt knowledge base, engine tie-breaking logic, and hard problem routing all referenced stale scores and excluded models.

### Completed Tasks

| # | Issue | Severity | Status | File(s) Changed |
|---|-------|----------|--------|-----------------|
| 7 | SystemPrompt scores didn't match v6 model_capabilities | HIGH | DONE | `config/SystemPrompt.md` -> v4.0 |
| 8 | SystemPrompt missing correlation_handling column | HIGH | DONE | Added `Corr` column to table |
| 9 | Hard problem fallback had 2/3 excluded models | HIGH | DONE | `engine.py` fallback -> [ARF, CART, BayesianNetwork] |
| 10 | Quality tie-break listed TabDDPM first (ranked #9) | HIGH | DONE | `engine.py` -> [CART, SMOTE, BayesianNetwork, ARF, NFlow] |
| 11 | Small data tie-break had 3/4 excluded models | HIGH | DONE | `engine.py` -> [ARF, CART, BayesianNetwork, SMOTE] |
| 12 | Speed tie-break had 3/5 excluded models | MEDIUM | DONE | `engine.py` -> [CART, ARF, SMOTE, TVAE, DPCART] |
| 13 | Hard problem large-data routed to TabDDPM (quality=0.685) | HIGH | DONE | Removed special-case routing; uses fallback list for all |
| 14 | small_data threshold: 500 (analyzer) vs 1000 (engine) | MEDIUM | DONE | `engine.py` aligned to 500 |
| 15 | GPU filter binary vs 3-tier mismatch | MEDIUM | DONE | SystemPrompt v4.0 uses binary GPU yes/no matching engine |
| 16 | Quick Reference table had stale recommendations | MEDIUM | DONE | Rebuilt with v6 scores, active models only |
| 17 | large_data scored dimension removed from SystemPrompt | MEDIUM | DONE | Removed; handled by max_recommended_rows constraint |
| 18 | docs/SystemPrompt_v3.1.md out of sync | LOW | DONE | Created docs/SystemPrompt_v4.0.md |
| 19 | SMOTE/NFlow 7/8 datasets noted | LOW | DONE | Documented in model_capabilities.json limitations |

### Additional Updates
- `CLAUDE.md` updated: SystemPrompt reference v3.1 -> v4.0, model_capabilities reference v5 -> v6

---

## Summary of All Files Changed

| File | Action | Version |
|------|--------|---------|
| `config/model_capabilities.json` | Updated scores, added 5 models | v6.0.0 |
| `src/synthony/recommender/model_capabilities.json` | Synced with config | v6.0.0 |
| `config/SystemPrompt.md` | Rewritten with v6 scores | v4.0 |
| `docs/SystemPrompt_v4.0.md` | New copy of SystemPrompt | v4.0 |
| `docs/Scoring_system_inspection_summary_v6.md` | New inspection report | - |
| `src/synthony/recommender/engine.py` | Fixed fallbacks, tie-breaking, threshold | - |
| `CLAUDE.md` | Updated references | - |
| `task.md` | This file | - |
