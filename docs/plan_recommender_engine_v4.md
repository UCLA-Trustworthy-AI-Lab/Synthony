# Recommender Engine Update Plan - v4

**Version:** 4.0
**Date:** 2026-02-08
**Status:** In Progress / Partially Complete

> **Update:** `SMOTE` has since been added to `model_capabilities.json`
> (currently at registry version `7.0.0`, 15 models total) — the
> "Remaining Gaps" and "New Models (Batch 2)" rows below are stale.
> `LTM_VAE` was never pursued and is not on the current roadmap. Also,
> the actual GPU constraint field is `requires_gpu` (boolean), not
> `gpu_recommendation` (low/medium/high) as described here.

---

## 1. Executive Summary

Update the `ModelRecommendationEngine` to finalize **v3 architecture** changes. Much of the v3 plan has been implemented, but gaps remain.

**Key Achievements (v3 -> v4)**:
- Implemented `gpu_recommendation` logic.
- Added major models: `Identity`, `CART`, `BayesianNetwork`, `NFlow`, `AIM`, `DPCART`.
- Removed non-existent models: `TabTree`, `GaussianCopula`.

**Remaining Gaps**:
- `SMOTE`: Not yet in `model_capabilities.json`.
- `LTM_VAE`: Not yet in `model_capabilities.json`.

---

## 2. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| `gpu_recommendation` | ✅ Complete | Using "low", "medium", "high" |
| CPU-only filter | ✅ Complete | Filters based on `gpu_recommendation != high` |
| New Models (Batch 1) | ✅ Complete | CART, BayesianNetwork, NFlow, AIM, DPCART, Identity |
| **New Models (Batch 2)** | ❌ Pending | **SMOTE**, **LTM_VAE** |
| Hard Problem Fallback | ⚠️ Partial | Uses `TabSyn`, `ARF`, `NFlow`, but `LTM_VAE` missing |

---

## 3. Remaining Work

### 3.1 Add Missing Models
- **SMOTE**: Need to add capabilities and empirical scores to `model_capabilities.json`.
- **LTM_VAE**: Need to integrate as a high-capacity fallback for hard problems.

### 3.2 Update `model_capabilities.json`
- Current Version: `7.0.0`
- Target Version: `7.1.0` (with SMOTE + LTM_VAE)

### 3.3 Verify Tie-Breaking
- Ensure `SMOTE` is correctly prioritized for small data / imbalanced datasets once added.
- Ensure `LTM_VAE` is correctly set as a fallback for large datasets.
