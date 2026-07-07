# System Architecture Update (v5)

**Date:** 2026-02-08
**Status:** Partial Implementation of v3 Recommender Plan

> **Historical snapshot, not current state.** This describes the plan as
> of 2026-02-08. As of the current codebase: there is one unified
> `synthony` package (not a 3-package split — `table-synthesizers` and
> `synthcity` are external model-implementation sources referenced by
> `package` metadata in `model_capabilities.json`, not sub-packages of
> this repo); the registry has 15 models, all implemented (including
> SMOTE); `LTM_VAE` was never added; and the GPU constraint field is
> `requires_gpu` (boolean), not `gpu_recommendation`. See `CLAUDE.md` for
> the current architecture.

---

## 1. Documentation Map

This document serves as the master index for the current system architecture state.

| Component | Current Version | Description |
|-----------|-----------------|-------------|
| **Master Design** | [v3](DESIGN_MODEL_OPERATIONS_AND_EVALUATION_v3.md) | Overview of the 3-package ecosystem and 16-model vision. |
| **Visual Guide** | [v1](VISUAL_ARCHITECTURE_v1.md) | Simplified Mermaid diagrams for architecture and recommender logic. |
| **API Usage** | [v4](API_USAGE_v4.md) | Current API specification including `dataset_id` requirements and hybrid mode. |
| **Startup Guide** | [v3](API_STARTUP_GUIDE_v3.md) | Updated guide for running the server with current env vars. |
| **Recommender Plan** | [v4](plan_recommender_engine_v4.md) | Roadmap for remaining model implementations (SMOTE, LTM_VAE). |

## 2. Implementation Status

### Core System
- **Architecture**: Modular 3-package design is fully active.
- **API**: Working as described in `API_USAGE_v4`, with internal session tracking.

### Recommender Engine
- **Current State**: Transitioning from v2 to v3.
- **Implemented**: `gpu_recommendation` logic, CPU-only filtering, 13/16 models.
- **Pending**: `SMOTE` and `LTM_VAE` integration (see `plan_recommender_engine_v4.md`).

## 3. Next Steps
1. **Implement Remaining Models**: Add SMOTE and LTM_VAE to `model_capabilities.json`.
2. **Public Session API**: Expose session management endpoints (currently internal-only).
3. **Verify Fallbacks**: Test "Hard Problem" fallbacks with the full model suite.
