# NL-Table Synthesizer Classifier - Master Architecture (v3)

**Version:** 3.0
**Date:** 2026-02-08

> **Historical planning document — superseded.** Written before the
> project was renamed to **Synthony**, and describes a "Three-Package
> Ecosystem" that was never built. The current codebase is a single
> `synthony` package (`src/synthony/`); `table-synthesizers` and
> `synthcity` are external model-implementation sources referenced via
> `package` metadata in `model_capabilities.json`, not sub-packages of
> this repo. The registry has 15 implemented models (not 16
> planned/13 implemented — SMOTE is implemented, `LTM_VAE` was never
> added). The Zipfian threshold below is also wrong: it's `>0.80` in
> `src/synthony/utils/constants.py`, not `>0.05`. See `CLAUDE.md` for
> the current architecture.

## 1. Project Vision

The **NL-Table Synthesizer Classifier (NLTSC)** is an intelligent orchestration platform designed to democratize access to privacy-preserving synthetic data. It acts as a bridge between raw tabular data and State-of-the-Art (SOTA) synthesis models.

**Model Count**: 16 Planned / 13 Implemented (SMOTE, LTM_VAE pending).

## 2. System Architecture: The Three-Package Ecosystem

The system is strictly modularized to separate **Data Profiling**, **Model Training**, and **Decision Intelligence**.

### Package 1: `nl-table-data` (Data Infrastructure)
* **Role**: A standalone library for robust data ingestion and statistical profiling.
* **Key Responsibility**: Converting raw CSV/Parquet into a "Stress Profile" (Skewness, Cardinality, Correlation Complexity).
* **Core Class**: `StochasticDataAnalyzer`.

### Package 2: `table-synthesizers` (External Core & Interface)
* **Role**: The heavy-lifting training engine.
* **Key Responsibility**: Hosting the models (TabDDPM, GReaT, CTGAN, etc.).
* **Integration**: NLTSC maintains a **Shadow Interface** (`model_capabilities.json`) to understand these models without importing their heavy dependencies.

### Package 3: `nl-table-classifier-api` (Orchestration)
* **Role**: The brain of the operation.
* **Key Responsibility**: Merging the Data Profile (Pkg 1) with Model Capabilities (Pkg 2) using a Hybrid Rule-Based + LLM engine to generate recommendations.

## 3. Core Intelligence Engine

The decision logic handles the "Hard Problems" of tabular synthesis through a multi-stage funnel:

| Component | Function | Key Thresholds |
| :--- | :--- | :--- |
| **Hard Filters** | Eliminates impossible options. | `gpu_recommendation` (Low/Med/High), `strict_dp` (Differential Privacy), `large_data` (>50k rows). |
| **Stress Detectors** | Identifies data difficulty. | **Severe Skew** (>2.0), **Zipfian Ratio** (>0.05), **High Cardinality** (>500). |
| **Hybrid Scorer** | Ranks models 0-4. | Uses `SystemPrompt.md` knowledge base. |
| **Tie-Breaker** | Resolves conflicts. | Prioritizes **ARF** for small data; **GReaT** for "Hard Problems" (Skew+Zipfian). |

## 4. Benchmark-Driven Feedback Loop

The system is designed to be self-correcting.

1. **Analyze**: User uploads data.
2. **Recommend**: System suggests a model based on current scores.
3. **Validate (Offline)**: We run benchmarks (WD, TVD) on "Synthetic Control Datasets".
4. **Refine**: If empirical results differ from theoretical scores, `model_capabilities.json` is updated.
