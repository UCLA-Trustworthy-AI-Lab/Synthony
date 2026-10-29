# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Synthony** is an intelligent orchestration platform that recommends the optimal synthetic tabular data generation model from 15 SOTA models based on dataset characteristics. It analyzes data "stress factors" (skewness, cardinality, Zipfian distributions) and matches them to model capabilities using a hybrid rule-based + LLM decision engine.

**Published at the 2nd DeLTa Workshop, ICLR 2026** ([Paper](https://openreview.net/forum?id=cj4SNumWqf))

## Common Commands

### Installation

> **This project is installed under the anaconda `base` environment**
> (`/opt/anaconda3/bin/python3`), not any other Python on this machine —
> there are at least two other unrelated interpreters installed (Homebrew's
> `python3.13`, python.org's `python3.12`) that do **not** have Synthony's
> dependencies. `python3`/`pip3` are aliased to anaconda in `~/.zshrc`, but
> that alias only applies in an interactive zsh shell that sources it —
> scripts, other shell types, or some IDE terminals can silently fall
> through to a different `python3` on `PATH` and fail with
> `ModuleNotFoundError` for things that are clearly installed. If that
> happens, activate the environment explicitly rather than relying on the
> alias:
> ```bash
> source /opt/anaconda3/etc/profile.d/conda.sh && conda activate base
> ```

```bash
pip install -e .                  # Core only
pip install -e ".[cli]"           # CLI tools (typer, rich)
pip install -e ".[api]"           # FastAPI server
pip install -e ".[llm]"           # LLM support (requires OPENAI_API_KEY)
pip install -e ".[mcp]"           # MCP server
pip install -e ".[all]"           # Everything
pip install -e ".[dev]"           # Development (pytest, black, ruff, mypy)
pip install -e ".[test]"          # Test suite (pytest, pytest-asyncio, mcp, openai)
pip install -e ".[polars]"        # Polars dataframe support
```

### CLI Commands

```bash
# Profile a dataset
synthony-profile data.csv --verbose
synthony-profile data.csv -o profile.json

# Compare original vs synthetic data quality
synthony-benchmark -r original.csv -s synthetic.csv --verbose
synthony-benchmark -r original.csv -s synthetic.csv -o results.json

# Get model recommendation
synthony-recommender -i data.csv --method hybrid
synthony-recommender -i data.csv --method rulebased --cpu-only
synthony-recommender -i data.csv --method llm --strict-dp
```

### Running Tests

```bash
pytest                                    # All tests with coverage
pytest tests/unit/ -v                     # Unit tests only
pytest tests/integration/ -v              # Integration tests only
pytest tests/unit/test_skewness_detector.py::test_detect_severe_skew -v  # Single test
pytest -m "not requires_llm"              # Exclude LLM-dependent tests
pytest --cov=synthony --cov-report=html   # Coverage with HTML report

# MCP server tests are a separate pytest root, not under tests/:
cd mcp_server && pytest test_server.py test_protocol.py test_agent_tools.py -v

# Or use the wrapper (mirrors cicd.yml exactly, incl. the MCP step above):
./scripts/run_tests.sh --all            # everything CI runs
./scripts/run_tests.sh --mcp            # just MCP
./scripts/run_tests.sh --all --slow     # + real synthcity subprocess training (tens of seconds)
```

> **Note:** 6 tests in `tests/integration/test_api_integration.py` are skipif-gated on a live API server at `localhost:8000` (health check, upload/analyze, session retrieval, storage stats, system prompt list/upload). Without a running server they skip silently; start one first (`uvicorn synthony.api.server:app --port 8000`) for full integration coverage. Check for and kill any stale server already bound to that port before starting a fresh one -- a leftover process from a previous run can make failures look like application bugs.

> **Note:** Two Agentic Mode tests (`tests/functional/test_synthcity_runner_e2e.py`) run real synthcity training/cancellation and are gated on `SYNTHONY_RUN_SLOW_TESTS=1` (in addition to `@pytest.mark.slow`, since no pytest invocation in this repo passes `-m "not slow"` -- the env var is what actually keeps them out of CI). See `docs/TESTING.md` for the full test-category reference and `docs/agents.md` for per-phase testing/verification detail (Agentic Mode Phases 1-6).

### Code Quality

```bash
black src/ tests/                         # Format
ruff check src/ tests/                    # Lint
mypy src/                                 # Type check
```

### Servers

```bash
# FastAPI REST server (via entry point or uvicorn)
synthony-api
uvicorn synthony.api.server:app --reload
# Then visit http://localhost:8000/docs for API documentation

# MCP server (via entry point or module)
synthony-mcp
python -m mcp_server.server --verbose
```

### Docker

```bash
docker build -t synthony .                    # Production container
docker build -f Dockerfile.mcp -t synthony-mcp .  # MCP server container
docker-compose up                             # Full stack
```

## Architecture

### Data Flow

```
Input (CSV/Parquet)
    ↓
StochasticDataAnalyzer.analyze()         # src/synthony/core/analyzer.py
    ├── SkewnessDetector                  # detectors/skewness.py
    ├── CardinalityDetector               # detectors/cardinality.py (+ Zipfian)
    ├── CorrelationDetector               # detectors/correlation.py
    └── DataSizeClassifier                # detectors/data_size.py
    ↓
DatasetProfile (Pydantic)                 # core/schemas.py
    ↓
ModelRecommendationEngine.recommend()    # recommender/engine.py
    ├── Load config from registry (thresholds, priorities)
    ├── Apply hard filters (cpu_only, strict_dp, exclude, row limits)
    ├── Hard Problem detection (skew + cardinality + zipfian)
    ├── Capability scoring + empirical quality bonus
    ├── GPU/CPU-aware tie-breaking
    ├── [Optional] LLM scoring (with SystemPrompt)
    └── [Optional] Focus/scale_factors for custom weighting
    ↓
RecommendationResult
```

### Key Components

| Directory | Purpose |
|-----------|---------|
| `src/synthony/core/` | Data loading (`loaders.py`), analysis (`analyzer.py`), schemas (`schemas.py`) |
| `src/synthony/detectors/` | Stress detection: skewness, cardinality, correlation, data_size |
| `src/synthony/recommender/` | Recommendation engine + model capabilities registry |
| `src/synthony/api/` | FastAPI REST server with SQLite persistence; `run_endpoints.py` exposes agentic-mode run history (`GET /runs*`) |
| `src/synthony/llm/` | Provider-agnostic LLM client (OpenAI, Anthropic, vLLM) — see `docs/agents.md` |
| `src/synthony/agents/` | Agentic-mode agents: `RunManager`/`Orchestrator` state machine, `RecommenderAgent` multi-turn reasoning, synthesizer/benchmark steps, `StepRecorder` monitoring — see `docs/agents.md` |
| `src/synthony/runners/` | Subprocess-isolated synthesizer training (synthcity plugins, registry, real cancellation) — see `docs/agents.md` |
| `src/synthony/benchmark/` | Data quality metrics (KL/JS divergence, fidelity, utility, privacy) |
| `src/synthony/utils/` | Configuration constants (`AnalyzerConfig`), thresholds |
| `mcp_server/` | MCP protocol server for AI agent integration (tools, resources, prompts) |
| `config/` | Model capabilities registry (`model_capabilities.json`), LLM system prompt |
| `scripts/` | Optimization scripts (`optimize_scaling.py` for Bayesian calibration) |
| `ablation/` | Ablation study runner (`run_ablations.py`) |
| `baselines/` | Non-Synthony comparison baselines: static heuristic, vanilla LLM, random search, meta-feature kNN (`run_baselines.py` orchestrates all four) |
| `tests/` | Unit, integration, functional, evaluation, and regression tests |

### Stress Detection Thresholds

Defined in `src/synthony/utils/constants.py`:

| Factor | Threshold | Impact |
|--------|-----------|--------|
| Severe Skew | \|skewness\| > 2.0 | Breaks basic GANs/VAEs |
| High Cardinality | unique > 500 | Mode collapse risk |
| Zipfian Distribution | Top 20% > 80% | Requires specialized tokenization |
| Small Data | rows < 1,000 | Overfitting risk → prefer ARF |
| Large Data | rows > 50,000 | LLMs impractical due to latency |

### Model Registry

`config/model_capabilities.json` (v7.0.2) tracks 15 models with 0-4 capability scores calibrated from spark benchmarks (10 datasets, 13 benchmarked models) with density-normalized cardinality scoring:
- **Diffusion**: TabDDPM, TabSyn, AutoDiff
- **LLM/Transformer**: GReaT
- **GAN**: CTGAN, PATECTGAN
- **VAE**: TVAE
- **Tree-based**: ARF, CART, DPCART
- **Statistical**: BayesianNetwork, NFlow, SMOTE, AIM
- **Baseline**: Identity

> **Note:** `ModelRecommendationEngine` defaults to loading `src/synthony/recommender/model_capabilities.json` (a package-local copy), not `config/model_capabilities.json` directly. `scripts/generate_model_capabilities.py` only writes to `config/model_capabilities.json`. After regenerating the registry, manually sync the file into `src/synthony/recommender/` (or pass `capabilities_path=` explicitly) for the engine to pick up the changes.

The registry also stores all engine configuration:
- `metadata.dp_threshold` — DP filter threshold (default: 3)
- `metadata.capability_thresholds` — 11 scoring thresholds
- `metadata.hard_problem_confidence` — confidence scores for hard problem path
- `metadata.score_decay` — match score decay curve (1.0/0.7/0.4/0.0)
- `hard_problem_routing` — primary model, large data fallback, fallback priority
- `tie_breaking_priority` — GPU/CPU/speed/small_data priority lists

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SYNTHONY_DATA_DIR` | Dataset directory (default: `dataset/input_data`) |
| `OPENAI_API_KEY` | Required for LLM-based recommendations (legacy path, or `synthony.llm` `provider="openai"`) |
| `ANTHROPIC_API_KEY` | Required for `synthony.llm` `provider="anthropic"` |
| `SYNTHONY_LLM_PROVIDER` | `synthony.llm` provider: `openai` (default), `anthropic`, or `vllm` |
| `SYNTHONY_LLM_MODEL` | Model override for `synthony.llm`'s selected provider |
| `MCP_DEBUG` | Set to any non-empty value to enable verbose MCP server logging (equivalent to `--verbose` flag) |

## Key Files

- `src/synthony/core/schemas.py` - Pydantic models: `DatasetProfile`, `StressFactors`, `RecommendationResult`
- `src/synthony/core/analyzer.py` - `StochasticDataAnalyzer` main profiling class
- `src/synthony/recommender/engine.py` - `ModelRecommendationEngine` core scoring logic (1,400+ lines)
- `src/synthony/utils/constants.py` - `AnalyzerConfig` with all configurable thresholds
- `config/model_capabilities.json` - Model registry v7.0.2 (15 models × 6 capability dims)
- `config/SystemPrompt.md` - LLM system prompt v6.0 (canonical, generated by `synthony_update_system_prompt`)
- `scripts/optimize_scaling.py` - Bayesian optimization of scale factors (Optuna/TPE)
- `ablation/run_ablations.py` - Ablation study experiment runner
- `baselines/run_baselines.py` - Runs all 4 non-Synthony baselines and produces a comparison table
- `docs/scoring_methodology.md` - Capability scoring formulas (one stage of the pipeline below)
- `docs/capability_registry_pipeline.md` - How `config/model_capabilities.json` actually gets built: table-synthesizers simulation → dual measurement (profile-comparison + oracle-ranking) → Bayesian calibration → runtime use
- `docs/agents.md` - Agentic Mode reference: all 6 phases (LLM client → CLI/hardening), per-phase verification evidence
- `docs/realy_integration_plan.md` - Multi-agent plan for integrating Synthony into Realy (`table_reasoning`) as an optional RS adapter — parked pending PI sign-off

## MCP Server Integration

The MCP server exposes tools for AI agents:

| Tool | Purpose |
|------|---------|
| `synthony_list_datasets` | Discover datasets in configured directory |
| `synthony_load_dataset` | Load and preview dataset metadata |
| `synthony_analyze_stress_profile` | Profile dataset stress factors |
| `synthony_rank_models_hybrid` | Get recommendations (rule + LLM) |
| `synthony_rank_models_rule` | Rule-based recommendations only |
| `synthony_rank_models_llm` | LLM-based recommendations only |
| `synthony_benchmark_compare` | Compare synthetic vs original data quality |
| `synthony_get_model_info` | Model capabilities and constraints |
| `synthony_check_model_constraints` | Validate model compatibility (row count) |
| `synthony_get_tie_breaker_logic` | Resolve ties when top models score within 5% |
| `synthony_explain_recommendation_reasoning` | Explain why a model was recommended |
| `synthony_generate_benchmark_dataset` | Create test datasets with known stress |
| `synthony_list_models` | List all 15 models with optional type filter |
| `synthony_update_model_capabilities` | Update capability scores / empirical metrics in model_capabilities.json |
| `synthony_update_system_prompt` | Regenerate SystemPrompt.md from current model_capabilities.json scores, store in DB |

Test MCP protocol:
```bash
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | python -m mcp_server.server
# Tool names now use synthony_ prefix (e.g. synthony_analyze_stress_profile)
```

## Python API Usage

```python
from synthony import StochasticDataAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine

# Profile dataset
analyzer = StochasticDataAnalyzer()
profile = analyzer.analyze("data.csv")

# Check stress factors
print(f"Severe Skew: {profile.stress_factors.severe_skew}")
print(f"High Cardinality: {profile.stress_factors.high_cardinality}")

# Get recommendation
engine = ModelRecommendationEngine()
result = engine.recommend(profile, method="rule_based")
print(f"Recommended: {result.recommended_model.model_name}")

# With intent-conditioned scale factors
result = engine.recommend(profile, method="rule_based", focus="privacy")
result = engine.recommend(profile, method="rule_based", focus="fidelity")

# With constraints
result = engine.recommend(
    profile,
    method="rule_based",
    constraints={"cpu_only": True, "strict_dp": True},
)
```

## Entry Points

Defined in `pyproject.toml`:

| Command | Source | Purpose |
|---------|--------|---------|
| `synthony-profile` | `synthony.cli:profile_command` | Profile dataset stress factors |
| `synthony-benchmark` | `synthony.cli:benchmark_command` | Compare original vs synthetic |
| `synthony-recommender` | `synthony.cli:recommender_command` | Get model recommendations |
| `synthony-api` | `synthony.api.server:main` | Start FastAPI REST server |
| `synthony-mcp` | `mcp_server.server:main` | Start MCP server |
