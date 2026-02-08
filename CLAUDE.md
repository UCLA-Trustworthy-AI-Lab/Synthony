# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Synthony** is an intelligent orchestration platform that recommends the optimal synthetic tabular data generation model from 15 SOTA models based on dataset characteristics. It analyzes data "stress factors" (skewness, cardinality, Zipfian distributions) and matches them to model capabilities using a hybrid rule-based + LLM decision engine.

## Common Commands

### Installation

```bash
pip install -e .                  # Core only
pip install -e ".[cli]"           # CLI tools (typer, rich)
pip install -e ".[api]"           # FastAPI server
pip install -e ".[llm]"           # LLM support (requires OPENAI_API_KEY)
pip install -e ".[mcp]"           # MCP server
pip install -e ".[all]"           # Everything
pip install -e ".[dev]"           # Development (pytest, black, ruff, mypy)
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
```

### Code Quality

```bash
black src/ tests/                         # Format
ruff check src/ tests/                    # Lint
mypy src/                                 # Type check
```

### Servers

```bash
# FastAPI REST server
uvicorn synthony.api.server:app --reload
# Then visit http://localhost:8000/docs for API documentation

# MCP server (for AI agent integration)
python -m mcp_server.server --verbose
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
| `src/synthony/api/` | FastAPI REST server with SQLite persistence |
| `src/synthony/benchmark/` | Data quality metrics (KL/JS divergence, fidelity, utility, privacy) |
| `mcp_server/` | MCP protocol server for AI agent integration |

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

`config/model_capabilities.json` (v7.0.0) tracks 15 models with 0-4 capability scores calibrated from spark benchmarks (10 datasets, 14 models) with density-normalized cardinality scoring:
- **Diffusion**: TabDDPM, TabSyn, AutoDiff
- **LLM/Transformer**: GReaT
- **GAN**: CTGAN, PATECTGAN
- **VAE**: TVAE
- **Tree-based**: ARF, CART, DPCART
- **Statistical**: BayesianNetwork, NFlow, SMOTE, AIM
- **Baseline**: Identity

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
| `OPENAI_API_KEY` | Required for LLM-based recommendations |
| `MCP_DEBUG` | Enable verbose MCP server logging |

## Key Files

- `src/synthony/core/schemas.py` - Pydantic models: `DatasetProfile`, `StressFactors`, `RecommendationResult`
- `src/synthony/utils/constants.py` - `AnalyzerConfig` with all configurable thresholds
- `config/SystemPrompt.md` - LLM system prompt v5.0 (canonical, used by engine)
- `docs/scoring_methodology.md` - Capability scoring formulas and engine pipeline
- `docs/ISSUE_recommender_accuracy_v7.md` - Resolved accuracy issue report

## MCP Server Integration

The MCP server exposes tools for AI agents:

| Tool | Purpose |
|------|---------|
| `list_datasets` | Discover datasets in configured directory |
| `load_dataset` | Load and preview dataset metadata |
| `analyze_stress_profile` | Profile dataset stress factors |
| `rank_models_hybrid` | Get recommendations (rule + LLM) |
| `rank_models_rule` | Rule-based recommendations only |
| `rank_models_llm` | LLM-based recommendations only |

Test MCP protocol:
```bash
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | python -m mcp_server.server
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
```
