# Synthony

**Orchestrating the right synthetic data model for your tabular data.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)]()

Synthony is an intelligent recommendation platform that analyzes your tabular dataset's characteristics and recommends the optimal synthetic data generation model from 15 state-of-the-art options. Like a symphony conductor orchestrating instruments, Synthony orchestrates the right model for your data.

## Why Synthony?

Choosing the right synthetic data model is hard. Each model has strengths and weaknesses:

- **GANs** struggle with skewed distributions
- **VAEs** collapse on high-cardinality columns
- **LLMs** are too slow for large datasets
- **Tree-based models** excel at small data but miss complex correlations

Synthony solves this by:

1. **Profiling** your data to detect "stress factors" (skewness, cardinality, Zipfian distributions)
2. **Matching** those factors against model capabilities using a hybrid rule-based + LLM engine
3. **Recommending** the best model with clear reasoning

## Features

### Data Profiling

- **Severe Skewness Detection** — Fisher-Pearson coefficient > 2.0 breaks basic GANs/VAEs
- **High Cardinality Analysis** — >500 unique values risk mode collapse
- **Zipfian Distribution Detection** — Power-law concentration (top 20% > 80% of data)
- **Data Size Classification** — Small (<1000 rows) vs Large (>50k rows) constraints
- **Higher-Order Correlation Detection** — Dense but non-linear relationships

### Model Recommendation

- **15 SOTA Models** — CART, SMOTE, BayesianNetwork, ARF, NFlow, TVAE, TabDDPM, AutoDiff, AIM, DPCART, PATECTGAN, TabSyn, CTGAN, GReaT, Identity
- **Constraint Support** — CPU-only, differential privacy requirements
- **Hybrid Engine** — Rule-based scoring + LLM reasoning
- **Explainable Results** — Clear reasoning for every recommendation

### API & Integration

- **REST API** — FastAPI server with OpenAPI documentation
- **Python SDK** — Direct library usage
- **CLI Tools** — Command-line profiling and benchmarking
- **MCP Server** — Integration with AI agents (Claude, Cline, Continue.dev, Cursor)

## Installation

### Basic Installation

```bash
pip install -e .
```

### With All Features

```bash
# CLI tools
pip install -e ".[cli]"

# API server
pip install -e ".[api]"

# LLM-based recommendations (requires OpenAI API key)
pip install -e ".[llm]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from synthony import StochasticDataAnalyzer

# Analyze your dataset
analyzer = StochasticDataAnalyzer()
profile = analyzer.analyze_from_file("your_data.csv")

# Check detected stress factors
print(f"Severe Skew: {profile.stress_factors.severe_skew}")
print(f"High Cardinality: {profile.stress_factors.high_cardinality}")
print(f"Zipfian Distribution: {profile.stress_factors.zipfian_distribution}")
print(f"Small Data: {profile.stress_factors.small_data}")

# Get detailed metrics
if profile.skewness:
    print(f"Max Skewness: {profile.skewness.max_skewness:.2f}")
    print(f"Problematic Columns: {profile.skewness.severe_columns}")
```

### REST API

```bash
# Start the API server
python start_api.py

# Or with uvicorn directly
uvicorn synthony.api.server:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

**Analyze and get recommendations in one call:**

```bash
curl -X POST "http://localhost:8000/analyze-and-recommend" \
  -F "file=@your_data.csv" \
  -F "method=hybrid"
```

### CLI

```bash
# Profile a dataset
synthony-profile data.csv --verbose
synthony-profile data.csv -o profile.json

# Get model recommendation
synthony-recommender -i data.csv --method rulebased
synthony-recommender -i data.csv --cpu-only --strict-dp
synthony-recommender -i data.csv --method hybrid --skew-sf 2.0

# Compare original vs synthetic data quality
synthony-benchmark -r original.csv -s synthetic.csv --verbose
```

### MCP Server for AI Agents

Synthony includes an **MCP (Model Context Protocol) server** that integrates with AI agents like Claude Code, Cline, Continue.dev, and Cursor.

**Quick Install:**

```bash
# Install with MCP support
pip install -e ".[mcp]"

# Test the server
python -m mcp_server.server --test
```

**Connect to Claude Desktop (macOS):**

```bash
mkdir -p ~/Library/Application\ Support/Claude
cat > ~/Library/Application\ Support/Claude/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "synthony": {
      "command": "synthony-mcp",
      "env": {
        "SYNTHONY_DATA_DIR": "$(pwd)/dataset/input_data"
      }
    }
  }
}
EOF
```

Then restart Claude Desktop. You can now ask Claude to analyze datasets and get recommendations:

```
"Analyze the Bean dataset and recommend the best synthetic data model"
```

**Connect to Other Agents:**

- **Cline (VS Code)**: `.cline/config.json`
- **Continue.dev (VS Code)**: `.continue/config.json`
- **Cursor (AI Editor)**: Settings → MCP Servers

For detailed setup instructions across all platforms, including Windows, Linux, npm installation, Docker, and troubleshooting: **See [MCP_SETUP.md](docs/MCP_SETUP.md)**

## How It Works

### 1. Stress Detection

Synthony identifies data characteristics that break traditional models:

| Stress Factor | Threshold | Impact |
|--------------|-----------|--------|
| Severe Skew | \|skewness\| > 2.0 | Breaks basic GANs/VAEs |
| High Cardinality | unique > 500 | Mode collapse risk |
| Zipfian Distribution | Top 20% > 80% | Requires specialized tokenization |
| Small Data | rows < 1,000 | Overfitting risk |
| Large Data | rows > 50,000 | LLMs impractical |

### 2. Model Scoring

Each model is scored 0-4 on capability dimensions (calibrated from spark benchmarks v7.0.0):

| Model | Type | GPU | Skew | Card | Zipfian | Small | Corr | DP | Quality |
|-------|------|-----|------|------|---------|-------|------|----|---------|
| CART | Tree | no | 3 | 4 | 2 | 4 | 4 | 0 | 0.981 |
| SMOTE | Statistical | no | 3 | 4 | 2 | 4 | 4 | 0 | 0.979 |
| BayesianNetwork | Statistical | no | 3 | 4 | 2 | 4 | 3 | 0 | 0.971 |
| ARF | Tree | no | 2 | 4 | 3 | 4 | 4 | 0 | 0.962 |
| NFlow | Flow | no | 2 | 4 | 2 | 4 | 1 | 0 | 0.915 |
| TVAE | VAE | yes | 2 | 4 | 1 | 3 | 4 | 0 | 0.865 |
| TabSyn | Diffusion | yes | 2 | 4 | 3 | 3 | 2 | 0 | 0.848 |
| CTGAN | GAN | no | 1 | 4 | 2 | 2 | 3 | 0 | 0.809 |
| DPCART | Tree+DP | no | 2 | 0 | 2 | 2 | 3 | 3 | 0.759 |
| TabDDPM | Diffusion | yes | 1 | 2 | 2 | 2 | 3 | 0 | 0.697 |
| AutoDiff | Diffusion | yes | 1 | 3 | 2 | 2 | 1 | 0 | 0.634 |
| AIM | Stat+DP | no | 3 | 0 | 1 | 2 | 3 | 4 | 0.540 |
| PATECTGAN | GAN+DP | yes | 0 | 4 | 2 | 1 | 0 | 4 | 0.455 |
| GReaT | LLM | yes | 3 | 4 | 4 | 3 | 3 | 0 | N/A |
| Identity | Baseline | no | 4 | 4 | 4 | 4 | 4 | 0 | 0.989* |

*Identity is a passthrough baseline for testing. GReaT scores are literature-derived.

### 3. Constraint Filtering

Hard constraints eliminate impossible options:

- **`cpu_only`** — Removes GPU-dependent models (TabDDPM, TabSyn, AutoDiff, TVAE, PATECTGAN, GReaT)
- **`strict_dp`** — Keeps only models with DP score >= threshold (AIM, PATECTGAN, DPCART)

### 4. Quality Bonus & Tie-Breaking

An empirical quality bonus from spark benchmarks is added to capability scores. When top models score within 5%, deterministic tie-breaking applies:

- **Small data** (<1000 rows) → ARF > CART > BayesianNetwork > SMOTE
- **GPU available** (cpu_only=false) → GReaT > TabDDPM > TabSyn > AutoDiff > TVAE
- **CPU only** → CART > SMOTE > BayesianNetwork > ARF > NFlow
- **Speed preference** → CART > ARF > SMOTE > TVAE > DPCART

All thresholds and priority lists are configurable via `config/model_capabilities.json`.

## Output Schema

```json
{
  "dataset_id": "my_dataset",
  "row_count": 10000,
  "column_count": 15,
  "stress_factors": {
    "severe_skew": true,
    "high_cardinality": false,
    "zipfian_distribution": true,
    "small_data": false,
    "large_data": false
  },
  "skewness": {
    "max_skewness": 4.2,
    "severe_columns": ["income", "transaction_amount"]
  },
  "cardinality": {
    "max_cardinality": 250,
    "high_cardinality_columns": []
  },
  "recommendation": {
    "model": "TabSyn",
    "confidence": 0.87,
    "reasoning": [
      "Severe skew detected (4.2) - requires diffusion-based model",
      "Zipfian distribution in 'category' column - TabSyn handles well",
      "10k rows is within optimal range for TabSyn"
    ]
  }
}
```

## Benchmark Datasets

Synthony includes three synthetic control datasets for validation:

| Dataset | Purpose | Key Metric |
|---------|---------|------------|
| **The Long Tail** | Test skew handling | Skewness ≈ 4.5 |
| **The Needle in Haystack** | Test Zipfian/mode collapse | Top 20% = 90% |
| **The Small Data Trap** | Test overfitting prevention | 200 rows |

```python
from synthony.benchmark import BenchmarkDatasetGenerator

# Generate all benchmark datasets
BenchmarkDatasetGenerator.save_benchmarks("./benchmarks")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Synthony                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Profiler   │───▶│   Matcher    │───▶│  Recommender │       │
│  │              │    │              │    │              │       │
│  │ - Skewness   │    │ - Scoring    │    │ - Rule-based │       │
│  │ - Cardinality│    │ - Filtering  │    │ - LLM-based  │       │
│  │ - Zipfian    │    │ - Tie-break  │    │ - Hybrid     │       │
│  │ - Correlation│    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Model Registry (15 SOTA models with capability scores v7.0.0)  │
└─────────────────────────────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=synthony --cov-report=term-missing

# Run specific test
pytest tests/unit/test_skewness_detector.py -v
```

### Code Quality

```bash
# Format
black src/ tests/

# Lint
ruff src/ tests/

# Type check
mypy src/
```

## Roadmap

- [x] Core stress detection algorithms
- [x] Pydantic schemas and JSON serialization
- [x] CSV/Parquet file loading
- [x] Benchmark dataset generators
- [x] CLI tools
- [x] FastAPI REST server
- [x] Rule-based recommendation engine
- [x] LLM-based recommendation engine
- [x] React + TypeScript frontend
- [x] Docker deployment
- [x] MCP server implementation for AI agent integration
- [ ] PyPI package publication

## Contributing

Contributions welcome! Please see `CLAUDE.md` for architecture details and coding conventions.

## License

MIT License with prior authorization requirement — see [LICENSE.md](LICENSE.md) for details.

**Note**: Prior written authorization from the author or UCLA Trustworthy AI Lab is required before public distribution or commercial use.

## Citation

```bibtex
@software{synthony,
  title = {Synthony: Intelligent Synthetic Data Model Recommendation},
  author = {Son, Hochan, Xiaofeng Lin, Jason Ni, and Guang Cheng},
  organization = {UCLA Trustworthy AI Lab},
  url = {https://github.com/ohsono/Synthony},
  year = {2025}
}
```

## Acknowledgments

Developed by **Hochan Son** at **UCLA Trustworthy AI Lab** under the supervision of Prof. Guang Cheng.
Xiaofeng for the table-synthesizer implementation and model architecture design. Jason Ni for testing and evaluation.
