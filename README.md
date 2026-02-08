# Synthony

**Orchestrating the right synthetic data model for your tabular data.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)]()

Synthony is an intelligent recommendation platform that analyzes your tabular dataset's characteristics and recommends the optimal synthetic data generation model from 13+ state-of-the-art options. Like a symphony conductor orchestrating instruments, Synthony orchestrates the right model for your data.

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
- **Data Size Classification** — Small (<500 rows) vs Large (>50k rows) constraints
- **Higher-Order Correlation Detection** — Dense but non-linear relationships

### Model Recommendation

- **13+ SOTA Models** — TabDDPM, TabSyn, AutoDiff, GReaT, TVAE, CTGAN, PATE-CTGAN, DPCART, AIM, GaussianCopula, ARF
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
synthony-profile data.csv

# Save profile to JSON
synthony-profile data.csv --output profile.json

# Generate benchmark datasets
synthony-benchmark --output-dir ./benchmarks
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
| Small Data | rows < 500 | Overfitting risk |
| Large Data | rows > 50,000 | LLMs impractical |

### 2. Model Scoring

Each model is scored 0-4 on capability dimensions:

| Model | Skew | Cardinality | Zipfian | Small Data | Privacy |
|-------|------|-------------|---------|------------|---------|
| GReaT | 4 | 4 | 4 | 2 | 0 |
| TabDDPM | 3 | 2 | 2 | 2 | 0 |
| TabSyn | 3 | 3 | 3 | 2 | 0 |
| ARF | 2 | 3 | 3 | 4 | 0 |
| TVAE | 1 | 2 | 1 | 3 | 0 |
| PATE-CTGAN | 1 | 2 | 1 | 2 | 4 |

### 3. Constraint Filtering

Hard constraints eliminate impossible options:

- **`cpu_only`** — Removes GPU-dependent models (TabDDPM, TabSyn, GReaT)
- **`strict_dp`** — Keeps only differential privacy models (PATE-CTGAN, DPCART, AIM)

### 4. Tie-Breaking

When top models score within 5%:

- **Small data (<500 rows)** → Prefer ARF
- **Large data with hard problems** → Prefer TabDDPM (GReaT too slow)
- **Otherwise** → Prefer faster models (TVAE, ARF)

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
│  Model Registry (13+ SOTA models with capability scores)        │
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
