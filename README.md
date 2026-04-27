# Synthony

**Orchestrating the right synthetic data model for your tabular data.**

[![Paper](https://img.shields.io/badge/ICLR%202026-DeLTa%20Workshop-blueviolet)](https://openreview.net/forum?id=cj4SNumWqf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT-NC](https://img.shields.io/badge/License-MIT--NC-blue.svg)](LICENSE.md)

Synthony is an intelligent orchestration platform that analyzes your tabular dataset's statistical characteristics ("stress factors") and recommends the optimal synthetic data generation model from 15 state-of-the-art options. Like a symphony conductor, Synthony ensures the right model plays the right role for your data.

> **Published at the 2nd DeLTa Workshop, ICLR 2026**
> [Paper](https://openreview.net/forum?id=cj4SNumWqf) | [Code](https://github.com/UCLA-Trustworthy-AI-Lab/Synthony)

---

## Why Synthony?

Choosing the right synthetic data model is non-trivial. Each model family has distinct failure modes:

| Model Family | Common Failure Mode |
|---|---|
| GANs | Collapse on skewed distributions |
| VAEs | Mode collapse on high-cardinality columns |
| LLMs (GReaT) | Too slow and memory-intensive for large datasets |
| Tree-based | Miss complex cross-column correlations |
| Diffusion | High GPU cost, slow to train on small data |

Synthony solves this by:

1. **Profiling** your data to detect statistical stress factors
2. **Scoring** those factors against a calibrated model capability registry
3. **Recommending** the best model with hard constraints and clear reasoning

---

## Architecture Overview

```mermaid
graph TD
    Input["Input Data\n(CSV / Parquet)"] --> Analyzer

    subgraph Core["Synthony Core (src/synthony/)"]
        Analyzer["StochasticDataAnalyzer\nanalyzer.py"] --> SD["SkewnessDetector"]
        Analyzer --> CD["CardinalityDetector\n+ Zipfian"]
        Analyzer --> CR["CorrelationDetector"]
        Analyzer --> DS["DataSizeClassifier"]
        SD & CD & CR & DS --> Profile["DatasetProfile\n(Pydantic schema)"]
    end

    Profile --> Engine

    subgraph Recommender["Recommendation Engine (recommender/engine.py)"]
        Engine["ModelRecommendationEngine"] --> Filter["Hard Filters\n(cpu_only, strict_dp, row limits)"]
        Filter --> HardPath{"Hard Problem?\n(skew+card+zipfian)"}
        HardPath -- "yes" --> HP["Hard Problem Path\nGReaT → TabDDPM → ARF"]
        HardPath -- "no" --> Score["Capability Scoring\n+ Empirical Quality Bonus"]
        Score --> Tie["Tie-Breaking\n(GPU/CPU/speed/small_data)"]
        HP & Tie --> LLM{"LLM Scoring?\n(method=hybrid/llm)"}
        LLM -- "yes" --> OpenAI["OpenAI GPT\n+ SystemPrompt v6.0"]
        LLM -- "no" --> Result
        OpenAI --> Result["RecommendationResult\n(ranked models + reasoning)"]
    end

    Registry["model_capabilities.json\nv7.0.2 — 15 models × 6 dims"] -.->|loads| Engine

    subgraph Interfaces["Interfaces"]
        Result --> API["FastAPI REST\n:8000/docs"]
        Result --> MCP["MCP Server\nsynthony_mcp"]
        Result --> CLI["CLI\nsynthony-recommender"]
    end
```

---

## Data Flow: End-to-End

```mermaid
sequenceDiagram
    participant U as User / Agent
    participant MCP as MCP Server
    participant Analyzer as StochasticDataAnalyzer
    participant Engine as RecommendationEngine
    participant LLM as OpenAI (optional)
    participant DB as SQLite DB

    U->>MCP: synthony_analyze_stress_profile(dataset_name)
    MCP->>Analyzer: analyze(data.csv)
    Analyzer-->>MCP: DatasetProfile {stress_factors, skewness, cardinality, ...}
    MCP-->>U: profile JSON

    U->>MCP: synthony_rank_models_hybrid(dataset_profile)
    MCP->>Engine: recommend(profile, method="hybrid")
    Engine->>Engine: Hard filter → Score → Tie-break
    Engine->>LLM: chat.completions (SystemPrompt v6.0)
    LLM-->>Engine: ranked JSON
    Engine-->>MCP: RecommendationResult
    MCP-->>U: ranked models + confidence + reasoning

    U->>MCP: synthony_explain_recommendation_reasoning(result, profile)
    MCP-->>U: natural language explanation

    Note over MCP,DB: synthony_update_model_capabilities writes<br/>both src/ and config/ JSON copies atomically
```

---

## Features

### Stress Detection

| Factor | Threshold | Impact |
|---|---|---|
| Severe Skew | \|skewness\| > 2.0 | Breaks GANs / VAEs |
| High Cardinality | unique > 500 | Mode collapse risk |
| Zipfian Distribution | Top 20% > 80% of mass | Requires specialized tokenization |
| Small Data | rows < 1,000 | Overfitting risk |
| Large Data | rows > 50,000 | LLMs impractical |

### Model Registry (v7.0.2 — 15 Models)

Capability scores 0–4 calibrated from spark benchmarks (10 datasets):

| Model | Type | GPU | Skew | Card | Zipfian | Small | Corr | DP | Quality |
|---|---|---|---|---|---|---|---|---|---|
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
| GReaT | LLM | yes | 4 | 4 | 4 | 2 | 3 | 0 | N/A† |
| Identity | Baseline | no | 4 | 4 | 4 | 4 | 4 | 0 | 0.989* |

*Identity is a passthrough baseline for testing only.
†GReaT scores are literature-derived, not empirically validated.

---

## Installation

```bash
pip install -e .                   # Core only
pip install -e ".[cli]"            # CLI tools
pip install -e ".[api]"            # FastAPI REST server
pip install -e ".[llm]"            # LLM recommendations (requires OPENAI_API_KEY)
pip install -e ".[mcp]"            # MCP server
pip install -e ".[all]"            # Everything
pip install -e ".[dev]"            # Development tools
```

---

## Quick Start

### Python API

```python
from synthony import StochasticDataAnalyzer
from synthony.recommender.engine import ModelRecommendationEngine

analyzer = StochasticDataAnalyzer()
profile = analyzer.analyze("data.csv")

print(f"Severe skew: {profile.stress_factors.severe_skew}")
print(f"High cardinality: {profile.stress_factors.high_cardinality}")

engine = ModelRecommendationEngine()
result = engine.recommend(profile, method="rule_based")
print(f"Recommended: {result.recommended_model.model_name}")
print(f"Confidence: {result.confidence:.2f}")

# With intent conditioning
result = engine.recommend(profile, method="rule_based", focus="privacy")

# With hard constraints
result = engine.recommend(profile, method="hybrid",
                          constraints={"cpu_only": True, "strict_dp": True})
```

### CLI

```bash
# Profile a dataset
synthony-profile data.csv --verbose
synthony-profile data.csv -o profile.json

# Get a model recommendation
synthony-recommender -i data.csv --method hybrid
synthony-recommender -i data.csv --method rulebased --cpu-only
synthony-recommender -i data.csv --method llm --strict-dp

# Benchmark synthetic vs original
synthony-benchmark -r original.csv -s synthetic.csv --verbose
synthony-benchmark -r original.csv -s synthetic.csv -o results.json
```

### REST API

```bash
# Start the server
synthony-api
# or: uvicorn synthony.api.server:app --reload

# Visit http://localhost:8000/docs for interactive API docs

# Analyze and recommend in one call
curl -X POST "http://localhost:8000/analyze-and-recommend" \
  -F "file=@data.csv" \
  -F "method=hybrid"
```

### MCP Server (AI Agent Integration)

```bash
# Install with MCP support
pip install -e ".[mcp]"

# Start the server
synthony-mcp
# or: python -m mcp_server.server --verbose

# Test the protocol
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' \
  | python -m mcp_server.server
```

**Claude Desktop integration (`~/Library/Application Support/Claude/claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "synthony": {
      "command": "synthony-mcp",
      "env": {
        "SYNTHONY_DATA_DIR": "/path/to/your/datasets"
      }
    }
  }
}
```

---

## Recommendation Engine: Decision Logic

```mermaid
flowchart TD
    A[DatasetProfile] --> B{Hard Filters}
    B -->|cpu_only=true| B1[Remove GPU models\nTabSyn/TVAE/TabDDPM/AutoDiff/PATECTGAN]
    B -->|strict_dp=true| B2[Keep DP≥3 only\nAIM, PATECTGAN, DPCART]
    B -->|exclude list| B3[Remove excluded models]
    B1 & B2 & B3 --> C{Hard Problem?\nskew>2.0 AND card>500\nAND zipfian>0.05}

    C -->|yes + large data| D1["TabDDPM\n(conf=0.85)"]
    C -->|yes + normal| D2["GReaT\n(conf=0.95)"]
    C -->|yes + fallback| D3["ARF→TabSyn→CART\n(conf=0.70)"]

    C -->|no| E[Score each model\ncapability × scale_factor]
    E --> F[+ Empirical quality bonus\n× 0.3 weight]
    F --> G{Top scores\nwithin 5%?}
    G -->|yes| H{Tie-break rules}
    H -->|small_data| H1[ARF > CART > BayesNet > SMOTE]
    H -->|gpu_available| H2[GReaT > TabDDPM > TabSyn > AutoDiff > TVAE]
    H -->|cpu_only| H3[CART > SMOTE > BayesNet > ARF > NFlow]
    H -->|prefer_speed| H4[ARF > CART > DPCART > SMOTE > TVAE]
    G -->|no| I[Winner by score]
    H1 & H2 & H3 & H4 & I --> J{method=hybrid\nor llm?}
    J -->|yes| K[OpenAI GPT\nSystemPrompt v6.0]
    K --> L[Merge rule + LLM scores]
    J -->|no| L
    D1 & D2 & D3 & L --> M[RecommendationResult\nranked models + confidence + reasoning]
```

---

## MCP Server Tool Map

```mermaid
graph LR
    subgraph Data["Data Tools"]
        LD[synthony_list_datasets]
        LOD[synthony_load_dataset]
    end

    subgraph Profiling["Profiling Tools"]
        ASP[synthony_analyze_stress_profile]
        GBD[synthony_generate_benchmark_dataset]
    end

    subgraph Models["Model Tools"]
        LM[synthony_list_models]
        GMI[synthony_get_model_info]
        CMC[synthony_check_model_constraints]
        UMC[synthony_update_model_capabilities]
        USP[synthony_update_system_prompt]
    end

    subgraph Recommendation["Recommendation Tools"]
        RMH[synthony_rank_models_hybrid]
        RMR[synthony_rank_models_rule]
        RML[synthony_rank_models_llm]
        GTB[synthony_get_tie_breaker_logic]
        ERR[synthony_explain_recommendation_reasoning]
    end

    subgraph Benchmark["Benchmark Tools"]
        BC[synthony_benchmark_compare]
    end

    LD --> LOD --> ASP --> RMH
    ASP --> RMR
    ASP --> RML
    RMH --> ERR
    RMR --> ERR
    RML --> ERR
    RMH --> GTB
    LM --> GMI --> CMC
    UMC -->|bumps patch version| USP
    BC -.->|validate| RMH
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `SYNTHONY_DATA_DIR` | Dataset directory (default: `dataset/input_data`) |
| `OPENAI_API_KEY` | Required for LLM-based recommendations |
| `MCP_DEBUG` | Set to any non-empty value to enable verbose MCP server logging |

---

## Project Structure

```
Synthony/
├── src/synthony/
│   ├── core/           # DataLoader, StochasticDataAnalyzer, schemas, errors
│   ├── detectors/      # SkewnessDetector, CardinalityDetector, CorrelationDetector, DataSizeClassifier
│   ├── recommender/    # ModelRecommendationEngine, focus_profiles, model_capabilities.json
│   ├── benchmark/      # DataQualityBenchmark, metrics
│   ├── api/            # FastAPI server, database, storage, security
│   ├── cli.py          # CLI entry points
│   └── utils/          # AnalyzerConfig, constants
├── mcp_server/
│   ├── server.py       # MCP server entry point (SynthonyMCPServer)
│   └── tools/          # data_tools, profiling_tools, model_tools, recommendation_tools, benchmark_tools
├── config/
│   ├── model_capabilities.json  # Canonical registry v7.0.2 (15 models)
│   └── SystemPrompt.md          # LLM system prompt v6.0
├── tests/
│   ├── unit/           # Detector and engine unit tests
│   ├── integration/    # End-to-end pipeline tests
│   ├── functional/     # CLI, API, and MCP functional tests
│   ├── evaluation/     # Recommendation accuracy evaluation
│   └── regression/     # Baseline and schema regression tests
├── docs/               # Architecture docs, scoring methodology, API guides
├── scripts/            # Bayesian optimization, benchmark runners, analysis scripts
└── ablation/           # Ablation study runner and results
```

---

## Scoring Methodology

Capability scores are derived empirically from preservation rates across 10 benchmark datasets:

| Score | Threshold | Meaning |
|---|---|---|
| 4 | preservation ≥ 0.90 | Excellent |
| 3 | preservation ≥ 0.75 | Good |
| 2 | preservation ≥ 0.50 | Moderate |
| 1 | preservation ≥ 0.25 | Poor |
| 0 | preservation < 0.25 | Fails |

Cardinality uses a density-normalized formula: `(synth_unique / synth_rows) / (orig_unique / orig_rows)` to correct for row-count sampling bias. Full methodology: [`docs/scoring_methodology.md`](docs/scoring_methodology.md).

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Code quality
black src/ tests/
ruff check src/ tests/
mypy src/

# Run all tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Exclude LLM-dependent tests (no API key required)
pytest -m "not requires_llm"

# Coverage report
pytest --cov=synthony --cov-report=html
```

---

## Roadmap

- [ ] PyPI package publication
- [ ] Learned capability embeddings (replacing hand-crafted registry)
- [ ] Expanded benchmark suite (20+ datasets)
- [ ] Web frontend for dataset upload and visualization

---

## Contributing

Contributions are welcome. See [`CLAUDE.md`](CLAUDE.md) for architecture details and [`AGENTS.md`](AGENTS.md) for coding agent guidelines.

---

## License

MIT-NC (MIT Non-Commercial) — free for academic and research use with attribution. Commercial use requires prior written authorization. See [LICENSE.md](LICENSE.md).

For commercial licensing: ohsono@gmail.com / hochanson@g.ucla.edu

---

## Citation

```bibtex
@inproceedings{son2026synthony,
  title     = {{SYNTHONY}: A Stress-Aware, Intent-Conditioned Agent for Deep
               Tabular Generative Model Selection},
  author    = {Hochan Son and Xiaofeng Lin and Jason Ni and Guang Cheng},
  booktitle = {ICLR 2026 2nd Workshop on Deep Generative Model in Machine
               Learning: Theory, Principle and Efficacy (DeLTa)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=cj4SNumWqf}
}
```

## Team

All authors are members of the **UCLA Trustworthy AI Lab**.

- **Hochan Son** — ohsono@gmail.com / hochanson@g.ucla.edu
- **Xiaofeng Lin** — Bernardo1998@g.ucla.edu
- **Jason Ni** — jasonni19@g.ucla.edu
- **Guang Cheng** (PI) — guangcheng@g.ucla.edu
