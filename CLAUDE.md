# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Synthony** is an intelligent orchestration platform that automatically recommends the optimal synthetic data generation model from 13+ State-of-the-Art (SOTA) models based on dataset characteristics. The system analyzes data "stress factors" (Skewness, Cardinality, Zipfian distributions) and matches them to model capabilities using a hybrid rule-based + LLM decision engine.

**Architecture Transition**: The project is transitioning from a Package 3 FastAPI design to an **MCP (Model Context Protocol) server** architecture to enable direct integration with AI agents like Claude Code.

## System Architecture

The codebase is designed as a modular ecosystem with strict separation of concerns:

### Package 1: `synthony` (Data Infrastructure)
- **Purpose**: Standalone library for data ingestion and statistical profiling
- **Core Class**: `StochasticDataAnalyzer`
- **Responsibility**: Convert raw CSV/Parquet into a "Stress Profile" containing:
  - Skewness metrics (Fisher-Pearson coefficient)
  - Cardinality analysis (unique value counts)
  - Zipfian distribution detection (top 20% category concentration)
  - Correlation complexity (dense correlation matrix detection)
- **Output**: Strictly typed JSON `DatasetProfile` (metadata only, not raw data)
- **Tech Stack**: Pandas, Scipy, Polars (for performance)

### Package 2: `table-synthesizers` (External Repository)
- **Purpose**: Heavy-lifting training engine hosting 13+ SOTA models
- **Models**: TabDDPM, TabSyn, AutoDiff, GReaT, TabTree, TVAE, CTGAN, PATE-CTGAN, DPCART, AIM, GaussianCopula, ARF
- **Integration**: Synthony maintains a **Shadow Interface** (`model_capabilities.json`) to understand model capabilities without importing heavy dependencies
- **Note**: This is an external repository; Synthony does not directly import these models

### MCP Server (Orchestration Brain)
**Replaces the original Package 3 FastAPI design**

- **Purpose**: MCP server that exposes data profiling and model recommendation capabilities to AI agents
- **Protocol**: JSON-RPC 2.0 over stdio transport (optimized for local AI agent communication)
- **Core Advantage**: Bidirectional, stateful communication vs REST request/response pattern

#### MCP Server Structure

```
mcp-server-synthony/
├── server.py                      # Main MCP server entry point
├── tools/
│   ├── profiling_tools.py        # Tools for Package 1 integration
│   ├── model_tools.py            # Tools for Package 2 shadow interface
│   └── recommendation_tools.py   # Hybrid rule-based + LLM engine
├── resources/
│   ├── model_registry.py         # Expose model capabilities
│   ├── profile_cache.py          # Manage cached data profiles
│   └── benchmark_data.py         # Historical benchmark results
├── prompts/
│   └── workflows.py              # Guided recommendation workflows
└── schemas/
    ├── tools_schema.json         # Tool definitions
    └── resources_schema.json     # Resource templates
```

## MCP Server Capabilities

### Tools (Model-Controlled, Actively Executable)

| Tool | Purpose | Package Mapping |
|------|---------|-----------------|
| `analyze_stress_profile` | Extract skewness, cardinality, zipfian ratio from tabular data | Package 1: StochasticDataAnalyzer |
| `check_model_constraints` | Validate constraints (cpu_only, strict_dp, data size limits) | Package 2: Shadow Interface |
| `rank_models_hybrid` | Score models 0-4 using rule-based + LLM decision logic | MCP Server: Recommendation Engine |
| `get_tie_breaker_logic` | Resolve conflicts when models score within 5% | MCP Server: Tie-Breaker Rules |
| `explain_recommendation_reasoning` | Generate user-friendly explanation for model selection | MCP Server: LLM Narrative Engine |
| `generate_benchmark_dataset` | Create synthetic control datasets for validation | Package 1: BenchmarkGenerator |

### Resources (Application-Driven, Read-Only Context)

| Resource URI | Content |
|--------------|---------|
| `models://registry` | Full model catalog with capability scores (0-4 scale) |
| `datasets://profiles/{id}` | Cached analysis results for previously profiled data |
| `benchmarks://thresholds` | Stress detector thresholds (Skew>2.0, Zipfian>0.05, Cardinality>500) |
| `guidelines://system-prompt` | Current knowledge base (SystemPrompt.md) for scoring logic |
| `benchmarks://results/{model}/{dataset_type}` | Historical WD/TVD validation results |

### Prompts (User-Controlled Workflows)

| Prompt | Arguments | Purpose |
|--------|-----------|---------|
| `/analyze-and-recommend` | `data_path`, `constraints` | Full workflow from data upload to model recommendation |
| `/explain-hard-problem` | `dataset_id` | Deep dive into complex cases (severe skew, zipfian distributions) |
| `/validate-recommendation` | `dataset_id`, `model_name` | Run offline benchmark validation |
| `/update-knowledge-base` | `benchmark_results` | Refine SystemPrompt.md scores from empirical feedback |

## Core Decision Logic

The recommendation engine uses a **multi-stage funnel** approach:

### 1. Hard Filters (Eliminates Impossible Options)
- `cpu_only`: Removes GPU-dependent models (TabDDPM, TabSyn, GReaT)
- `strict_dp`: Keeps only differential privacy models (PATE-CTGAN, DPCART, AIM)
- `large_data` (>50k rows): Eliminates LLMs due to context window/latency constraints

### 2. Stress Detection (Identifies Data Difficulty)
Critical thresholds that define "Hard Problems":
- **Severe Skew**: Skewness > 2.0 (disqualifies basic GANs/VAEs, favors Diffusion/LLMs)
- **High Cardinality**: Unique count > 500 (triggers rare-category collapse checks)
- **Zipfian Distribution**: Top 20% categories > 80% of data (requires specialized tokenization)
- **Small Data**: Row count < 500 (forces ARF or GaussianCopula to prevent overfitting)
- **Large Data**: Row count > 50k (eliminates LLMs)

### 3. Model Scoring (0-4 Capability Scale)
Models are scored across dimensions:
- **Skew Handling**: GReaT (4), TabDDPM/TabSyn/AutoDiff/TabTree (3), ARF (2), Others (1)
- **High Cardinality**: GReaT/TabTree (4), CTGAN/TabSyn/ARF (3), Others (1-2)
- **Zipfian**: GReaT (4), TabSyn/TabTree/ARF (3), Others (1-2)
- **Small Data**: ARF/GaussianCopula (4), TVAE/DPCART (3), Others (1-2)
- **Privacy (DP)**: PATE-CTGAN/AIM (4), DPCART (3), Others (0)

### 4. Tie-Breaking Rules
When top models are within 5% score:
- Rows < 500: Prefer ARF (best for small data)
- Rows > 50k with "Hard Problem" (Skew>2 & Card>500 & Zipf>0.05): Prefer TabDDPM (GReaT too slow)
- Otherwise: Prefer faster models (TVAE/ARF) over slower ones (Diffusion/LLMs)

## Critical Design Patterns

### "Hard Problem" Detection
The system specifically identifies data characteristics that break traditional models:
1. **The Long Tail**: LogNormal skew > 2.0 (basic GANs fail to capture tail distribution)
2. **The Needle in Haystack**: Zipfian with 1000+ categories where top 10 = 90% volume (mode collapse risk)
3. **The Small Data Trap**: < 500 rows (overfitting/memorization risk)

### Shadow Interface Pattern
`model_capabilities.json` maintains model metadata without importing heavy ML libraries:
- Allows fast recommendation without loading TensorFlow/PyTorch
- Decouples decision logic from model implementation
- Enables independent versioning and updates

### Benchmark-Driven Feedback Loop
The system is designed to be self-correcting:
1. **Analyze**: User uploads data → generate profile
2. **Recommend**: Match profile to model using scores
3. **Validate**: Run offline benchmarks (Wasserstein Distance, TVD) on synthetic control datasets
4. **Refine**: Update `SystemPrompt.md` scores if empirical results differ from theoretical expectations

### MCP Notification Pattern
Use server-sent notifications when:
- Benchmark validation completes and changes recommendation confidence
- `SystemPrompt.md` knowledge base is updated from empirical feedback
- New models are added to the registry

## Key Algorithms

### Zipfian Detection
```
1. Sort category counts in descending order
2. Calculate cumulative sum of top 20%
3. Ratio = (top 20% sum) / (total count)
4. If ratio > 0.80, flag as Zipfian
```

### Higher-Order Correlation Detection
```
1. Compute full correlation matrix
2. Check if > 50% of pairs have |correlation| > 0.1 (dense matrix)
3. Compute linear R² for each feature pair
4. If matrix is dense but R² is low, flag as higher-order correlation
```

## Validation Strategy

### Synthetic Control Datasets
Three benchmark datasets validate model scores:

1. **Dataset A: "The Long Tail"**
   - Generation: `scipy.stats.lognorm(s=0.95, scale=exp(5))`
   - Metric: Wasserstein Distance < 0.1
   - Tests: Skew handling (Score 3-4 should pass, Score 1 should fail)

2. **Dataset B: "The Needle in Haystack"**
   - 10k rows, 1000 categories, top 10 = 90% volume
   - Metric: Rare category coverage ≥ 80%
   - Tests: Zipfian/mode collapse resistance

3. **Dataset C: "The Small Data Trap"**
   - 200 rows multivariate
   - Metric: R² > 0.6 on holdout set
   - Tests: Overfitting prevention

## Development Commands

Note: This repository is in early stages. The following structure is planned:

### Package 1 (synthony)
```bash
# Run data profiler tests
pytest tests/test_stochastic_analyzer.py

# Generate benchmark datasets
python -m src.benchmark.datasets

# Profile a CSV file
python -m src.profiler.analyze --input data.csv --output profile.json
```

### MCP Server
```bash
# Start MCP server (stdio transport for local AI agent)
python -m mcp_server.server

# Test MCP protocol
python -m mcp_server.tests.test_protocol

# Validate tool definitions
python -m mcp_server.validator.check_tools

# Run MCP server in debug mode
MCP_DEBUG=1 python -m mcp_server.server

# Test a specific tool via MCP JSON-RPC
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"analyze_stress_profile","arguments":{"data_path":"test.csv"}},"id":1}' | python -m mcp_server.server
```

### MCP Discovery Commands
```bash
# List all available tools
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | python -m mcp_server.server

# List all resources
echo '{"jsonrpc":"2.0","method":"resources/list","params":{},"id":1}' | python -m mcp_server.server

# List all prompts
echo '{"jsonrpc":"2.0","method":"prompts/list","params":{},"id":1}' | python -m mcp_server.server
```

## Important Files

- `docs/architecture_v2.md`: Complete system architecture and design rationale
- `docs/Implementation_plan.md`: Project roadmap and milestone tracking
- `docs/SystemPrompt_v2.md`: Knowledge base with 0-4 model capability scores (used by LLM engine)
- `docs/validation_plan_knowledge_base_v2.md`: Benchmark datasets and validation strategy
- `model_capabilities.json`: (Planned) Static registry for MCP server decision engine
- `mcp_server/server.py`: (Planned) Main MCP server entry point
- `mcp_server/schemas/`: (Planned) JSON Schema definitions for tools and resources

## Code Style Conventions

Based on the architecture documents:
- Use strict type annotations for all data profile outputs (JSON schemas)
- Maintain separation: Package 1 knows nothing about models; MCP Server knows nothing about heavy ML imports
- All thresholds (e.g., Skew > 2.0, Cardinality > 500) should be configurable constants
- Statistical calculations must use established libraries (scipy, numpy) rather than custom implementations
- Decision logic must be deterministic and traceable (log reasoning chain)

### MCP-Specific Conventions
- Tool descriptions must be detailed enough for AI agents to understand when to call them
- All tool inputs use JSON Schema validation
- Resources should be versioned and support efficient caching
- Use resource templates for parameterized access (e.g., `datasets://profiles/{id}`)
- Implement idempotent tools where possible
- Return structured, clearly-formatted results from all tools

## Testing Philosophy

- **Unit Tests**: Each stress detector (Skew, Zipfian, Cardinality) has isolated tests
- **Integration Tests**: Full pipeline from CSV → Profile → Recommendation
- **Benchmark Tests**: Model scores validated against synthetic control datasets
- **Regression Tests**: When `SystemPrompt.md` scores change, re-run all benchmarks to prevent degradation
- **MCP Protocol Tests**: Validate JSON-RPC 2.0 compliance, tool execution, resource access
- **Tool Schema Tests**: Ensure all tool inputs/outputs match declared JSON schemas

## MCP Integration Notes

### FastAPI to MCP Migration
The original Package 3 design was FastAPI-based. The transition to MCP provides:
- **Stateful connections** vs stateless REST endpoints
- **Bidirectional communication** for real-time benchmark updates
- **Native AI agent integration** without HTTP overhead
- **Capability negotiation** through MCP protocol handshake

### Why MCP for This Project
1. **Direct Claude Code Integration**: AI agents can call profiling and recommendation tools directly
2. **Resource Subscriptions**: Clients can subscribe to benchmark updates and knowledge base changes
3. **Guided Workflows**: Prompts codify best practices for using the recommendation engine
4. **No API Key Management**: Stdio transport eliminates authentication overhead for local use
5. **Protocol Versioning**: MCP's capability negotiation ensures backwards compatibility

### Using the MCP Server with Claude Code
When the MCP server is configured, Claude Code can:
- Call `analyze_stress_profile` to understand dataset characteristics
- Query `models://registry` to see all available synthesis models
- Execute `rank_models_hybrid` to get recommendations
- Use `/analyze-and-recommend` prompt for full guided workflow
- Subscribe to `benchmarks://results` to monitor validation updates
