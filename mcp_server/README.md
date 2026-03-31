# Synthony MCP Server

Model Context Protocol (MCP) server for Synthony - enabling AI agents to access data profiling and model recommendation capabilities.

## Overview

The Synthony MCP Server exposes the following capabilities through the Model Context Protocol:

- **Tools**: Executable functions for data profiling, model querying, and recommendations
- **Resources**: Read-only access to model registry, cached profiles, and benchmarks
- **Prompts**: Guided workflows for common tasks

## Architecture

The MCP server is organized into modular components:

```
mcp_server/
├── server.py                      # Main MCP server entry point
├── tools/
│   ├── profiling_tools.py        # Data analysis and stress profiling
│   ├── model_tools.py            # Model capabilities and constraints
│   └── recommendation_tools.py   # Hybrid rule-based + LLM engine
├── resources/
│   ├── model_registry.py         # Model catalog access
│   ├── profile_cache.py          # Cached dataset profiles
│   └── benchmark_data.py         # Historical benchmark results
├── prompts/
│   └── workflows.py              # Guided recommendation workflows
└── schemas/
    ├── tools_schema.json         # Tool definitions
    └── resources_schema.json     # Resource templates
```

## Installation

1. Install MCP SDK:

```bash
pip install "mcp>=0.9.0"
```

1. Install Synthony with all dependencies:

```bash
pip install -e ".[all]"
```

## Running the Server

### Standalone Mode (stdio transport)

```bash
python -m mcp_server.server
```

Or using the installed command:

```bash
synthony-mcp
```

### Client Integration

For detailed setup instructions for all supported clients (VSCode, Antigravity, Claude Desktop, Claude Code, Cline, Continue.dev, Cursor), see **[MCP_SETUP.md](../docs/MCP_SETUP.md)**.

Quick examples:

**VSCode** (`.vscode/mcp.json`):

```json
{
  "servers": {
    "synthony": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

**Antigravity / Claude Desktop / Claude Code**:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/Synthony",
      "env": {
        "PYTHONPATH": "/path/to/Synthony"
      }
    }
  }
}
```

## Available Tools

### Profiling Tools

#### `analyze_stress_profile`

Analyze a tabular dataset and extract stress profile.

**Arguments:**

- `data_path` (required): Path to CSV or Parquet file
- `dataset_id` (optional): Identifier for caching

**Returns:**

- Dataset profile with stress factors
- Column-level analysis
- Dataset metadata

**Example:**

```json
{
  "name": "analyze_stress_profile",
  "arguments": {
    "data_path": "./data/my_dataset.csv",
    "dataset_id": "my_dataset"
  }
}
```

#### `generate_benchmark_dataset`

Generate synthetic control datasets for validation.

**Arguments:**

- `dataset_type` (required): One of `long_tail`, `needle_haystack`, `small_data_trap`
- `output_path` (required): Path to save generated CSV
- `num_rows` (optional): Number of rows (overrides defaults)

**Returns:**

- Generated dataset metadata
- Expected stress factors

### Model Tools

#### `check_model_constraints`

Check which models satisfy given constraints.

**Arguments:**

- `cpu_only` (optional): CPU-only constraint
- `strict_dp` (optional): Differential privacy requirement
- `row_count` (optional): Dataset row count

**Returns:**

- Compatible models
- Excluded models with reasons
- Constraints applied

#### `get_model_info`

Get detailed information about a specific model.

**Arguments:**

- `model_name` (required): Model name (e.g., "GReaT", "TabDDPM", "ARF")

**Returns:**

- Full model specification
- Capabilities (0-4 scores)
- Constraints and limitations

#### `list_models`

List all available models with optional filtering.

**Arguments:**

- `model_type` (optional): Filter by type (GAN, VAE, Diffusion, etc.)
- `cpu_only` (optional): CPU-compatible only
- `requires_dp` (optional): Differential privacy support

**Returns:**

- Model catalog
- Model rankings

### Recommendation Tools

#### `rank_models_hybrid`

Rank models using hybrid rule-based + LLM approach.

**Arguments:**

- `dataset_profile` (required): Profile from `analyze_stress_profile`
- `column_analysis` (optional): Column-level analysis
- `constraints` (optional): Hard constraints (cpu_only, strict_dp)
- `method` (optional): `rule_based`, `llm`, or `hybrid` (default)
- `top_n` (optional): Number of alternatives (default: 3)

**Returns:**

- Primary recommendation
- Alternative models
- Reasoning and warnings

#### `get_tie_breaker_logic`

Resolve ties when top models score within 5%.

**Arguments:**

- `tied_models` (required): List of model names
- `dataset_profile` (required): Dataset profile
- `prefer_speed` (optional): Prioritize faster models

**Returns:**

- Winner model
- Reasoning
- Rule applied

#### `explain_recommendation_reasoning`

Generate user-friendly explanation for recommendation.

**Arguments:**

- `recommendation_result` (required): Result from `rank_models_hybrid`
- `dataset_profile` (required): Dataset profile
- `detail_level` (optional): `brief`, `detailed`, or `technical`

**Returns:**

- Natural language explanation
- Key factors
- Model strengths
- Alternatives comparison

## Available Resources

### Model Registry

#### `models://registry`

Complete model catalog with capability scores (0-4 scale).

#### `models://model/{model_name}`

Detailed information for a specific model.

**Example:** `models://model/GReaT`

### Dataset Profiles

#### `datasets://profiles/{dataset_id}`

Cached dataset profiles from previous analyses.

**Example:** `datasets://profiles/my_dataset`

### Benchmarks

#### `benchmarks://thresholds`

Stress detector thresholds (Skew>2.0, Zipfian>0.05, Cardinality>500).

#### `benchmarks://results/{model}/{dataset_type}`

Historical validation results (Wasserstein Distance, TVD).

**Example:** `benchmarks://results/GReaT/long_tail`

### Guidelines

#### `guidelines://system-prompt`

Active system prompt with model capability scores (from database).

## Available Prompts

### `/analyze-and-recommend`

Complete workflow from data upload to model recommendation.

**Arguments:**

- `data_path` (required): Path to dataset
- `constraints` (optional): JSON constraints

### `/explain-hard-problem`

Deep dive into complex dataset characteristics.

**Arguments:**

- `dataset_id` (required): Dataset from previous analysis

### `/validate-recommendation`

Run offline benchmark validation.

**Arguments:**

- `dataset_id` (required): Dataset ID
- `model_name` (required): Model to validate

### `/update-knowledge-base`

Update SystemPrompt scores from empirical feedback.

**Arguments:**

- `benchmark_results` (required): Validation metrics JSON

## Testing

Run component tests:

```bash
python mcp_server/test_server.py
```

This validates:

- All components can be imported
- Tool definitions are properly structured
- Resource definitions are correct
- Prompt definitions are valid

## Usage Example

### Via Claude Code

1. Start a conversation with Claude Code
2. Claude Code automatically connects to the MCP server
3. Use natural language to request analysis:

```
Analyze my dataset at data/sales.csv and recommend the best synthesis model.
```

Claude Code will:

1. Call `analyze_stress_profile` tool
2. Call `rank_models_hybrid` tool
3. Call `explain_recommendation_reasoning` tool
4. Provide a complete recommendation with explanation

### Direct JSON-RPC

List available tools:

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | python -m mcp_server.server
```

Call a tool:

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"analyze_stress_profile","arguments":{"data_path":"data/test.csv"}},"id":1}' | python -m mcp_server.server
```

List resources:

```bash
echo '{"jsonrpc":"2.0","method":"resources/list","params":{},"id":1}' | python -m mcp_server.server
```

Read a resource:

```bash
echo '{"jsonrpc":"2.0","method":"resources/read","params":{"uri":"models://registry"},"id":1}' | python -m mcp_server.server
```

## Design Philosophy

### Why MCP vs FastAPI?

The MCP server replaces the original FastAPI design for several key advantages:

1. **Stateful Connections**: MCP maintains bidirectional communication vs stateless REST
2. **Native AI Agent Integration**: Direct integration with Claude Code and other AI agents
3. **Resource Subscriptions**: Clients can subscribe to updates (benchmark results, knowledge base changes)
4. **Guided Workflows**: Prompts codify best practices for using the recommendation engine
5. **No API Key Management**: stdio transport eliminates authentication overhead for local use
6. **Protocol Versioning**: MCP's capability negotiation ensures backwards compatibility

### Shadow Interface Pattern

The MCP server uses a "Shadow Interface" to access model capabilities without importing heavy ML libraries:

- `model_capabilities.json` maintains model metadata
- Allows fast recommendations without loading TensorFlow/PyTorch
- Decouples decision logic from model implementation
- Enables independent versioning and updates

### Self-Correcting Feedback Loop

The system is designed to improve over time:

1. **Analyze**: User uploads data → generate profile
2. **Recommend**: Match profile to model using scores
3. **Validate**: Run offline benchmarks (Wasserstein Distance, TVD)
4. **Refine**: Update SystemPrompt scores if empirical results differ from expectations

The `/update-knowledge-base` prompt facilitates this loop.

## Integration with Synthony Ecosystem

The MCP server integrates with the three-package architecture:

- **Package 1 (synthony)**: Uses `StochasticDataAnalyzer` and `ColumnAnalyzer` for profiling
- **Package 2 (table-synthesizers)**: Shadow interface via `model_capabilities.json`
- **MCP Server**: Orchestration brain that combines profiling + model selection

## Troubleshooting

### "No active system prompt found"

# Via API

curl -X POST <http://localhost:8000/systemprompt/upload> \
  -F "file=@docs/SystemPrompt_v3.md" \
  -F "version=v3.0" \
  -F "set_active=true"

### Import errors

Ensure all dependencies are installed:

```bash
pip install -e ".[all,mcp]"
```

### Database errors

The MCP server requires the Synthony database to be initialized. If using for the first time, ensure the database is set up:

```bash
# Start API server once to initialize database
python start_api.py
# Then stop it and start MCP server
python -m mcp_server.server
```

## Contributing

When adding new tools, resources, or prompts:

1. Add implementation to appropriate module in `tools/`, `resources/`, or `prompts/`
2. Update JSON schemas in `schemas/`
3. Register in `server.py` handlers
4. Add tests to `test_server.py`
5. Update this README

## License

MIT License - See LICENSE.md
