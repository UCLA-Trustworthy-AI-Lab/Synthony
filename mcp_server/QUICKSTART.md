# Synthony MCP Server - Quick Start Guide

## Installation

```bash
# Install MCP SDK
pip install "mcp>=0.9.0"

# Install Synthony with all dependencies
pip install -e ".[all]"
```

## Test the Server

```bash
# Run component tests
python mcp_server/test_server.py
```

Expected output:

```
✓ All tests passed!
```

## Running the Server

### Option 1: Direct Execution

```bash
# Start MCP server on stdio transport
python -m mcp_server.server
```

### Option 2: Using Entry Point

```bash
# After pip install
synthony-mcp
```

## Quick Examples

### Example 1: Analyze a Dataset

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "synthony_analyze_stress_profile",
    "arguments": {
      "data_path": "./data/my_dataset.csv",
      "dataset_id": "my_dataset"
    }
  },
  "id": 1
}
```

### Example 2: Get Model Recommendations

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "synthony_rank_models_hybrid",
    "arguments": {
      "dataset_profile": {
        // ... profile from analyze_stress_profile
      },
      "method": "hybrid",
      "top_n": 3
    }
  },
  "id": 2
}
```

### Example 3: Read Model Registry

```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "models://registry"
  },
  "id": 3
}
```

## Claude Code Integration

Add to `.claude/claude_code.json` or MCP settings:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/Synthony",
      "env": {
        "PYTHONPATH": "/path/to/Synthony/src"
      }
    }
  }
}
```

Then in Claude Code:

```
Analyze my dataset at data/sales.csv and recommend the best synthesis model.
```

## Common Workflows

### Workflow 1: Full Analysis + Recommendation

1. Upload dataset → `synthony_analyze_stress_profile`
2. Get recommendations → `synthony_rank_models_hybrid`
3. Get explanation → `synthony_explain_recommendation_reasoning`

### Workflow 2: Model Selection with Constraints

1. Check constraints → `synthony_check_model_constraints`
2. List compatible models → `synthony_list_models`
3. Get model details → `synthony_get_model_info`
4. Make recommendation → `synthony_rank_models_hybrid`

### Workflow 3: Validation

1. Generate benchmark → `synthony_generate_benchmark_dataset`
2. Run validation (external to Synthony)
3. Check results → `benchmarks://results/{model}/{type}`
4. Update knowledge → `/update-knowledge-base` prompt

## Tools Reference

| Tool | Purpose |
|------|---------|
| `synthony_list_datasets` | List datasets in data directory |
| `synthony_load_dataset` | Load dataset metadata + preview |
| `synthony_analyze_stress_profile` | Extract dataset stress factors |
| `synthony_generate_benchmark_dataset` | Create validation datasets |
| `synthony_check_model_constraints` | Filter models by row count |
| `synthony_get_model_info` | Get model details |
| `synthony_list_models` | List all models |
| `synthony_rank_models_hybrid` | Get recommendations (rule + LLM) |
| `synthony_rank_models_rule` | Get recommendations (rule-based only) |
| `synthony_rank_models_llm` | Get recommendations (LLM only) |
| `synthony_get_tie_breaker_logic` | Resolve close scores |
| `synthony_explain_recommendation_reasoning` | Generate explanations |
| `synthony_benchmark_compare` | Compare original vs synthetic quality |

## Resources Reference

| Resource | Purpose |
|----------|---------|
| `models://registry` | Complete model catalog |
| `models://model/{name}` | Individual model info |
| `datasets://profiles/{id}` | Cached profiles |
| `benchmarks://thresholds` | Detection thresholds |
| `benchmarks://results/{model}/{type}` | Validation results |
| `guidelines://system-prompt` | Active system prompt |

## Prompts Reference

| Prompt | Purpose |
|--------|---------|
| `/analyze-and-recommend` | Full workflow |
| `/explain-hard-problem` | Deep dive on complexity |
| `/validate-recommendation` | Run benchmarks |
| `/update-knowledge-base` | Refine scores |

## Troubleshooting

### Server won't start

```bash
# Check imports
python -c "import mcp_server.server"

# Check dependencies
pip list | grep mcp
pip list | grep synthony
```

### No active system prompt

```bash
# Check database
sqlite3 data/synthony.db "SELECT * FROM system_prompts WHERE is_active = 1;"

# Upload new prompt via API
curl -X POST http://localhost:8000/systemprompt/upload \
  -F "file=@docs/SystemPrompt_v3.md" \
  -F "version=v2.0" \
  -F "set_active=true"
```

### Database not initialized

```bash
# Start API once to initialize
python start_api.py
# Then stop and start MCP
python -m mcp_server.server
```

## Next Steps

1. **Read Full Documentation**: See `mcp_server/README.md`
2. **Explore Tools**: Run test examples
3. **Integrate with Claude Code**: Add to MCP settings
4. **Try Workflows**: Use guided prompts

## Support

- Documentation: `mcp_server/README.md`
- Architecture: `docs/architecture_v3.md`
- System Prompt: `docs/SystemPrompt_v3.md`
- Tests: `mcp_server/test_server.py`
