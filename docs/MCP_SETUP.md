# MCP Server Setup Guide

Connect the Synthony MCP server to your AI coding client. The server runs over **stdio transport** and exposes tools for dataset profiling, model recommendation, and benchmarking.

---

## Prerequisites

```bash
cd /path/to/Synthony
pip install -e ".[mcp]"
```

Verify the server starts:

```bash
python -m mcp_server.server --verbose
# You should see: "Synthony MCP Server initialized" on stderr
```

Quick protocol test:

```bash
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' \
  | python -m mcp_server.server
```

---

## 1. Claude Code (CLI)

Claude Code auto-discovers MCP servers from a `.mcp.json` file in the project root.

### Setup

The `.mcp.json` file is already included in this repository:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--verbose"],
      "env": {
        "PYTHONPATH": ".:src"
      }
    }
  }
}
```

That's it — just `cd` into the Synthony directory and start Claude Code. The server connects automatically.

### With LLM recommendations

To enable hybrid (rule + LLM) mode, add your API key to the env block:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--verbose"],
      "env": {
        "PYTHONPATH": ".:src",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### With self-hosted vLLM

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--verbose"],
      "env": {
        "PYTHONPATH": ".:src",
        "VLLM_URL": "http://localhost:8000/v1",
        "VLLM_API_KEY": "your-key",
        "VLLM_MODEL": "Qwen/Qwen2.5-32B-Instruct"
      }
    }
  }
}
```

### Verify

After starting Claude Code, the Synthony tools should appear in the tool list. Try asking:

> "List the available datasets" or "Analyze the stress profile of dataset/input_data/insurance.csv"

---

## 2. VSCode (Copilot Chat)

VSCode supports MCP servers natively via `.vscode/mcp.json`.

### Workspace Configuration (recommended)

Create `.vscode/mcp.json` in the Synthony project root:

```json
{
  "servers": {
    "synthony": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src",
        "SYNTHONY_DATA_DIR": "${workspaceFolder}/dataset/input_data"
      }
    }
  }
}
```

### With secure API key input

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "openai-api-key",
      "description": "OpenAI API Key (for LLM recommendations)",
      "password": true
    }
  ],
  "servers": {
    "synthony": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src",
        "OPENAI_API_KEY": "${input:openai-api-key}",
        "SYNTHONY_DATA_DIR": "${workspaceFolder}/dataset/input_data"
      }
    }
  }
}
```

### Global Configuration

To make Synthony available across all workspaces, open Command Palette (`Ctrl+Shift+P`) > **MCP: Open User Configuration** and add the `synthony` entry with an absolute `cwd` path.

### Notes

- Top-level key is `"servers"` (not `"mcpServers"`)
- `"type": "stdio"` is the default and can be omitted
- VSCode shows inline start/stop/restart buttons for MCP servers

---

## 3. Cursor

Cursor supports MCP servers via `.cursor/mcp.json` in the project root or through the settings UI.

### Config file

Create `.cursor/mcp.json` in the Synthony project root:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src"
      }
    }
  }
}
```

### Via Settings UI

1. Open **Settings > MCP Servers**
2. Click **Add new server**
3. Fill in:
   - **Name**: `synthony`
   - **Type**: `command` (stdio)
   - **Command**: `python -m mcp_server.server`
   - **Working Directory**: `/path/to/Synthony`

### Notes

- Cursor uses `"mcpServers"` as the top-level key
- MCP tools appear in Cursor's Composer (Agent mode)

---

## 4. Antigravity

Add the following to your Antigravity MCP configuration:

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/Synthony",
      "env": {
        "PYTHONPATH": "/path/to/Synthony:/path/to/Synthony/src"
      }
    }
  }
}
```

### With environment variables

```json
{
  "mcpServers": {
    "synthony": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--verbose"],
      "cwd": "/path/to/Synthony",
      "env": {
        "PYTHONPATH": "/path/to/Synthony:/path/to/Synthony/src",
        "OPENAI_API_KEY": "sk-...",
        "SYNTHONY_DATA_DIR": "/path/to/Synthony/dataset/input_data"
      }
    }
  }
}
```

Replace `/path/to/Synthony` with your actual installation path.

---

## 5. OpenClaw

OpenClaw configures MCP servers in `~/.openclaw/openclaw.json` under the `mcp.servers` key.

### Via config file

Edit `~/.openclaw/openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "synthony": {
        "command": "python",
        "args": ["-m", "mcp_server.server", "--verbose"],
        "cwd": "/path/to/Synthony",
        "env": {
          "PYTHONPATH": "/path/to/Synthony:/path/to/Synthony/src",
          "SYNTHONY_DATA_DIR": "/path/to/Synthony/dataset/input_data"
        }
      }
    }
  }
}
```

### Via CLI

```bash
openclaw mcp set synthony '{
  "command": "python",
  "args": ["-m", "mcp_server.server", "--verbose"],
  "cwd": "/path/to/Synthony",
  "env": {
    "PYTHONPATH": "/path/to/Synthony:/path/to/Synthony/src"
  }
}'
```

### Manage servers

```bash
openclaw mcp list          # List all configured servers
openclaw mcp show synthony # View synthony config
openclaw mcp unset synthony # Remove server
```

### Notes

- Top-level key is `mcp.servers` (not `mcpServers`)
- A server entry must use either `command` (stdio) or `url` (HTTP/SSE) — not both
- Env values support `${ENV_VAR}` interpolation

---

## Available Tools

| Tool | Description |
|------|-------------|
| `list_datasets` | Discover datasets in configured data directory |
| `load_dataset` | Load and preview dataset metadata |
| `analyze_stress_profile` | Profile dataset stress factors |
| `generate_benchmark_dataset` | Generate synthetic control datasets |
| `check_model_constraints` | Check model compatibility with dataset size |
| `get_model_info` | Get detailed model specification |
| `list_models` | List all models with optional filtering |
| `rank_models_hybrid` | Hybrid rule + LLM recommendations |
| `rank_models_rule` | Pure rule-based recommendations |
| `rank_models_llm` | Pure LLM-based recommendations |
| `get_tie_breaker_logic` | Resolve tied model scores |
| `explain_recommendation_reasoning` | Natural language explanation |
| `benchmark_compare` | Compare original vs synthetic data quality |

## Available Resources

| Resource | URI |
|----------|-----|
| Model Registry | `models://registry` |
| Individual Model | `models://model/{name}` |
| Dataset Profile Cache | `datasets://profiles/{id}` |
| Benchmark Thresholds | `benchmarks://thresholds` |
| Benchmark Results | `benchmarks://results/{model}/{dataset_type}` |
| System Prompt | `guidelines://system-prompt` |

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `PYTHONPATH` | Ensures `synthony` and `mcp_server` are importable | Yes (if running outside project dir) |
| `SYNTHONY_DATA_DIR` | Dataset directory (default: `dataset/input_data`) | No |
| `OPENAI_API_KEY` | LLM-based recommendations | For `hybrid`/`llm` methods |
| `OPENAI_MODEL` | OpenAI model name (default: `gpt-4o-mini`) | No |
| `VLLM_URL` | Self-hosted vLLM endpoint (replaces OpenAI) | No |
| `VLLM_API_KEY` | API key for vLLM endpoint | No |
| `VLLM_MODEL` | Model name for vLLM | No |
| `MCP_DEBUG` | Enable verbose MCP logging | No |

---

## Troubleshooting

### Server won't start

```bash
# Check which Python is active
which python
python --version

# Verify mcp SDK
pip show mcp

# Verify synthony is installed
python -c "import synthony; print('OK')"

# Test the server directly
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' \
  | python -m mcp_server.server
```

### ModuleNotFoundError: No module named 'synthony'

This means the active Python doesn't have synthony installed. Either:

1. **Install it**: `pip install -e ".[mcp]"`
2. **Set PYTHONPATH** in your MCP config to include the project root and `src/` directory
3. **Use absolute Python path** if conda/virtualenv isn't activating correctly:
   ```json
   {
     "command": "/path/to/miniconda/envs/synthony/bin/python",
     "args": ["-m", "mcp_server.server"]
   }
   ```

### Wrong Python / wrong uvicorn

If `python` resolves to a system Python instead of your conda/venv, use the full path to the correct Python binary in your MCP config. Check with `which python`.

### Data directory not found

Create the default directory or set `SYNTHONY_DATA_DIR`:

```bash
mkdir -p dataset/input_data
# Place CSV/Parquet files here
```

### "No active system prompt found"

The guidelines resource requires the database to have an active system prompt. Start the API server once to initialize:

```bash
./start_api.sh
# Then Ctrl+C and restart the MCP server
```
