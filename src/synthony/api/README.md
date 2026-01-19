# Synthony Data Analysis & Model Recommendation API

REST API for tabular data profiling and synthetic data model recommendation with intelligent LLM-powered insights.

## Quick Start

### 1. Install Dependencies

```bash
# Install with API support
pip install -e ".[api]"

# Or install everything (includes LLM support)
pip install -e ".[all]"
```

### 2. (Optional) Set OpenAI API Key for LLM Mode

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Start the Server

```bash
# Using script entry point
synthony-api

# Or using Python module
python -m synthony.api.server
```

Server will start at: **<http://localhost:8000>**

### 4. View Interactive Documentation

Open in your browser:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

---

## API Endpoints

### Core Analysis & Recommendation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/analyze` | POST | Analyze CSV/Parquet → Dataset Profile |
| `/recommend` | POST | Profile → Model Recommendations |
| `/analyze-and-recommend` | POST | CSV → Analysis + Recommendations (one-shot) |
| `/models` | GET | List available synthesis models |
| `/models/{name}` | GET | Get model details |

### System Prompt Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/systemprompt/upload` | POST | Upload system prompt version |
| `/systemprompt/list` | GET | List all prompt versions |
| `/systemprompt/active` | GET | Get active prompt |
| `/systemprompt/activate/{prompt_id}` | PUT | Activate by UUID |
| `/systemprompt/activate/version/{version}` | PUT | Activate by version string |

---

## Tutorial: Complete Workflow

### Step 1: Analyze Your Dataset

Upload your CSV or Parquet file to get a stress profile:

```bash
curl -X 'POST' 'http://localhost:8000/analyze' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./data/HTRU2.csv' \
  -F 'dataset_id=my_dataset'
```

**Response:**

```json
{
  "session_id": "d4842f8b-3c2e-490d-ac4e-e62ab0b9654d",
  "analysis_id": "e9544d9d-47d3-4151-88a9-f4c16909346c",
  "dataset_id": "my_dataset",
  "dataset_profile": {
    "row_count": 17898,
    "column_count": 9,
    "stress_factors": {
      "severe_skew": true,
      "high_cardinality": true,
      "zipfian_distribution": true
    }
  },
  "column_analysis": {
    "max_column_difficulty": 4,
    "difficult_columns": ["Var3", "Var4", "Var5"]
  },
  "message": "Analysis completed: 17898 rows × 9 columns"
}
```

**Save the `analysis_id`** - you'll use it in the next step!

### Step 2: Get Model Recommendations (Using Analysis ID)

Now use the `analysis_id` to get recommendations without re-uploading data:

```bash
curl -X 'POST' 'http://localhost:8000/recommend' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "my_dataset",
    "analysis_id": "e9544d9d-47d3-4151-88a9-f4c16909346c",
    "method": "hybrid",
    "constraints": {
      "cpu_only": false,
      "strict_dp": false
    },
    "top_n": 3
  }'
```

**Response:**

```json
{
  "recommended_model": {
    "model_name": "TabDDPM",
    "confidence_score": 0.92,
    "reasoning": "Best suited for high-cardinality data with severe skew"
  },
  "alternatives": [
    {"model_name": "CTGAN", "confidence_score": 0.85},
    {"model_name": "TVAE", "confidence_score": 0.78}
  ]
}
```

### Step 3: One-Shot Workflow (Shortcut)

Or do both steps at once:

```bash
curl -X 'POST' 'http://localhost:8000/analyze-and-recommend?method=hybrid&cpu_only=false&top_n=3' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./data/HTRU2.csv'
```

### Step 4: Reuse Existing Analysis

If you've already analyzed a dataset, just provide the `dataset_id`:

```bash
curl -X 'POST' 'http://localhost:8000/analyze-and-recommend?dataset_id=my_dataset&method=hybrid'
```

---

## Tutorial: Model Discovery

### List All Available Models

```bash
curl -X 'GET' 'http://localhost:8000/models'
```

### Filter Models by Capabilities

```bash
# CPU-only models
curl 'http://localhost:8000/models?cpu_only=true'

# Differential Privacy models
curl 'http://localhost:8000/models?requires_dp=true'

# GAN models only
curl 'http://localhost:8000/models?model_type=GAN'
```

### Get Detailed Model Information

```bash
curl 'http://localhost:8000/models/CTGAN'
```

**Response:**

```json
{
  "model_name": "CTGAN",
  "full_name": "Conditional Tabular GAN",
  "type": "GAN",
  "capabilities": {
    "skew_handling": 4,
    "cardinality_handling": 4,
    "privacy_dp": 0
  },
  "strengths": ["Handles high cardinality", "Good for mixed data types"],
  "limitations": ["No differential privacy", "Requires GPU for large datasets"]
}
```

---

## Tutorial: System Prompt Management

### Upload a New System Prompt Version

```bash
curl -X 'POST' 'http://localhost:8000/systemprompt/upload?version=v3.0&set_active=true' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./prompts/system_prompt_v3.md'
```

**Features:**

- ✅ Automatic version tracking
- ✅ Hash-based deduplication (prevents duplicate uploads)
- ✅ Only one prompt can be active at a time

### List All System Prompt Versions

```bash
curl 'http://localhost:8000/systemprompt/list'
```

**Response:**

```json
{
  "total": 3,
  "prompts": [
    {"version": "v3.0", "is_active": true, "created_at": "2026-01-18T..."},
    {"version": "v2.0", "is_active": false, "created_at": "2026-01-17T..."},
    {"version": "v1.0", "is_active": false, "created_at": "2026-01-16T..."}
  ],
  "active_version": "v3.0"
}
```

### Activate a Different Version

**By Version String:**

```bash
curl -X 'PUT' 'http://localhost:8000/systemprompt/activate/version/v2.0'
```

**By UUID:**

```bash
curl -X 'PUT' 'http://localhost:8000/systemprompt/activate/24136f21-8c30-4646-adc1-ee33be695301'
```

### Get Active System Prompt

```bash
curl 'http://localhost:8000/systemprompt/active'
```

---

## Python Client Examples

### Basic Analysis

```python
import requests

# Analyze dataset
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        params={"dataset_id": "my_data"}
    )

result = response.json()
analysis_id = result["analysis_id"]
print(f"Analysis ID: {analysis_id}")
```

### Recommendation with Analysis ID

```python
# Get recommendations using analysis_id
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "dataset_id": "my_data",
        "analysis_id": analysis_id,  # From previous step
        "method": "hybrid",
        "constraints": {"cpu_only": True},
        "top_n": 5
    }
)

recommendation = response.json()
print(f"Recommended: {recommendation['recommended_model']['model_name']}")
print(f"Confidence: {recommendation['recommended_model']['confidence_score']}")
```

### One-Shot Workflow

```python
# Analyze + Recommend in one call
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze-and-recommend",
        params={
            "method": "hybrid",
            "cpu_only": False,
            "strict_dp": False,
            "top_n": 3
        },
        files={"file": f}
    )

result = response.json()
print(f"Dataset: {result['dataset_id']}")
print(f"Recommended: {result['recommendation']['recommended_model']['model_name']}")
```

---

## Recommendation Methods

| Method | Description | Requirements | Speed |
|--------|-------------|--------------|-------|
| `rule_based` | Fast deterministic scoring based on capability matrix | None | ⚡ Instant |
| `llm` | OpenAI GPT-4 with reasoning and SystemPrompt | OPENAI_API_KEY | 🐢 ~5-10s |
| `hybrid` | Rule-based pre-filtering + LLM re-ranking | OPENAI_API_KEY | ⚖️ ~3-5s |

**Recommended:** Use `hybrid` for best results (combines speed + intelligence).

---

## Supported Models

13 State-of-the-Art synthesis models:

### By Type

- **Diffusion**: TabDDPM, TabSyn, AutoDiff
- **LLM-based**: GReaT
- **Tree-based**: TabTree, ARF
- **GAN**: CTGAN
- **VAE**: TVAE
- **Statistical**: GaussianCopula
- **Differential Privacy**: PATE-CTGAN, AIM, DPCART

### By Capability

- **High Cardinality**: CTGAN, TabDDPM, TabSyn
- **Severe Skew**: GReaT, AutoDiff, TabDDPM
- **Small Data**: GaussianCopula, TabTree, ARF
- **Privacy-Preserving**: PATE-CTGAN, AIM, DPCART

---

## Architecture

```
FastAPI Server
├── Dataset Analysis
│   ├── StochasticDataAnalyzer (dataset-level stress factors)
│   ├── ColumnAnalyzer (per-column difficulty scores)
│   └── Correlation Analysis (higher-order dependencies)
│
├── Recommendation Engine
│   ├── Rule-based scoring (model capabilities × dataset requirements)
│   ├── LLM inference (OpenAI GPT-4 with SystemPrompt)
│   └── Hybrid mode (pre-filter + re-rank)
│
├── Model Registry
│   └── model_capabilities.json (13 SOTA models)
│       ├── Capabilities (skew, cardinality, zipfian, etc.)
│       ├── Constraints (GPU, CPU, DP requirements)
│       └── Performance metrics
│
└── System Prompt Management
    ├── Version tracking & hash-based deduplication
    ├── Active prompt switching
    └── Audit logging
```

---

## Development

### Run in Debug Mode

```bash
uvicorn synthony.api.server:app --reload --log-level debug --port 8000
```

### Run Tests

```bash
# Start server in one terminal
synthony-api

# Run tests in another terminal
python scripts/test_api.py
```

### Check Syntax

```bash
python -m py_compile src/synthony/api/*.py
```

---

## Troubleshooting

### Server won't start?

- Check if port 8000 is available: `lsof -i :8000`
- Install dependencies: `pip install -e ".[api]"`
- Check Python version: `python --version` (requires 3.8+)

### LLM mode not working?

- Set OPENAI_API_KEY: `export OPENAI_API_KEY="sk-..."`
- Install openai: `pip install openai`
- Test API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

### File upload errors?

- Ensure file is CSV or Parquet format
- Check file size limits (default: no limit, but be reasonable)
- Verify file is not corrupted: `head data.csv`
- Try with `-F 'file=@path/to/file.csv;type=text/csv'`

### Empty file parameter with multipart/form-data?

**Don't** send `-F 'file='` if you want to use `dataset_id` only:

```bash
# ❌ Wrong
curl -F 'file=' -F 'dataset_id=my_data' ...

# ✅ Correct
curl -X POST 'http://localhost:8000/analyze-and-recommend?dataset_id=my_data'
```

### System prompt not activating?

Make sure you're using the correct endpoint:

- Use `/systemprompt/activate/version/{version}` for version strings (e.g., "v2.0")
- Use `/systemprompt/activate/{prompt_id}` for UUIDs

---

## Next Steps

1. ✨ Try the interactive docs: <http://localhost:8000/docs>
2. 🧪 Test with sample data: `python scripts/test_api.py`
3. 🔍 Explore model capabilities: `curl http://localhost:8000/models`
4. 📊 Analyze your own dataset: Upload a CSV and get recommendations
5. 🤖 Experiment with LLM mode: Set OPENAI_API_KEY and try `method=hybrid`

---

## API Design Principles

- **Simplicity**: One-shot endpoints for common workflows
- **Flexibility**: Separate endpoints for granular control
- **Efficiency**: Caching with `analysis_id` eliminates re-processing
- **Transparency**: Detailed reasoning in LLM responses
- **Versioning**: Track system prompt versions for reproducibility
