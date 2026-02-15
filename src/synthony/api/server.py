#!/usr/bin/env python3
"""
FastAPI server for Synthony data analysis and model recommendation.

Exposes REST endpoints for:
- Dataset profiling (CSV upload → StressProfile)
- Model recommendation (Profile → Recommended models)
- Combined workflow (CSV → Analysis → Recommendation)
"""

from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from synthony.api.database import (
    create_analysis,
    create_dataset,
    create_session,
    get_analysis,
    get_analysis_by_dataset,
    get_dataset,
    init_database,
    log_audit,
)
from synthony.api.security import get_client_info, log_error
from synthony.api.storage import get_storage_manager
from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.core.schemas import ColumnAnalysisResult, DatasetProfile
from synthony.recommender.engine import (
    ModelRecommendationEngine,
    RecommendationResult,
)

__version__ = "0.1.0"

# ============================================================================
# API Models (Request/Response Schemas)
# ============================================================================


class RecommendationMethod(str, Enum):
    """Recommendation method selection."""

    rule_based = "rule_based"
    llm = "llm"
    hybrid = "hybrid"


class AnalysisRequest(BaseModel):
    """Request model for analysis from pre-loaded data."""

    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    data_json: str = Field(
        ..., description="JSON string of pandas DataFrame (orient='records')"
    )


class RecommendationRequest(BaseModel):
    """Request model for recommendation from existing profile."""

    dataset_id: str = Field(..., description="Dataset identifier")
    analysis_id: str | None = Field(None, description="Analysis ID to retrieve cached profile (optional if dataset_profile provided)")
    dataset_profile: dict[str, Any] | None = Field(None, description="Dataset profile from StochasticDataAnalyzer (optional if analysis_id provided)")
    dataset_profile_id: str | None = Field(None, description="Dataset profile ID to retrieve stored profile (alternative to analysis_id)")
    column_analysis: dict[str, Any] | None = Field(
        None, description="Optional column-level analysis"
    )
    method: RecommendationMethod = Field(
        default=RecommendationMethod.hybrid, description="Recommendation method"
    )
    top_n: int = Field(default=3, ge=1, le=10, description="Number of alternatives to return")
    constraints: dict[str, Any] | None = Field(default=None, description="Constraints: cpu_only, strict_dp, prefer_speed")


class AnalysisResponse(BaseModel):
    """Response model for dataset analysis."""

    session_id: str
    analysis_id: str
    dataset_id: str
    dataset_profile: dict[str, Any]
    column_analysis: dict[str, Any]
    message: str = "Analysis completed successfully"


class ModelInfoResponse(BaseModel):
    """Response model for model registry information."""

    model_name: str
    full_name: str
    type: str
    capabilities: dict[str, int]
    constraints: dict[str, Any]
    performance: dict[str, str]
    description: str
    strengths: list[str]
    limitations: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    analyzer_available: bool = True
    recommender_available: bool = True
    llm_available: bool = False
    models_count: int = 0


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Synthony - Data Analysis & Model Recommendation API",
    description="""
    Intelligent orchestration platform for synthetic data generation model selection.

    **Features**:
    - Dataset profiling (Skewness, Cardinality, Zipfian, Correlation analysis)
    - Model recommendation (13+ SOTA models from table-synthesizers)
    - Hybrid rule-based + LLM decision engine

    **Recommendation Methods**:
    - `rule_based`: Fast deterministic scoring (no API key needed)
    - `llm`: OpenAI GPT-4 with SystemPrompt reasoning (requires OPENAI_API_KEY)
    - `hybrid`: Rule-based pre-filtering + LLM re-ranking (best of both)
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State (Initialized on startup)
# ============================================================================

analyzer: StochasticDataAnalyzer | None = None
column_analyzer: ColumnAnalyzer | None = None
recommender: ModelRecommendationEngine | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize analyzers and recommendation engine on startup."""
    import os

    from dotenv import load_dotenv
    global analyzer, column_analyzer, recommender

    # Load environment variables from .env file (checks multiple locations)
    env_paths = [
        Path("/app/.env"),          # Docker mount location
        Path.cwd() / ".env",        # Current working directory
        Path(__file__).parent.parent.parent.parent / ".env",  # Project root
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment from: {env_path}")
            break
    else:
        print("ℹ No .env file found, using system environment variables")

    # Initialize database
    init_database()

    # Initialize data analyzer
    analyzer = StochasticDataAnalyzer()

    # Initialize column analyzer
    column_analyzer = ColumnAnalyzer()

    # Initialize recommendation engine
    # Check for OpenAI or VLLM credentials
    try:
        vllm_url = os.getenv("VLLM_URL")

        if vllm_url:
            # VLLM mode
            openai_api_key = os.getenv("VLLM_API_KEY")
            openai_base_url = vllm_url
            openai_model = os.getenv("VLLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o")
        else:
            # OpenAI mode
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_base_url = os.getenv("OPENAI_URL")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        recommender = ModelRecommendationEngine(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
        )
        print("✓ Recommendation engine initialized")
        if recommender.openai_client:
            if openai_base_url:
                print(f"✓ LLM mode available (VLLM at {openai_base_url}, model: {recommender.openai_model})")
            else:
                print(f"✓ LLM mode available (OpenAI, model: {recommender.openai_model})")
        else:
            print("⚠ LLM mode unavailable (no API key). Using rule-based mode.")
    except Exception as e:
        print(f"⚠ Failed to initialize recommendation engine: {e}")
        recommender = None

    # Inject services into endpoints module
    from synthony.api import endpoints
    endpoints.set_services(analyzer, column_analyzer, recommender)


# ============================================================================
# Include Endpoints Router
# ============================================================================

# Import after app creation to avoid circular dependency (endpoints imports from server)
from synthony.api.endpoints import router as api_router  # noqa: E402

app.include_router(api_router)

# ============================================================================
# System Prompt Management Endpoints (kept in server.py)
# ============================================================================


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Synthony - Data Analysis & Model Recommendation API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global analyzer, column_analyzer, recommender

    models_count = 0
    llm_available = False

    if recommender:
        models_count = len(recommender.model_capabilities.get("models", {}))
        llm_available = recommender.openai_client is not None

    return HealthResponse(
        status= "healthy" if analyzer and recommender else "unhealthy",
        version=__version__,
        analyzer_available=analyzer is not None,
        recommender_available=recommender is not None,
        llm_available=llm_available,
        models_count=models_count,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV or Parquet file to analyze"),
    dataset_id: str | None = Query(None, description="Optional dataset identifier"),
    request: Request = None,
):
    """
    Analyze a CSV dataset and return stress profile with persistent storage.

    **NEW**: Returns `session_id` and `analysis_id` for later retrieval.

    **Process**:
    1. Create session and upload file to persistent storage
    2. Run StochasticDataAnalyzer (skewness, cardinality, zipfian, correlation)
    3. Run ColumnAnalyzer (per-column stress factors and difficulty scores)
    4. Cache analysis results in database
    5. Return DatasetProfile + ColumnAnalysis JSON with session tracking

    **Returns**: DatasetProfile with stress factors and ColumnAnalysis with session IDs
    """
    if not analyzer or not column_analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    # Validate file type
    if not file.filename.endswith(".csv") and not file.filename.endswith(".parquet"):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")

    # Generate dataset ID if not provided
    if not dataset_id:
        dataset_id = file.filename.replace(".csv", "").replace(".parquet", "")

    # Get client info for session tracking
    ip_address, user_agent = get_client_info(request)

    try:
        # Create session
        session = create_session(ip_address, user_agent)
        session_id = session.session_id

        # Read uploaded file
        contents = await file.read()

        # Save to persistent storage
        storage = get_storage_manager()
        file_path, file_size = storage.save_file(
            contents, file.filename, session_id, dataset_id
        )

        # Register dataset in database
        dataset = create_dataset(
            session_id=session_id,
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            format=file.filename.split(".")[-1],
        )

        # Load as DataFrame
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")

        # Run dataset-level analysis
        dataset_profile = analyzer.analyze(df)

        # Run column-level analysis
        column_analysis_result = column_analyzer.analyze(df, dataset_profile)

        # Serialize excluding DataFrame fields
        profile_json = dataset_profile.model_dump_json(exclude={'correlation': {'correlation_matrix'}})
        column_json = column_analysis_result.model_dump_json()

        # Cache analysis in database
        analysis = create_analysis(
            dataset_id=dataset.dataset_id,
            profile_json=profile_json,
            column_analysis_json=column_json,
        )

        # Log successful analysis
        log_audit(
            session_id=session_id,
            action="analyze",
            endpoint="/analyze",
            ip_address=ip_address,
            success=True,
        )

        # Serialize with DataFrame exclusion
        profile_dict = dataset_profile.model_dump(exclude={'correlation': {'correlation_matrix'}})
        column_dict = column_analysis_result.model_dump()

        return AnalysisResponse(
            session_id=session_id,
            analysis_id=analysis.analysis_id,
            dataset_id=dataset_id,
            dataset_profile=profile_dict,
            column_analysis=column_dict,
            message=f"Analysis completed: {dataset_profile.row_count} rows × {dataset_profile.column_count} columns",
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV or Parquet file is empty")
    except ValueError as e:
        # Storage quota or validation errors
        error_msg = log_error(session_id if 'session_id' in locals() else None, "analyze", e)
        log_audit(
            session_id=session_id if 'session_id' in locals() else None,
            action="analyze",
            endpoint="/analyze",
            ip_address=ip_address,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV or Parquet: {str(e)}")
    except Exception as e:
        error_msg = log_error(session_id if 'session_id' in locals() else None, "analyze", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@app.post("/recommend", response_model=RecommendationResult)
async def recommend_model(request: RecommendationRequest = Body(...)):
    """
    Recommend synthesis models based on dataset profile.

    **Two modes**:
    1. **Provide analysis_id**: Retrieve cached analysis from database
    2. **Provide dataset_profile**: Use provided profile directly

    **Process**:
    1. If analysis_id provided: Retrieve from database
    2. Otherwise: Use provided dataset_profile + column_analysis
    3. Run recommendation engine (rule_based / llm / hybrid)
    4. Return ranked models with reasoning

    **Recommendation Methods**:
    - `rule_based`: Fast scoring based on capability matrix (no API key needed)
    - `llm`: LLM inference with SystemPrompt (requires API key)
    - `hybrid`: Rule-based pre-filtering + LLM re-ranking (recommended)

    **Returns**: RecommendationResult with primary model, alternatives, reasoning, and warnings
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    try:
        # Mode 1: Retrieve from database using analysis_id
        if request.analysis_id:
            analysis = get_analysis(request.analysis_id)

            if not analysis:
                raise HTTPException(
                    status_code=404,
                    detail=f"No analysis found for analysis_id '{request.analysis_id}'. Please run /analyze first."
                )

            # Deserialize stored analysis
            import json
            dataset_profile = DatasetProfile(**json.loads(analysis.profile_json))
            column_analysis = ColumnAnalysisResult(**json.loads(analysis.column_analysis_json)) if analysis.column_analysis_json else None

        # Mode 2: Use provided profile
        elif request.dataset_profile:
            # Convert dictionaries to Pydantic objects if needed
            if isinstance(request.dataset_profile, dict):
                dataset_profile = DatasetProfile(**request.dataset_profile)
            else:
                dataset_profile = request.dataset_profile

            if request.column_analysis and isinstance(request.column_analysis, dict):
                column_analysis = ColumnAnalysisResult(**request.column_analysis)
            else:
                column_analysis = request.column_analysis

        # Validation: must have either analysis_id or dataset_profile
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'analysis_id' or 'dataset_profile' must be provided"
            )

        # Run recommendation
        result = recommender.recommend(
            dataset_profile=dataset_profile,
            column_analysis=column_analysis,
            constraints=request.constraints,
            method=request.method.value,
            top_n=request.top_n,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/analyze-and-recommend", response_model=dict[str, Any])
async def analyze_and_recommend(
    request: Request = None,
    file: UploadFile = File(default=None),
    dataset_id: str | None = Query(None, description="Dataset identifier (for reusing existing analysis or new upload)"),
    method: RecommendationMethod = Query(
        RecommendationMethod.hybrid, description="Recommendation method"
    ),
    top_n: int = Query(3, ge=1, le=10, description="Top N alternatives"),
    cpu_only: bool = Query(False, description="Only recommend CPU-compatible models"),
    strict_dp: bool = Query(False, description="Only recommend models with strong differential privacy"),
):
    """
    One-shot endpoint: Upload CSV/Parquet OR use existing dataset → Analyze → Recommend models.

    **Two modes**:
    1. **New upload**: Provide `file` (with optional `dataset_id` for naming)
    2. **Existing dataset**: Provide `dataset_id` only (reuses stored analysis)

    **Process**:
    1. If file provided: Upload and analyze
    2. If dataset_id only: Retrieve from database
    3. Generate recommendations
    4. Return combined result

    **Returns**: Combined response with analysis + recommendation
    """
    if not analyzer or not column_analyzer or not recommender:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Treat empty file uploads as None (handles both omitted and empty form fields)
    if file and (not file.filename or file.filename == ""):
        file = None

    # Validate input: must have either file or dataset_id
    if not file and not dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' (for new upload) or 'dataset_id' (for existing data) must be provided"
        )

    # Mode 1: New file upload
    if file:
        # Step 1: Analyze dataset (will create new entry or update existing if dataset_id provided)
        analysis_response = await analyze_dataset(file=file, dataset_id=dataset_id, request=request)

    # Mode 2: Use existing dataset
    else:
        # Retrieve existing analysis from database
        analysis = get_analysis_by_dataset(dataset_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis found for dataset_id '{dataset_id}'. Please upload the dataset first using /analyze endpoint."
            )

        # Deserialize stored analysis
        import json
        profile_dict = json.loads(analysis.profile_json)
        column_dict = json.loads(analysis.column_analysis_json)

        # Get session_id from dataset
        dataset = get_dataset(dataset_id)
        session_id = dataset.session_id if dataset else None

        # Create response object matching analyze_dataset output
        analysis_response = AnalysisResponse(
            session_id=session_id,
            analysis_id=analysis.analysis_id,
            dataset_id=dataset_id,
            dataset_profile=profile_dict,
            column_analysis=column_dict,
            message=f"Using cached analysis from {analysis.created_at.isoformat()}"
        )

    # Step 2: Recommend models
    constraints = {}
    if cpu_only:
        constraints["cpu_only"] = True
    if strict_dp:
        constraints["strict_dp"] = True

    recommendation_request = RecommendationRequest(
        dataset_id=analysis_response.dataset_id,
        dataset_profile=analysis_response.dataset_profile,
        column_analysis=analysis_response.column_analysis,
        method=method,
        top_n=top_n,
        constraints=constraints if constraints else None,
    )

    recommendation_result = await recommend_model(request=recommendation_request)

    # Step 3: Combine results
    return {
        "dataset_id": analysis_response.dataset_id,
        "analysis": {
            "dataset_profile": analysis_response.dataset_profile,
            "column_analysis": analysis_response.column_analysis,
        },
        "recommendation": recommendation_result,
    }


@app.get("/models", response_model=dict[str, Any])
async def list_models(
    model_type: str | None = Query(None, description="Filter by type (GAN, VAE, Diffusion, Tree-based, Statistical)"),
    cpu_only: bool = Query(False, description="Only show CPU-compatible models"),
    requires_dp: bool = Query(False, description="Only show models with differential privacy support"),
):
    """
    List available synthesis models from registry.

    **Filters**:
    - `model_type`: Filter by model type (GAN, VAE, Diffusion, Tree-based, Statistical)
    - `cpu_only`: Only show CPU-compatible models
    - `requires_dp`: Only show models with differential privacy support

    **Returns**: List of models with capabilities, constraints, and descriptions
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    models = recommender.model_capabilities.get("models", {})

    # Apply filters
    filtered_models = {}
    for name, info in models.items():
        # Type filter
        if model_type and info.get("type", "").lower() != model_type.lower():
            continue

        # CPU-only filter
        if cpu_only and not info.get("constraints", {}).get("cpu_only_compatible", False):
            continue

        # DP filter (threshold from registry metadata.dp_threshold)
        dp_threshold = recommender.model_capabilities.get("metadata", {}).get("dp_threshold", 3)
        if requires_dp and info.get("capabilities", {}).get("privacy_dp", 0) < dp_threshold:
            continue

        filtered_models[name] = info

    return {
        "total_models": len(models),
        "filtered_models": len(filtered_models),
        "models": filtered_models,
        "model_ranking": recommender.model_capabilities.get("model_ranking_by_capability", {}),
    }


@app.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.

    **Returns**: Full model specification including capabilities, constraints, strengths, and limitations
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    models = recommender.model_capabilities.get("models", {})

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}",
        )

    model_info = models[model_name]

    return ModelInfoResponse(
        model_name=model_info["name"],
        full_name=model_info["full_name"],
        type=model_info["type"],
        capabilities=model_info["capabilities"],
        constraints=model_info["constraints"],
        performance=model_info["performance"],
        description=model_info["description"],
        strengths=model_info["strengths"],
        limitations=model_info["limitations"],
    )

# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for running the API server."""
    import uvicorn

    print("=" * 80)
    print("Synthony - Data Analysis & Model Recommendation API")
    print("=" * 80)
    print("\nStarting server at http://0.0.0.0:8000")
    print("API Documentation: http://0.0.0.0:8000/docs")
    print("Alternative Docs: http://0.0.0.0:8000/redoc")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(
        "synthony.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
