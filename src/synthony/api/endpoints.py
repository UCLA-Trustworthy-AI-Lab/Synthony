"""
API Endpoints for Synthony - Data Analysis & Model Recommendation.

Separated from server.py for better organization and maintainability.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from synthony.api.database import (
    count_datasets,
    count_sessions,
    create_analysis,
    create_dataset,
    create_session,
    create_system_prompt,
    delete_session_by_id,
    get_active_prompt,
    get_analysis,
    get_analysis_by_dataset,
    get_analysis_with_relations,
    get_dataset,
    get_dataset_profile,
    get_session,
    get_session_with_details,
    list_sessions,
    list_system_prompts,
    log_audit,
    set_active_prompt,
    set_active_prompt_by_version,
)
from synthony.api.security import get_client_info, log_error
from synthony.api.server import (
    AnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
    RecommendationMethod,
    RecommendationRequest,
)
from synthony.api.storage import get_storage_manager
from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.core.schemas import ColumnAnalysisResult, DatasetProfile
from synthony.recommender.engine import (
    ModelRecommendationEngine,
    RecommendationResult,
)

# ============================================================================
# Create Router
# ============================================================================

router = APIRouter()

# These will be injected by server.py
analyzer: StochasticDataAnalyzer | None = None
column_analyzer: ColumnAnalyzer | None = None
recommender: ModelRecommendationEngine | None = None


def set_services(
    analyzer_inst: StochasticDataAnalyzer,
    column_analyzer_inst: ColumnAnalyzer,
    recommender_inst: ModelRecommendationEngine,
):
    """Set service instances from server.py"""
    global analyzer, column_analyzer, recommender
    analyzer = analyzer_inst
    column_analyzer = column_analyzer_inst
    recommender = recommender_inst


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Synthony - Data Analysis & Model Recommendation API",
        "version": "0.1.1",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_count = 0
    llm_available = False

    if recommender:
        models_count = len(recommender.model_capabilities.get("models", {}))
        llm_available = recommender.openai_client is not None

    return HealthResponse(
        status="healthy" if analyzer and recommender else "unhealthy",
        version="0.1.1",
        analyzer_available=analyzer is not None,
        recommender_available=recommender is not None,
        llm_available=llm_available,
        models_count=models_count,
    )


@router.post("/analyze", response_model=AnalysisResponse)
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
            dataset_id=dataset.dataset_id,
            dataset_profile=profile_dict,
            column_analysis=column_dict,
            message=f"Analysis completed: {dataset_profile.row_count} rows × {dataset_profile.column_count} columns",
        )

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
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV or Parquet file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV or Parquet: {str(e)}")
    except Exception as e:
        error_msg = log_error(session_id if 'session_id' in locals() else None, "analyze", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@router.post("/recommend", response_model=RecommendationResult)
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
            dataset_profile = DatasetProfile(**json.loads(analysis.profile_json))
            column_analysis = ColumnAnalysisResult(**json.loads(analysis.column_analysis_json)) if analysis.column_analysis_json else None

        # Mode 2: Use provided profile
        elif request.dataset_profile or request.dataset_profile_id:
            # Convert dictionaries to Pydantic objects if needed
            if isinstance(request.dataset_profile, dict):
                dataset_profile = DatasetProfile(**request.dataset_profile)
            elif isinstance(request.dataset_profile_id, str):
                profile_dict = get_dataset_profile(request.dataset_profile_id)
                if not profile_dict:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No dataset profile found for ID '{request.dataset_profile_id}'"
                    )
                dataset_profile = DatasetProfile(**profile_dict)
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
                detail="Either 'analysis_id' or 'dataset_profile_id' must be provided"
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


@router.post("/analyze-and-recommend", response_model=dict[str, Any])
async def analyze_and_recommend(
    request: Request = None,
    file: UploadFile = File(default=None),
    dataset_id: str | None = Query(None, description="Dataset identifier (for naming a new upload)"),
    dataset_profile_id: str | None = Query(None, description="Existing dataset profile ID to use (previously analyzed)"),
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
    2. **Existing dataset**: Provide `dataset_profile_id` only (reuses stored analysis)

    **Process**:
    1. If file provided: Upload and analyze
    2. If dataset_profile_id provided: Retrieve from database
    3. Generate recommendations
    4. Return combined result

    **Returns**: Combined response with analysis + recommendation
    """
    if not analyzer or not column_analyzer or not recommender:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Treat empty file uploads as None (handles both omitted and empty form fields)
        if file and (not file.filename or file.filename == ""):
            file = None

        # Validate input: must have either file or dataset_profile_id
        if not file and not dataset_profile_id:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' (for new upload) or 'dataset_profile_id' (for existing data) must be provided"
            )

        # Mode 1: New file upload
        if file:
            # Step 1: Analyze dataset (will create new entry or update existing if dataset_id provided)
            # Pass dataset_id only if it's meant for naming the new upload
            analysis_response = await analyze_dataset(file=file, dataset_id=dataset_id, request=request)

        # Mode 2: Use existing dataset
        elif dataset_profile_id:
            # Retrieve existing analysis from database
            # Note: In current schema, dataset_id is often used as the key.
            # We map dataset_profile_id to the lookups expected by get_analysis_by_dataset
            analysis = get_analysis_by_dataset(dataset_profile_id)

            if not analysis:
                raise HTTPException(
                    status_code=404,
                    detail=f"No analysis found for dataset_profile_id '{dataset_profile_id}'. Please upload the dataset first using /analyze endpoint."
                )

            # Deserialize stored analysis
            try:
                profile_dict = json.loads(analysis.profile_json)
                column_dict = json.loads(analysis.column_analysis_json)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to decode stored analysis data: {str(e)}")

            # Get session_id from dataset
            dataset = get_dataset(dataset_profile_id)
            session_id = dataset.session_id if dataset else None

            # Create response object matching analyze_dataset output
            analysis_response = AnalysisResponse(
                session_id=session_id,
                analysis_id=analysis.analysis_id,
                dataset_id=dataset_profile_id,
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

        # Step 4: Combine results
        return {
            "dataset_id": analysis_response.dataset_id,
            "analysis": {
                "dataset_profile": analysis_response.dataset_profile,
                "column_analysis": analysis_response.column_analysis,
            },
            "recommendation": recommendation_result,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch unexpected errors and return meaningful 500
        # In a real app, we might log the full traceback here
        error_msg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during analysis and recommendation: {error_msg}"
        )


@router.get("/models", response_model=dict[str, Any])
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


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
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
# System Prompt Management Endpoints
# ============================================================================


@router.post("/systemprompt/upload")
async def upload_system_prompt(
    file: UploadFile = File(..., description="System prompt markdown file"),
    version: str = Query(..., description="Version identifier (e.g., v2.1, 2026-01-16)"),
    set_active: bool = Query(True, description="Set as active prompt version"),
    request: Request = None,
):
    """
    Upload and version system prompt for recommendations.

    **Versioning**: Each upload creates a new version in the database
    **Active**: Only one prompt can be active at a time
    **Tracking**: All analyses/recommendations link to the prompt version used

    **Returns**: Prompt metadata with version info
    """
    # Validate file type
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only Markdown (.md) files are supported")

    # Get client info
    ip_address, user_agent = get_client_info(request)

    try:
        # Read file content
        content = (await file.read()).decode('utf-8')

        # Save to uploads directory
        prompt_dir = Path("./data/uploads/systemprompt")
        prompt_dir.mkdir(parents=True, exist_ok=True)

        file_path = prompt_dir / f"{version}_{file.filename}"
        file_path.write_text(content)

        # Store in database
        prompt = create_system_prompt(
            version=version,
            content=content,
            file_path=str(file_path),
            set_active=set_active
        )

        # Log audit
        log_audit(
            session_id=None,  # System-level action
            action="upload_system_prompt",
            endpoint="/systemprompt/upload",
            ip_address=ip_address,
            success=True,
            metadata=f"version={version}, active={set_active}"
        )

        return {
            "prompt_id": prompt.prompt_id,
            "version": prompt.version,
            "is_active": prompt.is_active,
            "content_length": len(content),
            "file_path": str(file_path),
            "message": f"System prompt {version} uploaded {'and set as active' if set_active else 'successfully'}"
        }

    except Exception as e:
        error_msg = log_error(None, "upload_prompt", e)
        raise HTTPException(status_code=500, detail=f"Failed to upload system prompt: {error_msg}")


@router.get("/systemprompt/list")
async def list_system_prompts_endpoint():
    """
    List all system prompt versions.

    **Returns**: List of all prompts with active status
    """
    prompts = list_system_prompts()
    return {
        "total": len(prompts),
        "prompts": [p.to_dict() for p in prompts],
        "active_version": next((p.version for p in prompts if p.is_active), None)
    }


@router.get("/systemprompt/active")
async def get_active_system_prompt():
    """
    Get currently active system prompt.

    **Returns**: Active prompt content and metadata
    """
    prompt = get_active_prompt()
    if not prompt:
        raise HTTPException(status_code=404, detail="No active system prompt found")

    return {
        "prompt_id": prompt.prompt_id,
        "version": prompt.version,
        "content": prompt.content,
        "created_at": prompt.created_at.isoformat(),
        "content_length": len(prompt.content)
    }

@router.put("/systemprompt/activate/{prompt_id}")
async def activate_system_prompt(prompt_id: str):
    """
    Set a specific prompt version as active.

    **Returns**: Confirmation message
    """
    set_active_prompt(prompt_id)

    return {
        "activated": True,
        "prompt_id": prompt_id,
        "message": "System prompt version activated successfully"
    }


@router.put("/systemprompt/activate/version/{version}")
async def activate_system_prompt_by_version(version: str):
    """
    Set a specific prompt version as active by version string.

    **Example**: PUT `/systemprompt/activate/version/v2.0`

    **Returns**: Confirmation message with prompt details
    """
    prompt = set_active_prompt_by_version(version)

    if not prompt:
        raise HTTPException(
            status_code=404,
            detail=f"System prompt version '{version}' not found"
        )

    return {
        "activated": True,
        "prompt_id": prompt.prompt_id,
        "version": prompt.version,
        "message": f"System prompt version '{version}' is now active"
    }


# ============================================================================
# Session Management Endpoints
#
# No authentication exists anywhere in this API today; session_id (an
# unguessable UUID) is already the sole thing gating access to a dataset's
# contents for existing routes like /analyze. Detail/delete/download-by-
# known-ID below are consistent with that existing trust model. GET
# /sessions (list-all) is different: it lets a caller discover session_ids
# they don't already have, which nothing today provides, so it's gated
# behind ENABLE_SESSION_LISTING (default off) and omits ip_address.
# ============================================================================


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    expires_at: str


class SessionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    sessions: list[SessionSummary]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    files_deleted: int
    bytes_freed: int
    datasets_deleted: int
    analyses_deleted: int
    message: str


class StorageStatsResponse(BaseModel):
    total_size_gb: float
    storage_limit_gb: int
    usage_percent: float
    active_sessions: int
    total_datasets: int
    db_active_sessions: int
    db_total_datasets: int


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(100, ge=1, le=500, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    include_expired: bool = Query(False, description="Include expired sessions"),
):
    """
    List sessions with metadata. Excludes expired sessions by default.

    **Disabled by default** — set `ENABLE_SESSION_LISTING=true` to enable.
    Unlike the other session routes, this lets a caller discover session_ids
    they don't already have, so it's opt-in.
    """
    if os.getenv("ENABLE_SESSION_LISTING", "false").lower() != "true":
        raise HTTPException(
            status_code=404,
            detail="Session listing is disabled. Set ENABLE_SESSION_LISTING=true to enable.",
        )

    sessions = list_sessions(include_expired=include_expired, limit=limit, offset=offset)
    total = count_sessions(include_expired=include_expired)
    return SessionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        sessions=[
            SessionSummary(session_id=s.session_id, created_at=s.created_at.isoformat(), expires_at=s.expires_at.isoformat())
            for s in sessions
        ],
    )


@router.get("/sessions/{session_id}", response_model=dict[str, Any])
async def get_session_detail(session_id: str):
    """Retrieve session details with all datasets and analyses."""
    detail = get_session_with_details(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return detail


@router.get("/sessions/{session_id}/data/{dataset_id}")
async def download_dataset_file(session_id: str, dataset_id: str, request: Request = None):
    """Download the original uploaded file for a dataset."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    dataset = get_dataset(dataset_id)
    if not dataset or dataset.session_id != session_id:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found in session '{session_id}'",
        )

    # Use the path recorded at upload time rather than reconstructing it from
    # dataset_id: /analyze's storage filename is keyed by a locally-derived
    # dataset_id (filename-based unless explicitly passed), which can differ
    # from the DB's generated Dataset.dataset_id. file_path is authoritative.
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    ip_address, _ = get_client_info(request)
    log_audit(
        session_id=session_id,
        action="download",
        endpoint=f"/sessions/{session_id}/data/{dataset_id}",
        ip_address=ip_address,
        success=True,
    )

    media_type = "text/csv" if dataset.format == "csv" else "application/octet-stream"
    return FileResponse(path=str(file_path), filename=dataset.filename, media_type=media_type)


@router.get("/sessions/{session_id}/analyses/{analysis_id}", response_model=dict[str, Any])
async def get_analysis_detail(session_id: str, analysis_id: str):
    """Retrieve cached analysis results, scoped to the given session."""
    analysis = get_analysis_with_relations(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")

    dataset = get_dataset(analysis.dataset_id)
    if not dataset or dataset.session_id != session_id:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found in session '{session_id}'",
        )

    return {
        **analysis.to_dict(),
        "dataset_profile": json.loads(analysis.profile_json) if analysis.profile_json else None,
        "column_analysis": json.loads(analysis.column_analysis_json) if analysis.column_analysis_json else None,
        "recommendation": json.loads(analysis.recommendation_json) if analysis.recommendation_json else None,
    }


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session_endpoint(session_id: str, request: Request = None):
    """
    Delete a session and all associated data (datasets, analyses, and
    uploaded files).

    Deletes uploaded files first, then the database rows: if the filesystem
    step fails, the session row is untouched and a retry re-attempts both;
    if it succeeds but the DB step then fails, a retry's filesystem step is
    a safe no-op and only the DB delete needs to succeed.
    """
    ip_address, _ = get_client_info(request)

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    try:
        files_deleted, bytes_freed = get_storage_manager().delete_session(session_id)
    except Exception as e:
        error_msg = log_error(session_id, "delete", e)
        log_audit(
            session_id=session_id,
            action="delete",
            endpoint=f"/sessions/{session_id}",
            ip_address=ip_address,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete session files: {error_msg}")

    try:
        db_summary = delete_session_by_id(session_id)
    except Exception as e:
        error_msg = log_error(session_id, "delete", e)
        log_audit(
            session_id=session_id,
            action="delete",
            endpoint=f"/sessions/{session_id}",
            ip_address=ip_address,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete session database records: {error_msg}")

    if db_summary is None:
        # Session existed above but was deleted concurrently (e.g. by
        # cleanup_expired_sessions). Files are already gone too; treat as
        # a successful, idempotent outcome.
        db_summary = {"datasets_deleted": 0, "analyses_deleted": 0}

    log_audit(
        session_id=session_id,
        action="delete",
        endpoint=f"/sessions/{session_id}",
        ip_address=ip_address,
        success=True,
        metadata=json.dumps({"files_deleted": files_deleted, "bytes_freed": bytes_freed}),
    )

    return SessionDeleteResponse(
        session_id=session_id,
        deleted=True,
        files_deleted=files_deleted,
        bytes_freed=bytes_freed,
        datasets_deleted=db_summary["datasets_deleted"],
        analyses_deleted=db_summary["analyses_deleted"],
        message=(
            f"Session {session_id} deleted: {files_deleted} files ({bytes_freed} bytes), "
            f"{db_summary['datasets_deleted']} datasets, {db_summary['analyses_deleted']} analyses removed"
        ),
    )


@router.get("/storage/stats", response_model=StorageStatsResponse)
async def get_storage_stats_endpoint():
    """Get storage usage statistics, cross-checked against DB session/dataset counts."""
    fs_stats = get_storage_manager().get_storage_stats()
    return StorageStatsResponse(
        **fs_stats,
        db_active_sessions=count_sessions(include_expired=False),
        db_total_datasets=count_datasets(),
    )
