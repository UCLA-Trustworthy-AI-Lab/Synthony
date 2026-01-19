"""
API Endpoints for Synthony - Data Analysis & Model Recommendation.

Separated from server.py for better organization and maintainability.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from fastapi import File, HTTPException, UploadFile, Query, Body, Request, APIRouter
from pydantic import BaseModel, Field

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.core.column_analyzer import ColumnAnalyzer
from synthony.core.schemas import DatasetProfile, ColumnAnalysisResult
from synthony.recommender.engine import (
    ModelRecommendationEngine,
    RecommendationResult,
)
from synthony.api.database import (
    create_session,
    create_dataset,
    create_analysis,
    log_audit,
    create_system_prompt,
    get_active_prompt,
    list_system_prompts,
    set_active_prompt,
    set_active_prompt_by_version,
    get_dataset,
    get_analysis_by_dataset,
    get_analysis,
)
from synthony.api.storage import get_storage_manager
from synthony.api.security import get_client_info, log_error

# ============================================================================
# Create Router
# ============================================================================

router = APIRouter()

# These will be injected by server.py
analyzer: Optional[StochasticDataAnalyzer] = None
column_analyzer: Optional[ColumnAnalyzer] = None
recommender: Optional[ModelRecommendationEngine] = None


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


# Import request/response models from server
from synthony.api.server import (
    AnalysisResponse,
    RecommendationRequest,
    RecommendationMethod,
    ModelInfoResponse,
    HealthResponse,
)


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Synthony - Data Analysis & Model Recommendation API",
        "version": "0.1.0",
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
        version="0.1.0",
        analyzer_available=analyzer is not None,
        recommender_available=recommender is not None,
        llm_available=llm_available,
        models_count=models_count,
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV or Parquet file to analyze"),
    dataset_id: Optional[str] = Query(None, description="Optional dataset identifier"),
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
    3. Apply user constraints (cpu_only, strict_dp, etc.)
    4. Run recommendation engine (rule_based / llm / hybrid)
    5. Return ranked models with reasoning

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


@router.post("/analyze-and-recommend", response_model=Dict[str, Any])
async def analyze_and_recommend(
    request: Request = None,
    file: UploadFile = File(default=None),
    dataset_id: Optional[str] = Query(None, description="Dataset identifier (for reusing existing analysis or new upload)"),
    method: RecommendationMethod = Query(
        RecommendationMethod.hybrid, description="Recommendation method"
    ),
    cpu_only: bool = Query(False, description="CPU-only constraint (exclude GPU models)"),
    strict_dp: bool = Query(False, description="Strict DP requirement"),
    top_n: int = Query(3, ge=1, le=10, description="Top N alternatives"),
):
    """
    One-shot endpoint: Upload CSV/Parquet OR use existing dataset → Analyze → Recommend models.

    **Two modes**:
    1. **New upload**: Provide `file` (with optional `dataset_id` for naming)
    2. **Existing dataset**: Provide `dataset_id` only (reuses stored analysis)

    **Process**:
    1. If file provided: Upload and analyze
    2. If dataset_id only: Retrieve from database
    3. Apply constraints
    4. Generate recommendations
    5. Return combined result

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

    # Step 2: Build constraints
    constraints = {
        "cpu_only": cpu_only,
        "strict_dp": strict_dp,
    }

    # Step 3: Recommend models
    recommendation_request = RecommendationRequest(
        dataset_id=analysis_response.dataset_id,
        dataset_profile=analysis_response.dataset_profile,
        column_analysis=analysis_response.column_analysis,
        constraints=constraints,
        method=method,
        top_n=top_n,
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


@router.get("/models", response_model=Dict[str, Any])
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by type (GAN, VAE, Diffusion, Tree-based, Statistical)"),
    cpu_only: Optional[bool] = Query(None, description="Filter by CPU-only compatibility"),
    requires_dp: Optional[bool] = Query(None, description="Filter by differential privacy support"),
):
    """
    List available synthesis models from registry.

    **Filters**:
    - `model_type`: Filter by model type (GAN, VAE, Diffusion, Tree-based, Statistical)
    - `cpu_only`: Filter by CPU-only compatibility
    - `requires_dp`: Filter by differential privacy support

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

        # Differential privacy filter
        if requires_dp and info.get("capabilities", {}).get("privacy_dp", 0) == 0:
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