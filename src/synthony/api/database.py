"""
Database models and initialization for Synthony API persistence.

Provides:
- SQLite database with SQLAlchemy ORM
- Session tracking with unique IDs
- Dataset storage metadata
- Analysis result caching
- Security audit logging
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import joinedload, relationship, sessionmaker

Base = declarative_base()


def _safe_json_loads(value: str | None) -> object:
    """json.loads() that degrades to the raw string on invalid JSON instead
    of raising -- needed for RunStep.input_json/output_json, which may have
    been byte-truncated mid-structure by _truncate_json before storage."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# ============================================================================
# Database Models
# ============================================================================


class Session(Base):
    """User session tracking."""

    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    expires_at = Column(DateTime, nullable=False)

    # Relationships
    datasets = relationship("Dataset", back_populates="session", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (Index("idx_session_created", "created_at"),)

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
        }


class Dataset(Base):
    """Uploaded dataset metadata."""

    __tablename__ = "datasets"

    dataset_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)  # Bytes
    format = Column(String(10))  # 'csv' or 'parquet'
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    upload_status = Column(String(20), default="uploading")  # uploading, completed, failed

    # Relationships
    session = relationship("Session", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset", cascade="all, delete-orphan")

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "format": self.format,
            "uploaded_at": self.uploaded_at.isoformat(),
            "upload_status": self.upload_status,
        }


class SystemPrompt(Base):
    """System prompt versioning for reproducibility."""

    __tablename__ = "system_prompts"

    prompt_id = Column(String(36), primary_key=True)
    version = Column(String(50), nullable=False)  # e.g., "v2.0", "2026-01-16"
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 hash
    file_path = Column(String(512))  # Original file location
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=False)  # Currently active version

    # Indexes and constraints
    __table_args__ = (
        Index("idx_prompt_version", "version"),
        Index("idx_prompt_active", "is_active"),
        Index("idx_prompt_hash", "content_hash"),
    )

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "content_length": len(self.content) if self.content else 0,
        }


class Analysis(Base):
    """Cached analysis results."""

    __tablename__ = "analyses"

    analysis_id = Column(String(36), primary_key=True)
    dataset_id = Column(String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False)
    profile_json = Column(Text)  # DatasetProfile
    column_analysis_json = Column(Text)  # ColumnAnalysisResult
    recommendation_json = Column(Text, nullable=True)  # RecommendationResult
    prompt_id = Column(String(36), ForeignKey("system_prompts.prompt_id"), nullable=True)  # System prompt used
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default="pending")  # pending, completed, failed

    # Relationships
    dataset = relationship("Dataset", back_populates="analyses")
    system_prompt = relationship("SystemPrompt")

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.isoformat(),
            "has_recommendation": self.recommendation_json is not None,
            "status": self.status,
            "prompt_version": self.system_prompt.version if self.system_prompt else None,
        }


class AuditLog(Base):
    """Security audit trail."""

    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36))
    action = Column(String(50), nullable=False)  # upload, analyze, recommend, download, delete
    endpoint = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45))
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    metadata_json = Column(Text)

    # Indexes
    __table_args__ = (
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_session", "session_id"),
    )

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "log_id": self.log_id,
            "session_id": self.session_id,
            "action": self.action,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "success": self.success,
            "error_message": self.error_message,
        }


class Run(Base):
    """Agentic-mode orchestration run (Phase 2: persistence only; the
    orchestrator that actually drives status transitions lands in Phase 3).
    """

    __tablename__ = "runs"

    run_id = Column(String(36), primary_key=True)
    session_id = Column(
        String(36), ForeignKey("sessions.session_id", ondelete="SET NULL"), nullable=True
    )
    dataset_id = Column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="SET NULL"), nullable=True
    )
    mode = Column(String(20), nullable=False)  # "advise" | "full"
    status = Column(String(20), nullable=False, default="pending")
    owner = Column(String(64))  # process/worker identity, for MCP cancel ownership
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    config_json = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    current_iteration = Column(Integer, default=0)
    llm_provider = Column(String(20), nullable=True)
    llm_model = Column(String(100), nullable=True)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)

    steps = relationship(
        "RunStep", back_populates="run", cascade="all, delete-orphan", order_by="RunStep.step_id"
    )

    __table_args__ = (
        Index("idx_run_status", "status"),
        Index("idx_run_created", "created_at"),
    )

    def to_dict(self):
        """Serialize to dictionary, including parsed config/result payloads."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "mode": self.mode,
            "status": self.status,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "current_iteration": self.current_iteration,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "config": json.loads(self.config_json) if self.config_json else None,
            "result": json.loads(self.result_json) if self.result_json else None,
            "error_message": self.error_message,
        }


class RunStep(Base):
    """One recorded step of a run, written by agents/monitoring.py's StepRecorder."""

    __tablename__ = "run_steps"

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    iteration = Column(Integer, default=0)
    agent = Column(String(50), nullable=False)
    step_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="running")
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    input_json = Column(Text, nullable=True)  # truncated to ~16KB, see _truncate_json
    output_json = Column(Text, nullable=True)  # truncated to ~16KB, see _truncate_json
    error_message = Column(Text, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    run = relationship("Run", back_populates="steps")

    __table_args__ = (Index("idx_run_step_run_id", "run_id"),)

    def to_dict(self):
        """Serialize to dictionary, including parsed input/output payloads.

        input_json/output_json are byte-truncated before storage (see
        _truncate_json), which can leave invalid JSON if the cut landed
        mid-structure -- fall back to the raw truncated string rather than
        raising, so an oversized payload degrades to "less structured",
        not "endpoint 500s".
        """
        return {
            "step_id": self.step_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "agent": self.agent,
            "step_name": self.step_name,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
            "input": _safe_json_loads(self.input_json),
            "output": _safe_json_loads(self.output_json),
            "error_message": self.error_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class SchemaMeta(Base):
    """Tiny key-value table for future schema-version tracking (no Alembic yet)."""

    __tablename__ = "schema_meta"

    key = Column(String(50), primary_key=True)
    value = Column(String(255), nullable=False)


# ============================================================================
# Database Session Management
# ============================================================================

_engine = None
_SessionLocal = None
_CURRENT_SCHEMA_VERSION = "2"  # bumped when runs/run_steps/schema_meta were added


def get_database_url() -> str:
    """Get database URL from environment or default."""
    return os.getenv("DATABASE_URL", "sqlite:///./data/synthony.db")


def init_database(database_url: str | None = None):
    """Initialize database and create tables."""
    global _engine, _SessionLocal

    if database_url is None:
        database_url = get_database_url()

    # Ensure data directory exists
    if database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create engine with SQLite-specific settings for concurrency
    echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
    connect_args = {}
    if database_url.startswith("sqlite"):
        # timeout: wait up to 30s for locks instead of failing immediately
        # check_same_thread: allow connections to be used across threads
        connect_args = {"timeout": 30, "check_same_thread": False}
    _engine = create_engine(database_url, echo=echo, connect_args=connect_args)

    if database_url.startswith("sqlite"):
        # WAL lets readers (API) and writers (agent runs, MCP) share the file
        # concurrently instead of blocking on SQLite's default file lock;
        # busy_timeout backstops the remaining brief writer-vs-writer waits.
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA busy_timeout=30000")
            _enable_wal(cursor)
            cursor.close()

        event.listen(_engine, "connect", _set_sqlite_pragma)

    # Create tables (handle race condition with multiple workers). If another
    # worker creates a table between create_all's existence check and its CREATE,
    # that CREATE fails with "already exists" and create_all stops early -- so
    # retry until it completes; existing tables are skipped on each pass.
    for attempt in range(len(Base.metadata.tables) + 1):
        try:
            Base.metadata.create_all(bind=_engine)
            break
        except Exception as e:
            if "already exists" not in str(e).lower() or attempt == len(Base.metadata.tables):
                raise

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    _ensure_schema_meta()

    print(f"✓ Database initialized: {database_url}")


def _enable_wal(cursor, attempts: int = 100, delay_s: float = 0.05):
    """Switch the database file to WAL, tolerating concurrently starting workers.

    While another connection holds a write lock, SQLite fails the switch
    immediately with "database is locked" (busy_timeout is not consulted), which
    crashed one of the two uvicorn workers in the Docker image. WAL is persisted
    in the file, so once any worker has switched it the others just see "wal".
    """
    for attempt in range(attempts):
        try:
            if cursor.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
                return
            cursor.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def _ensure_schema_meta():
    """Seed the schema_version row on first init; leave it alone otherwise
    (a real migration path can read/compare this once one exists)."""
    db = _SessionLocal()
    try:
        existing = db.query(SchemaMeta).filter(SchemaMeta.key == "schema_version").first()
        if not existing:
            db.add(SchemaMeta(key="schema_version", value=_CURRENT_SCHEMA_VERSION))
            try:
                db.commit()
            except IntegrityError:
                # Another uvicorn worker seeded the row between our check and
                # insert (same startup race as create_all above); its row is
                # the one we wanted, so there is nothing left to do.
                db.rollback()
    finally:
        db.close()


def get_db_session() -> DBSession:
    """Get database session."""
    if _SessionLocal is None:
        init_database()
    return _SessionLocal()


# ============================================================================
# Database Operations
# ============================================================================


def create_session(ip_address: str, user_agent: str, retention_days: int = 30) -> Session:
    """Create new session."""
    db = get_db_session()
    try:
        session = Session(
            session_id=str(uuid4()),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(days=retention_days),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def get_session(session_id: str) -> Session | None:
    """Retrieve session by ID."""
    db = get_db_session()
    try:
        return db.query(Session).filter(Session.session_id == session_id).first()
    finally:
        db.close()


def create_dataset(
    session_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    format: str,
) -> Dataset:
    """Register uploaded dataset."""
    db = get_db_session()
    try:
        dataset = Dataset(
            dataset_id=str(uuid4()),
            session_id=session_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            format=format,
            upload_status="completed",
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset
    finally:
        db.close()


def update_dataset_status(dataset_id: str, status: str):
    """Update dataset upload status."""
    db = get_db_session()
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if dataset:
            dataset.upload_status = status
            db.commit()
    finally:
        db.close()


def create_analysis(
    dataset_id: str,
    profile_json: str,
    column_analysis_json: str,
    recommendation_json: str | None = None,
) -> Analysis:
    """Store analysis results."""
    db = get_db_session()
    try:
        analysis = Analysis(
            analysis_id=str(uuid4()),
            dataset_id=dataset_id,
            profile_json=profile_json,
            column_analysis_json=column_analysis_json,
            recommendation_json=recommendation_json,
            status="completed",
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis
    finally:
        db.close()


def log_audit(
    session_id: str,
    action: str,
    endpoint: str,
    ip_address: str,
    success: bool = True,
    error_message: str | None = None,
    metadata: str | None = None,
):
    """Write to audit log."""
    db = get_db_session()
    try:
        log = AuditLog(
            session_id=session_id,
            action=action,
            endpoint=endpoint,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            metadata_json=metadata,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


def cleanup_expired_sessions():
    """Delete expired sessions and associated data."""
    db = get_db_session()
    try:
        expired = db.query(Session).filter(Session.expires_at < datetime.utcnow()).all()
        count = len(expired)
        for session in expired:
            db.delete(session)
        db.commit()
        return count
    finally:
        db.close()


# ============================================================================
# Session Listing/Detail Operations (for /sessions REST API)
# ============================================================================
#
# get_session() closes its DB session before returning, so touching a lazy
# relationship (e.g. session.datasets) afterward raises DetachedInstanceError.
# The functions below eager-load relationships via joinedload and, where
# nested data is needed, fully materialize to a dict while the DB session is
# still open.


def list_sessions(include_expired: bool = False, limit: int = 100, offset: int = 0) -> list[Session]:
    """List sessions, most recent first.

    Args:
        include_expired: If False (default), excludes sessions past expires_at.
        limit: Max sessions to return.
        offset: Pagination offset.
    """
    db = get_db_session()
    try:
        query = db.query(Session)
        if not include_expired:
            query = query.filter(Session.expires_at >= datetime.utcnow())
        return query.order_by(Session.created_at.desc()).offset(offset).limit(limit).all()
    finally:
        db.close()


def count_sessions(include_expired: bool = False) -> int:
    """Total session count, for pagination metadata and /storage/stats."""
    db = get_db_session()
    try:
        query = db.query(Session)
        if not include_expired:
            query = query.filter(Session.expires_at >= datetime.utcnow())
        return query.count()
    finally:
        db.close()


def count_datasets() -> int:
    """Total dataset count, for /storage/stats."""
    db = get_db_session()
    try:
        return db.query(Dataset).count()
    finally:
        db.close()


def get_session_with_details(session_id: str) -> dict | None:
    """Fetch a session plus its datasets and each dataset's analyses,
    materialized to a dict while the DB session is still open.
    """
    db = get_db_session()
    try:
        session = (
            db.query(Session)
            .options(
                joinedload(Session.datasets)
                .joinedload(Dataset.analyses)
                .joinedload(Analysis.system_prompt)
            )
            .filter(Session.session_id == session_id)
            .first()
        )
        if not session:
            return None
        result = session.to_dict()
        result["datasets"] = [
            {**dataset.to_dict(), "analyses": [a.to_dict() for a in dataset.analyses]}
            for dataset in session.datasets
        ]
        return result
    finally:
        db.close()


def get_analysis_with_relations(analysis_id: str) -> Analysis | None:
    """Like get_analysis(), but eager-loads system_prompt so callers can
    safely call .to_dict() (which accesses analysis.system_prompt.version)
    after the DB session closes.
    """
    db = get_db_session()
    try:
        return (
            db.query(Analysis)
            .options(joinedload(Analysis.system_prompt))
            .filter(Analysis.analysis_id == analysis_id)
            .first()
        )
    finally:
        db.close()


def delete_session_by_id(session_id: str) -> dict | None:
    """Delete a session row; cascades to its datasets and analyses via the
    ORM relationship cascade (same mechanism cleanup_expired_sessions() uses).

    Returns:
        {"datasets_deleted": int, "analyses_deleted": int} counted before
        delete, or None if the session doesn't exist.
    """
    db = get_db_session()
    try:
        session = (
            db.query(Session)
            .options(joinedload(Session.datasets).joinedload(Dataset.analyses))
            .filter(Session.session_id == session_id)
            .first()
        )
        if not session:
            return None
        datasets_deleted = len(session.datasets)
        analyses_deleted = sum(len(d.analyses) for d in session.datasets)
        db.delete(session)
        db.commit()
        return {"datasets_deleted": datasets_deleted, "analyses_deleted": analyses_deleted}
    finally:
        db.close()


# ============================================================================
# System Prompt Operations
# ============================================================================


def create_system_prompt(version: str, content: str, file_path: str | None = None, set_active: bool = True) -> SystemPrompt:
    """Store new system prompt version.

    Args:
        version: Version identifier (e.g., "v2.0", "2026-01-16")
        content: Prompt content
        file_path: Original file location
        set_active: Whether to set as active version

    Returns:
        SystemPrompt instance

    Raises:
        ValueError: If same version and hash already exists
    """
    import hashlib

    # Calculate content hash
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    db = get_db_session()
    try:
        # Check if version+hash already exists
        existing = db.query(SystemPrompt).filter(
            SystemPrompt.version == version,
            SystemPrompt.content_hash == content_hash
        ).first()

        if existing:
            raise ValueError(
                f"System prompt version '{version}' with identical content already exists "
                f"(prompt_id: {existing.prompt_id}). No changes detected."
            )

        # Deactivate all if setting as active
        if set_active:
            db.query(SystemPrompt).update({"is_active": False})

        prompt = SystemPrompt(
            prompt_id=str(uuid4()),
            version=version,
            content=content,
            content_hash=content_hash,
            file_path=file_path,
            is_active=set_active,
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
        return prompt
    finally:
        db.close()


def get_active_prompt() -> SystemPrompt | None:
    """Get currently active system prompt."""
    db = get_db_session()
    try:
        return db.query(SystemPrompt).filter(SystemPrompt.is_active).first()
    finally:
        db.close()


def set_active_prompt(prompt_id: str):
    """Set a specific prompt as active."""
    db = get_db_session()
    try:
        # Deactivate all
        db.query(SystemPrompt).update({"is_active": False})

        # Activate specified
        prompt = db.query(SystemPrompt).filter(SystemPrompt.prompt_id == prompt_id).first()
        if prompt:
            prompt.is_active = True
            db.commit()
            return prompt
        return None
    finally:
        db.close()


def set_active_prompt_by_version(version: str):
    """Set a specific prompt version as active.

    Args:
        version: Version identifier (e.g., "v2.0")

    Returns:
        SystemPrompt instance if found, None otherwise
    """
    db = get_db_session()
    try:
        # Find prompt by version
        prompt = db.query(SystemPrompt).filter(SystemPrompt.version == version).first()
        if not prompt:
            return None

        # Deactivate all
        db.query(SystemPrompt).update({"is_active": False})

        # Activate specified version
        prompt.is_active = True
        db.commit()
        db.refresh(prompt)
        return prompt
    finally:
        db.close()


def list_system_prompts():
    """List all system prompt versions."""
    db = get_db_session()
    try:
        return db.query(SystemPrompt).order_by(SystemPrompt.created_at.desc()).all()
    finally:
        db.close()


def get_dataset(dataset_id: str) -> Dataset | None:
    """Retrieve dataset by ID."""
    db = get_db_session()
    try:
        return db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    finally:
        db.close()


def get_analysis_by_dataset(dataset_id: str) -> Analysis | None:
    """Get most recent analysis for a dataset."""
    db = get_db_session()
    try:
        return (
            db.query(Analysis)
            .filter(Analysis.dataset_id == dataset_id)
            .order_by(Analysis.created_at.desc())
            .first()
        )
    finally:
        db.close()


def get_analysis(analysis_id: str) -> Analysis | None:
    """Retrieve analysis by analysis_id."""
    db = get_db_session()
    try:
        return db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    finally:
        db.close()


def get_dataset_profile(profile_id: str) -> dict | None:
    """Retrieve dataset profile JSON by profile_id (analysis_id).

    Args:
        profile_id: Analysis ID containing the dataset profile

    Returns:
        Dataset profile as dictionary, or None if not found
    """
    analysis = get_analysis(profile_id)
    if analysis and analysis.profile_json:
        return json.loads(analysis.profile_json)
    return None


# ============================================================================
# Run Persistence (Agentic Mode Phase 2)
# ============================================================================
#
# Run/RunStep rows are written by agents/monitoring.py's StepRecorder (and,
# from Phase 3, the orchestrator) and read by the read-only /runs API
# (api/run_endpoints.py). output_json/input_json are truncated before
# storage so a verbose tool result can't blow up row size unboundedly.

_MAX_STEP_IO_BYTES = 16 * 1024


def _truncate_json(value: str | None, max_bytes: int = _MAX_STEP_IO_BYTES) -> str | None:
    """Byte-cap a JSON string for storage, keeping it valid UTF-8 even if the
    cut lands mid-character. Truncated values are for human/log inspection,
    not re-parsed as JSON."""
    if value is None:
        return None
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "...<truncated>"


def create_run(
    mode: str,
    session_id: str | None = None,
    dataset_id: str | None = None,
    owner: str | None = None,
    config: dict | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> Run:
    """Create a new run in 'pending' status."""
    db = get_db_session()
    try:
        run = Run(
            run_id=str(uuid4()),
            session_id=session_id,
            dataset_id=dataset_id,
            mode=mode,
            status="pending",
            owner=owner,
            config_json=json.dumps(config) if config is not None else None,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def update_run_status(run_id: str, status: str, error_message: str | None = None) -> Run | None:
    """Transition a run's status. Stamps started_at the first time it leaves
    'pending', so status is always persisted before the corresponding stage
    starts doing work (a crash mid-stage leaves a truthful status)."""
    db = get_db_session()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            return None
        run.status = status
        if status != "pending" and run.started_at is None:
            run.started_at = datetime.utcnow()
        if error_message is not None:
            run.error_message = error_message
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def set_run_iteration(run_id: str, iteration: int) -> Run | None:
    """Persists the current iteration number (Phase 4 full-mode loop)."""
    db = get_db_session()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            return None
        run.current_iteration = iteration
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def finish_run(
    run_id: str, status: str, result: dict | None = None, error_message: str | None = None
) -> Run | None:
    """Mark a run as finished (completed/failed/cancelled), stamping finished_at."""
    db = get_db_session()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            return None
        run.status = status
        run.finished_at = datetime.utcnow()
        if result is not None:
            run.result_json = json.dumps(result)
        if error_message is not None:
            run.error_message = error_message
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def delete_run(run_id: str) -> bool:
    """Delete a run; cascades to its steps via the ORM relationship cascade."""
    db = get_db_session()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            return False
        db.delete(run)
        db.commit()
        return True
    finally:
        db.close()


def add_run_step(
    run_id: str,
    agent: str,
    step_name: str,
    iteration: int = 0,
    input_data: dict | None = None,
) -> RunStep:
    """Insert a running (in-progress) step row. Returns the row so the
    caller (StepRecorder) can finish it later via finish_run_step()."""
    db = get_db_session()
    try:
        step = RunStep(
            run_id=run_id,
            iteration=iteration,
            agent=agent,
            step_name=step_name,
            status="running",
            input_json=_truncate_json(json.dumps(input_data)) if input_data is not None else None,
        )
        db.add(step)
        db.commit()
        db.refresh(step)
        return step
    finally:
        db.close()


def finish_run_step(
    step_id: int,
    status: str,
    output_data: dict | None = None,
    error_message: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> RunStep | None:
    """Complete a step: stamps finished_at/duration_ms, stores truncated
    output and any per-step token/cost accounting."""
    db = get_db_session()
    try:
        step = db.query(RunStep).filter(RunStep.step_id == step_id).first()
        if not step:
            return None
        step.status = status
        step.finished_at = datetime.utcnow()
        step.duration_ms = int((step.finished_at - step.started_at).total_seconds() * 1000)
        if output_data is not None:
            step.output_json = _truncate_json(json.dumps(output_data))
        if error_message is not None:
            step.error_message = error_message
        step.input_tokens = input_tokens
        step.output_tokens = output_tokens
        step.cost_usd = cost_usd
        db.commit()
        db.refresh(step)
        return step
    finally:
        db.close()


def accumulate_run_usage(
    run_id: str, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0
) -> Run | None:
    """Roll per-call LLM usage up into the run's running totals."""
    db = get_db_session()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            return None
        run.total_input_tokens = (run.total_input_tokens or 0) + input_tokens
        run.total_output_tokens = (run.total_output_tokens or 0) + output_tokens
        run.total_cost_usd = (run.total_cost_usd or 0.0) + cost_usd
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()


def get_run(run_id: str) -> Run | None:
    """Retrieve a run by ID, without its steps."""
    db = get_db_session()
    try:
        return db.query(Run).filter(Run.run_id == run_id).first()
    finally:
        db.close()


def get_run_with_steps(run_id: str) -> dict | None:
    """Fetch a run plus its steps, materialized to a dict while the DB
    session is still open (see get_session_with_details for why)."""
    db = get_db_session()
    try:
        run = (
            db.query(Run)
            .options(joinedload(Run.steps))
            .filter(Run.run_id == run_id)
            .first()
        )
        if not run:
            return None
        result = run.to_dict()
        result["steps"] = [s.to_dict() for s in run.steps]
        return result
    finally:
        db.close()


def list_run_steps(run_id: str) -> list[RunStep]:
    """List a run's steps in execution order."""
    db = get_db_session()
    try:
        return (
            db.query(RunStep)
            .filter(RunStep.run_id == run_id)
            .order_by(RunStep.step_id)
            .all()
        )
    finally:
        db.close()


def list_runs(status: str | None = None, limit: int = 100, offset: int = 0) -> list[Run]:
    """List runs, most recent first."""
    db = get_db_session()
    try:
        query = db.query(Run)
        if status is not None:
            query = query.filter(Run.status == status)
        return query.order_by(Run.created_at.desc()).offset(offset).limit(limit).all()
    finally:
        db.close()


def count_runs(status: str | None = None) -> int:
    """Total run count, for /runs pagination metadata."""
    db = get_db_session()
    try:
        query = db.query(Run)
        if status is not None:
            query = query.filter(Run.status == status)
        return query.count()
    finally:
        db.close()
