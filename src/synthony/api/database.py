"""
Database models and initialization for Synthony API persistence.

Provides:
- SQLite database with SQLAlchemy ORM
- Session tracking with unique IDs
- Dataset storage metadata
- Analysis result caching
- Security audit logging
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


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


# ============================================================================
# Database Session Management
# ============================================================================

_engine = None
_SessionLocal = None


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

    # Create tables (handle race condition with multiple workers)
    try:
        Base.metadata.create_all(bind=_engine)
    except Exception as e:
        # Ignore "table already exists" errors from concurrent worker startup
        if "already exists" not in str(e).lower():
            raise

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    print(f"✓ Database initialized: {database_url}")


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
        import json
        return json.loads(analysis.profile_json)
    return None
