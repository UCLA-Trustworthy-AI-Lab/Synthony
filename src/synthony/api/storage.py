"""
File storage management for uploaded datasets.

Handles:
- Secure file storage with sanitized filenames
- Storage quota enforcement
- File retrieval and cleanup
- Upload status tracking
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4


class StorageManager:
    """Manages persistent file storage for uploaded datasets."""

    def __init__(self, upload_dir: str = "./data/uploads"):
        """Initialize storage manager.

        Args:
            upload_dir: Base directory for uploads
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
        self.max_session_storage_mb = int(os.getenv("MAX_SESSION_STORAGE_MB", "500"))
        self.max_total_storage_gb = int(os.getenv("MAX_TOTAL_STORAGE_GB", "10"))

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove path separators
        filename = os.path.basename(filename)

        # Remove special characters except dots, dashes, underscores
        filename = re.sub(r'[^\w\s.-]', '', filename)

        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]

        return f"{name}{ext}"

    def get_session_dir(self, session_id: str) -> Path:
        """Get directory for session files."""
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_file(
        self,
        contents: bytes,
        filename: str,
        session_id: str,
        dataset_id: str,
    ) -> Tuple[str, int]:
        """Save uploaded file to persistent storage.

        Args:
            contents: File contents
            filename: Original filename
            session_id: Session identifier
            dataset_id: Dataset identifier

        Returns:
            Tuple of (file_path, file_size)

        Raises:
            ValueError: If file too large or quota exceeded
        """
        # Check file size
        file_size = len(contents)
        max_bytes = self.max_upload_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {self.max_upload_size_mb}MB)"
            )

        # Check session quota
        session_dir = self.get_session_dir(session_id)
        session_usage = self.get_directory_size(session_dir)
        max_session_bytes = self.max_session_storage_mb * 1024 * 1024

        if session_usage + file_size > max_session_bytes:
            raise ValueError(
                f"Session storage quota exceeded: "
                f"{(session_usage + file_size) / 1024 / 1024:.1f}MB "
                f"(max: {self.max_session_storage_mb}MB)"
            )

        # Check total quota
        total_usage = self.get_directory_size(self.upload_dir)
        max_total_bytes = self.max_total_storage_gb * 1024 * 1024 * 1024

        if total_usage + file_size > max_total_bytes:
            raise ValueError(
                f"Total storage quota exceeded: "
                f"{(total_usage + file_size) / 1024 / 1024 / 1024:.1f}GB "
                f"(max: {self.max_total_storage_gb}GB)"
            )

        # Sanitize filename and add dataset_id
        safe_filename = self.sanitize_filename(filename)
        _, ext = os.path.splitext(safe_filename)
        final_filename = f"{dataset_id}{ext}"

        # Save file
        file_path = session_dir / final_filename
        file_path.write_bytes(contents)

        return str(file_path), file_size

    def get_file(self, dataset_id: str, session_id: str) -> Optional[Path]:
        """Retrieve file path for dataset.

        Args:
            dataset_id: Dataset identifier
            session_id: Session identifier

        Returns:
            Path to file if exists, None otherwise
        """
        session_dir = self.get_session_dir(session_id)

        # Try common extensions
        for ext in [".csv", ".parquet"]:
            file_path = session_dir / f"{dataset_id}{ext}"
            if file_path.exists():
                return file_path

        return None

    def delete_file(self, dataset_id: str, session_id: str) -> bool:
        """Delete dataset file.

        Args:
            dataset_id: Dataset identifier
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        file_path = self.get_file(dataset_id, session_id)
        if file_path and file_path.exists():
            file_path.unlink()
            return True
        return False

    def delete_session(self, session_id: str) -> Tuple[int, int]:
        """Delete all files for a session.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        session_dir = self.get_session_dir(session_id)

        if not session_dir.exists():
            return 0, 0

        # Count files and size
        files_deleted = len(list(session_dir.glob("*")))
        bytes_freed = self.get_directory_size(session_dir)

        # Remove directory
        shutil.rmtree(session_dir)

        return files_deleted, bytes_freed

    def get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory.

        Args:
            path: Directory path

        Returns:
            Total size in bytes
        """
        if not path.exists():
            return 0

        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total

    def get_storage_stats(self) -> dict:
        """Get storage usage statistics.

        Returns:
            Dictionary with storage stats
        """
        total_size = self.get_directory_size(self.upload_dir)
        total_limit = self.max_total_storage_gb * 1024 * 1024 * 1024

        sessions = list(self.upload_dir.glob("*"))
        active_sessions = len([s for s in sessions if s.is_dir()])

        datasets = []
        for session_dir in sessions:
            if session_dir.is_dir():
                datasets.extend(list(session_dir.glob("*")))

        return {
            "total_size_gb": total_size / 1024 / 1024 / 1024,
            "storage_limit_gb": self.max_total_storage_gb,
            "usage_percent": (total_size / total_limit * 100) if total_limit > 0 else 0,
            "active_sessions": active_sessions,
            "total_datasets": len(datasets),
        }


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        upload_dir = os.getenv("UPLOAD_DIR", "./data/uploads")
        _storage_manager = StorageManager(upload_dir)
    return _storage_manager
