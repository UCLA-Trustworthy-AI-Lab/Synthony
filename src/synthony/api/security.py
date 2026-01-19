"""
Security utilities for audit logging and error handling.

Handles:
- Audit trail logging
- Error logging to error.log
- Request validation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Setup error logging
error_log_path = Path("./logs/error.log")
error_log_path.parent.mkdir(parents=True, exist_ok=True)

error_logger = logging.getLogger("synthony.errors")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(error_log_path)
error_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
error_logger.addHandler(error_handler)


def log_error(
    session_id: Optional[str],
    action: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Log error to error.log file.

    Args:
        session_id: Session identifier
        action: Action being performed
        error: Exception that occurred
        context: Additional context

    Returns:
        Error message for display
    """
    error_message = f"[{action}] {type(error).__name__}: {str(error)}"

    context_dict = {
        "session_id": session_id,
        "action": action,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if context:
        context_dict.update(context)

    error_logger.error(error_message, extra={"context": json.dumps(context_dict)})

    return error_message


def get_client_info(request) -> tuple:
    """Extract client IP and user agent from request.

    Args:
        request: FastAPI request object (can be None)

    Returns:
        Tuple of (ip_address, user_agent)
    """
    # Handle None request
    if request is None:
        return "unknown", "unknown"

    # Get IP address (handle proxies)
    ip_address = request.client.host if request and request.client else "unknown"

    # Check for forwarded IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip_address = forwarded.split(",")[0].strip()

    # Get user agent
    user_agent = request.headers.get("User-Agent", "unknown")

    return ip_address, user_agent
