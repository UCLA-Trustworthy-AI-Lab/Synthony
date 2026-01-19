# ============================================================
# Synthony - Production Dockerfile
# ============================================================
# Multi-stage build for optimized image size
#
# Build: docker build -t synthony:latest .
# Run:   docker run -p 8000:8000 synthony:latest
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api,llm]"

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies (for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY docs/SystemPrompt_v2.md ./docs/SystemPrompt_v2.md

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables (defaults)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=2 \
    SYNTHONY_SYSTEM_PROMPT=/app/docs/SystemPrompt_v2.md \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["python", "-m", "uvicorn", "synthony.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
