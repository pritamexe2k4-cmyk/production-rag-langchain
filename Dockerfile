# SOURCE: lang-production-api Dockerfile (3 GLUE edits marked below)
FROM python:3.12-slim

WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# Install uv (fast Python package manager)
RUN pip install uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# GLUE (not from course): writable home for the Chroma vector DB (so the compose volume is owned by appuser)
RUN mkdir -p /app/chroma_db

# Install dependencies (production only)
RUN uv sync --frozen --no-dev

# Copy application code
COPY --chown=appuser:appuser app/ app/
# GLUE (not from course): the corpus that /ingest reads
COPY --chown=appuser:appuser data/ data/

# Expose port
EXPOSE 8000

# Health check
# GLUE (not from course): python:3.12-slim has no curl, so the course's `curl -f` check
# always failed; same check done with python's stdlib instead.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Run uvicorn directly from venv (avoids uv re-syncing at runtime)
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]