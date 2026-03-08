# MLCopilot AI — Dockerfile
# Builds the FastAPI backend as a lightweight container
#
# Build:  docker build -t mlcopilot-api .
# Run:    docker run -p 8000:8000 --env-file .env mlcopilot-api

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevents .pyc files and enables real-time log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
# Install without PyTorch (not needed on the server — only the training machine needs it)
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        requests \
        pandas \
        openai \
        anthropic \
        boto3 \
        psycopg2-binary \
        python-dotenv

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/   ./backend/
COPY database/  ./database/
COPY sdk/       ./sdk/

# ── Database directory (used by SQLite) ───────────────────────────────────────
RUN mkdir -p /app/database

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
