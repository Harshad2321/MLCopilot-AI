# MLCopilot AI — Hugging Face Spaces Dockerfile
# Runs both the FastAPI backend (port 8000, internal) and the
# Streamlit dashboard (port 7860, exposed) in a single container.
#
# HF Spaces will automatically expose port 7860.
# Local Docker usage:
#   docker build -t mlcopilot-hf .
#   docker run -p 7860:7860 [-e OPENAI_API_KEY=...] mlcopilot-hf

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevents .pyc files and enables real-time log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# HF Spaces requires a non-root user with uid 1000
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────────────────────
RUN useradd -m -u 1000 user

USER user
WORKDIR /home/user/app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch first (keeps the image lean — no CUDA needed on HF)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        requests \
        pandas \
        openai \
        anthropic \
        python-dotenv \
        streamlit \
        plotly \
        optuna

# ── Application code ──────────────────────────────────────────────────────────
COPY --chown=user . .

# ── Persistent database directory ─────────────────────────────────────────────
RUN mkdir -p /home/user/app/database

# ── Startup script ────────────────────────────────────────────────────────────
RUN chmod +x /home/user/app/start.sh

# ── Expose Streamlit port (HF Spaces default) ─────────────────────────────────
EXPOSE 7860

# ── Launch both services ──────────────────────────────────────────────────────
CMD ["/home/user/app/start.sh"]
