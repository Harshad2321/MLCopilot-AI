"""
MLCopilot AI — Backend API  (backend/main.py)
==============================================
FastAPI entry-point for the hackathon prototype.

Endpoints
---------
POST /metrics          — receive training metrics from the local training script
GET  /metrics          — return all logged metrics (used by dashboard)
GET  /analysis         — return detected issues and LLM-enhanced suggestions
POST /experiment       — create a named experiment / run
GET  /health           — liveness probe

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os

# Make the project root importable from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from backend.database import init_db, insert_metrics, fetch_all_metrics, fetch_run_metrics
from backend.analyzer import analyze_metrics
from backend.llm_engine import generate_explanation

# ── Initialise SQLite on startup ──────────────────────────────────────────────
init_db()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLCopilot AI",
    description=(
        "Real-time ML training monitor: detects anomalies, explains root causes, "
        "and suggests fixes via a REST API."
    ),
    version="2.0.0",
)

# Allow the Streamlit dashboard (and any other origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class MetricsPayload(BaseModel):
    """Metrics sent from the training script each epoch."""
    run_id: str = Field(..., description="Unique identifier for this training run")
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None


class ExperimentRequest(BaseModel):
    run_id: str
    name: str
    config: dict = Field(default_factory=dict)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — returns 200 if the server is up."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "service": "MLCopilot AI",
        "docs": "/docs",
        "endpoints": {
            "POST /metrics": "Submit epoch metrics from your training script",
            "GET  /metrics": "Retrieve all logged metrics",
            "GET  /analysis": "Get detected issues and fix suggestions",
            "GET  /health": "Server health check",
        },
    }


@app.post("/metrics", status_code=201)
def receive_metrics(payload: MetricsPayload):
    """
    Called by the MLCopilot SDK after every epoch.
    Stores the metrics and immediately runs issue detection.
    Returns any issues found so the SDK can print them locally too.
    """
    # Persist to SQLite
    insert_metrics(
        run_id=payload.run_id,
        epoch=payload.epoch,
        train_loss=payload.train_loss,
        val_loss=payload.val_loss,
        accuracy=payload.accuracy,
        learning_rate=payload.learning_rate,
        gradient_norm=payload.gradient_norm,
    )

    # Run real-time issue detection on the latest window for this run
    history = fetch_run_metrics(payload.run_id)
    issues = analyze_metrics(history)

    return {
        "status": "logged",
        "epoch": payload.epoch,
        "issues_detected": len(issues),
        "issues": issues,
    }


@app.get("/metrics")
def get_metrics(run_id: Optional[str] = None):
    """
    Return logged metrics.
    - ?run_id=<id>  — filter to a specific run
    - (no param)    — return all metrics
    """
    if run_id:
        rows = fetch_run_metrics(run_id)
    else:
        rows = fetch_all_metrics()
    return {"count": len(rows), "metrics": rows}


@app.get("/analysis")
def get_analysis(run_id: str):
    """
    Full analysis for a completed (or in-progress) training run.
    1. Fetches all metrics for the run.
    2. Runs the rule-based issue detector.
    3. For each issue, calls the LLM engine for a human-readable explanation.
    Returns a list of detected issues with explanations and fix suggestions.
    """
    history = fetch_run_metrics(run_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No metrics found for run_id='{run_id}'")

    issues = analyze_metrics(history)

    # Enrich each issue with an LLM explanation (no-op if no API key is set)
    enriched = []
    for issue in issues:
        explanation = generate_explanation(issue, history[-1])
        enriched.append({**issue, "llm_explanation": explanation})

    return {
        "run_id": run_id,
        "epochs_analyzed": len(history),
        "total_issues": len(enriched),
        "results": enriched,
    }
