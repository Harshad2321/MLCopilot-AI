"""
MLCopilot AI - FastAPI Main Application
Entry point for the backend server.
"""

import sys
import os

# Ensure project root is on the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.analyze_training import router as training_router
from backend.routes.suggestions import router as suggestions_router
from database.storage import init_db

# Initialize database
init_db()

app = FastAPI(
    title="MLCopilot AI",
    description=(
        "An AI assistant that monitors ML model training, detects issues, "
        "performs root-cause analysis, and suggests fixes."
    ),
    version="1.0.0",
)

# CORS (allow dashboard / external tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(training_router)
app.include_router(suggestions_router)


@app.get("/")
def root():
    return {
        "name": "MLCopilot AI",
        "version": "1.0.0",
        "description": "AI assistant for ML training debugging and optimization",
        "endpoints": {
            "POST /api/experiment": "Create a new experiment",
            "GET  /api/experiments": "List all experiments",
            "POST /api/metrics": "Submit training metrics",
            "POST /api/analyze": "Run full analysis on an experiment",
            "GET  /api/metrics/{id}": "Get metrics history",
            "GET  /api/analysis/{id}": "Get analysis history",
            "POST /api/suggest": "Get suggestions for a specific problem",
            "GET  /api/problems": "List diagnosable problems",
            "GET  /docs": "Interactive API documentation (Swagger)",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
