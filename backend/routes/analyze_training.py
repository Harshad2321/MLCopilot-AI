"""
MLCopilot AI - Training Analysis Route
Endpoint for submitting training metrics and receiving analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from backend.services.metrics_monitor import MetricsMonitor
from backend.services.anomaly_detector import AnomalyDetector
from backend.services.root_cause_engine import RootCauseEngine
from backend.services.suggestion_engine import SuggestionEngine
from database.storage import (
    create_experiment,
    log_metrics,
    get_metrics,
    get_recent_metrics,
    save_analysis,
    get_experiment,
    get_all_experiments,
    update_experiment_status,
    get_analysis_history,
)

router = APIRouter(prefix="/api", tags=["training"])

# ---------------------------------------------------------------------------
# Shared state: per-experiment monitors
# ---------------------------------------------------------------------------
monitors: dict[int, MetricsMonitor] = {}
detector = AnomalyDetector()
root_cause_engine = RootCauseEngine()
suggestion_engine = SuggestionEngine()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class CreateExperimentRequest(BaseModel):
    name: str
    config: dict = Field(default_factory=dict)


class MetricsPayload(BaseModel):
    experiment_id: int
    epoch: int
    step: int = 0
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    grad_norm: Optional[float] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    extra: dict = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    experiment_id: int
    window: int = 20  # how many recent epochs to analyze


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/experiment")
def create_new_experiment(req: CreateExperimentRequest):
    """Create a new experiment / training run."""
    exp_id = create_experiment(req.name, req.config)
    monitors[exp_id] = MetricsMonitor()
    return {"experiment_id": exp_id, "name": req.name, "status": "created"}


@router.get("/experiments")
def list_experiments():
    """List all experiments."""
    return get_all_experiments()


@router.get("/experiment/{experiment_id}")
def get_experiment_details(experiment_id: int):
    """Get details for a specific experiment."""
    exp = get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.post("/metrics")
def submit_metrics(payload: MetricsPayload):
    """
    Submit a single epoch/step of training metrics.
    Returns real-time analysis if issues are detected.
    """
    # Ensure monitor exists
    if payload.experiment_id not in monitors:
        monitors[payload.experiment_id] = MetricsMonitor()

    monitor = monitors[payload.experiment_id]

    # Build metrics dict
    metrics = {
        "epoch": payload.epoch,
        "step": payload.step,
        "loss": payload.loss,
        "val_loss": payload.val_loss,
        "accuracy": payload.accuracy,
        "val_accuracy": payload.val_accuracy,
        "grad_norm": payload.grad_norm,
        "lr": payload.lr,
        "batch_size": payload.batch_size,
        **payload.extra,
    }

    # Record in monitor
    enriched = monitor.record(metrics)

    # Store in database
    log_metrics(payload.experiment_id, payload.epoch, metrics, payload.step)

    # Run real-time analysis
    history = monitor.get_history()
    stats = enriched.get("stats", {})
    issues = detector.analyze(history, stats)

    response = {
        "status": "ok",
        "epoch": payload.epoch,
        "metrics_recorded": True,
        "issues_detected": len(issues),
    }

    if issues:
        issue_dicts = [i.to_dict() for i in issues]
        root_causes = root_cause_engine.analyze(issue_dicts)
        suggestions = suggestion_engine.generate(issue_dicts, root_causes)

        # Save analysis
        save_analysis(
            payload.experiment_id, payload.epoch, issue_dicts, root_causes, suggestions
        )

        response["issues"] = issue_dicts
        response["root_causes"] = root_causes
        response["suggestions"] = suggestions
        response["report"] = suggestion_engine.format_report(suggestions)

    return response


@router.post("/analyze")
def analyze_experiment(req: AnalyzeRequest):
    """
    Run a full analysis on an experiment's metrics history.
    Can be called at any time, even after training completes.
    """
    metrics_list = get_metrics(req.experiment_id)
    if not metrics_list:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for experiment {req.experiment_id}",
        )

    # Use the most recent window
    recent = metrics_list[-req.window :]

    # Build a temporary monitor to compute stats
    temp_monitor = MetricsMonitor(window_size=req.window)
    for m in recent:
        temp_monitor.record(m)

    stats = temp_monitor._compute_stats()
    issues = detector.analyze(recent, stats)

    if not issues:
        return {
            "experiment_id": req.experiment_id,
            "status": "healthy",
            "message": "No issues detected in the training metrics.",
            "epochs_analyzed": len(recent),
        }

    issue_dicts = [i.to_dict() for i in issues]
    root_causes = root_cause_engine.analyze(issue_dicts)
    suggestions = suggestion_engine.generate(issue_dicts, root_causes)

    save_analysis(
        req.experiment_id, recent[-1].get("epoch", 0), issue_dicts, root_causes, suggestions
    )

    report = suggestion_engine.format_report(suggestions)

    return {
        "experiment_id": req.experiment_id,
        "status": "issues_found",
        "epochs_analyzed": len(recent),
        "issues": issue_dicts,
        "root_causes": root_causes,
        "suggestions": suggestions,
        "report": report,
    }


@router.get("/metrics/{experiment_id}")
def get_experiment_metrics(experiment_id: int, last_n: Optional[int] = None):
    """Retrieve metrics for an experiment."""
    if last_n:
        return get_recent_metrics(experiment_id, last_n)
    return get_metrics(experiment_id)


@router.get("/analysis/{experiment_id}")
def get_experiment_analysis(experiment_id: int):
    """Get the analysis history for an experiment."""
    return get_analysis_history(experiment_id)


@router.post("/experiment/{experiment_id}/complete")
def complete_experiment(experiment_id: int):
    """Mark an experiment as completed."""
    update_experiment_status(experiment_id, "completed")
    # Cleanup monitor
    monitors.pop(experiment_id, None)
    return {"status": "completed", "experiment_id": experiment_id}
