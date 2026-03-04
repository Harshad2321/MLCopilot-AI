"""
MLCopilot AI - Suggestions Route
Endpoint for getting AI suggestions and hyperparameter recommendations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from backend.services.suggestion_engine import SuggestionEngine

router = APIRouter(prefix="/api", tags=["suggestions"])

suggestion_engine = SuggestionEngine()


class DirectQueryRequest(BaseModel):
    """For getting suggestions about a specific problem without full analysis."""
    problem: str
    severity: str = "medium"
    context: dict = Field(default_factory=dict)


@router.post("/suggest")
def get_suggestion(req: DirectQueryRequest):
    """
    Get AI suggestions for a specific training problem.
    Useful for quick lookups without running full analysis.
    """
    # Create a synthetic issue
    issue = {
        "name": req.problem,
        "severity": req.severity,
        "description": f"User-reported issue: {req.problem}",
        "evidence": req.context,
    }

    # Import root cause engine
    from backend.services.root_cause_engine import RootCauseEngine
    rce = RootCauseEngine()

    root_causes = rce.analyze([issue])
    suggestions = suggestion_engine.generate([issue], root_causes)

    if not suggestions:
        raise HTTPException(
            status_code=404,
            detail=f"No suggestions available for problem: {req.problem}",
        )

    report = suggestion_engine.format_report(suggestions)
    return {
        "suggestions": suggestions,
        "report": report,
    }


@router.get("/problems")
def list_known_problems():
    """List all problems the system can diagnose."""
    return {
        "problems": [
            {
                "name": "Exploding Gradients",
                "description": "Gradient norms grow uncontrollably during training.",
            },
            {
                "name": "Vanishing Gradients",
                "description": "Gradients become too small for effective weight updates.",
            },
            {
                "name": "Overfitting",
                "description": "Model performs well on training data but poorly on validation.",
            },
            {
                "name": "Underfitting",
                "description": "Model fails to learn the underlying patterns in the data.",
            },
            {
                "name": "Loss Stagnation",
                "description": "Training loss stops decreasing, indicating a plateau.",
            },
            {
                "name": "Learning Rate Too High",
                "description": "Learning rate causes unstable or divergent training.",
            },
            {
                "name": "Loss Divergence",
                "description": "Loss grows uncontrollably, training has failed.",
            },
            {
                "name": "NaN/Inf Detected",
                "description": "Numerical instability producing NaN or Inf values.",
            },
        ]
    }
