"""
MLCopilot AI - Root Cause Analysis Engine
Infers most likely causes for detected training issues.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Knowledge base: maps issue names → possible root causes
# Each root cause has a description, confidence, and conditions.
# ---------------------------------------------------------------------------
ROOT_CAUSE_DB = {
    "Exploding Gradients": [
        {
            "cause": "Learning rate too high",
            "description": "A large learning rate amplifies gradient updates, causing instability.",
            "confidence": 0.88,
            "condition": lambda evidence: evidence.get("lr", 0) > 0.005 if evidence.get("lr") else True,
        },
        {
            "cause": "Missing gradient clipping",
            "description": "Without gradient clipping, large gradients propagate unchecked.",
            "confidence": 0.80,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Improper weight initialization",
            "description": "Weights initialized with too-large values amplify activations and gradients.",
            "confidence": 0.60,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Network too deep without residual connections",
            "description": "Very deep networks without skip connections can accumulate gradient magnitudes.",
            "confidence": 0.55,
            "condition": lambda evidence: True,
        },
    ],
    "Vanishing Gradients": [
        {
            "cause": "Sigmoid/tanh activations in deep network",
            "description": "Saturating activation functions squash gradients toward zero in deep networks.",
            "confidence": 0.82,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Network too deep without residual connections",
            "description": "Gradients diminish exponentially through many layers.",
            "confidence": 0.78,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Poor weight initialization",
            "description": "Weights initialized too small cause activations and gradients to shrink.",
            "confidence": 0.70,
            "condition": lambda evidence: True,
        },
    ],
    "Overfitting": [
        {
            "cause": "Model too complex relative to dataset size",
            "description": "The model has too many parameters for the amount of training data.",
            "confidence": 0.85,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Insufficient regularization",
            "description": "No or inadequate dropout, weight decay, or data augmentation.",
            "confidence": 0.82,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Training for too many epochs",
            "description": "The model memorizes training data when trained too long without early stopping.",
            "confidence": 0.75,
            "condition": lambda evidence: evidence.get("epoch", 0) > 15,
        },
        {
            "cause": "Dataset too small",
            "description": "Insufficient training samples lead to poor generalization.",
            "confidence": 0.72,
            "condition": lambda evidence: True,
        },
    ],
    "Underfitting": [
        {
            "cause": "Model too simple",
            "description": "The model lacks capacity to learn the data distribution.",
            "confidence": 0.80,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Learning rate too low",
            "description": "Very small learning rate slows convergence, preventing the model from learning.",
            "confidence": 0.75,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Insufficient training epochs",
            "description": "The model hasn't been trained long enough to converge.",
            "confidence": 0.65,
            "condition": lambda evidence: evidence.get("epoch", 0) < 20,
        },
        {
            "cause": "Feature engineering issues",
            "description": "Input features may not be informative enough for the task.",
            "confidence": 0.60,
            "condition": lambda evidence: True,
        },
    ],
    "Loss Stagnation": [
        {
            "cause": "Stuck in local minimum or saddle point",
            "description": "The optimizer is trapped and cannot find a better solution.",
            "confidence": 0.78,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Learning rate too low",
            "description": "A very small learning rate causes extremely slow or no progress.",
            "confidence": 0.75,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Bad weight initialization",
            "description": "Poor initialization can place the model in a flat region of the loss surface.",
            "confidence": 0.60,
            "condition": lambda evidence: True,
        },
    ],
    "Learning Rate Too High": [
        {
            "cause": "Misconfigured learning rate",
            "description": "The learning rate was set too aggressively for this model/data combination.",
            "confidence": 0.92,
            "condition": lambda evidence: True,
        },
        {
            "cause": "No learning rate warmup",
            "description": "Starting with a large LR without warmup causes early instability.",
            "confidence": 0.70,
            "condition": lambda evidence: evidence.get("epoch", 0) < 5,
        },
    ],
    "Loss Divergence": [
        {
            "cause": "Learning rate too high",
            "description": "The learning rate is so large that optimization diverges.",
            "confidence": 0.90,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Numerical instability (no mixed-precision safeguards)",
            "description": "Floating point overflow leading to loss divergence.",
            "confidence": 0.70,
            "condition": lambda evidence: True,
        },
    ],
    "NaN/Inf Detected": [
        {
            "cause": "Numerical overflow in forward pass",
            "description": "Extremely large values cause NaN/Inf in computation.",
            "confidence": 0.85,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Learning rate too high",
            "description": "Very large updates lead to numerical instability.",
            "confidence": 0.80,
            "condition": lambda evidence: True,
        },
        {
            "cause": "Division by zero or log of zero",
            "description": "Certain loss functions can produce NaN for edge-case inputs.",
            "confidence": 0.65,
            "condition": lambda evidence: True,
        },
    ],
}


class RootCauseEngine:
    """
    After anomaly detection, infers the most likely root cause(s).
    Uses a rule-based knowledge base with conditional confidence scoring.
    """

    def __init__(self):
        self.knowledge_base = ROOT_CAUSE_DB

    def analyze(self, issues: list[dict]) -> list[dict]:
        """
        Given a list of detected issues, return root-cause analyses.
        Each result includes the problem, likely cause, confidence, and description.
        """
        results = []

        for issue in issues:
            name = issue.get("name", "")
            evidence = issue.get("evidence", {})
            evidence["epoch"] = issue.get("epoch")

            causes = self.knowledge_base.get(name, [])
            matched_causes = []

            for c in causes:
                try:
                    if c["condition"](evidence):
                        matched_causes.append({
                            "problem": name,
                            "cause": c["cause"],
                            "description": c["description"],
                            "confidence": round(c["confidence"], 2),
                        })
                except Exception:
                    # Skip if condition fails
                    matched_causes.append({
                        "problem": name,
                        "cause": c["cause"],
                        "description": c["description"],
                        "confidence": round(c["confidence"], 2),
                    })

            # Sort by confidence (highest first), take top 3
            matched_causes.sort(key=lambda x: x["confidence"], reverse=True)
            results.extend(matched_causes[:3])

        return results
