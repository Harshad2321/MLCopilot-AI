"""
MLCopilot AI — ML Issue Detection Engine  (backend/analyzer.py)
================================================================
Rule-based analyzer that inspects epoch-level training metrics and
returns a list of detected issues with explanations and suggestions.

Detected issues
---------------
1. Overfitting         — train_loss falls while val_loss rises
2. Underfitting        — both losses stay high after many epochs
3. Exploding Gradients — gradient_norm exceeds threshold
4. Vanishing Gradients — gradient_norm is near zero
5. Loss Stagnation     — loss does not improve for N consecutive epochs
"""

from __future__ import annotations
from typing import Optional

# ── Thresholds (easy to tune) ─────────────────────────────────────────────────
GRAD_EXPLODING_THRESHOLD = 10.0
GRAD_VANISHING_THRESHOLD = 1e-7
OVERFIT_GAP_THRESHOLD    = 0.15   # val_loss - train_loss
STAGNATION_WINDOW        = 5      # epochs without improvement
STAGNATION_DELTA         = 0.005  # minimum meaningful improvement
UNDERFIT_ACCURACY_MAX    = 0.55   # below this = underfitting
UNDERFIT_MIN_EPOCHS      = 8      # only flag after this many epochs


# ── Public interface ──────────────────────────────────────────────────────────

def analyze_metrics(history: list[dict]) -> list[dict]:
    """
    Analyze the full metric history for a run and return a list of issues.

    Each issue dict has the shape:
    {
        "issue":       str,            # human-readable issue name
        "severity":    str,            # "low" | "medium" | "high" | "critical"
        "reason":      str,            # why it was flagged
        "suggestions": list[str],      # actionable fixes
        "epoch":       int,            # epoch where it was detected
    }
    """
    if not history:
        return []

    issues: list[dict] = []
    issues.extend(_check_exploding_gradients(history))
    issues.extend(_check_vanishing_gradients(history))
    issues.extend(_check_overfitting(history))
    issues.extend(_check_underfitting(history))
    issues.extend(_check_loss_stagnation(history))

    return issues


# ── Individual detectors ──────────────────────────────────────────────────────

def _check_exploding_gradients(history: list[dict]) -> list[dict]:
    latest = history[-1]
    grad = latest.get("gradient_norm")
    if grad is None or grad <= GRAD_EXPLODING_THRESHOLD:
        return []

    severity = "critical" if grad > GRAD_EXPLODING_THRESHOLD * 5 else "high"
    return [{
        "issue": "Exploding Gradients",
        "severity": severity,
        "reason": (
            f"Gradient norm is {grad:.4f}, which exceeds the threshold of "
            f"{GRAD_EXPLODING_THRESHOLD}. This destabilises training."
        ),
        "suggestions": [
            "Reduce learning rate (try 1e-4 or 3e-4).",
            "Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.",
            "Add Batch Normalisation layers.",
            "Use a learning-rate scheduler with warmup.",
        ],
        "epoch": latest.get("epoch"),
    }]


def _check_vanishing_gradients(history: list[dict]) -> list[dict]:
    latest = history[-1]
    grad = latest.get("gradient_norm")
    if grad is None or grad >= GRAD_VANISHING_THRESHOLD:
        return []

    return [{
        "issue": "Vanishing Gradients",
        "severity": "high",
        "reason": (
            f"Gradient norm is {grad:.2e}, which is near zero "
            f"(threshold {GRAD_VANISHING_THRESHOLD:.1e}). Weights are not updating."
        ),
        "suggestions": [
            "Replace sigmoid/tanh activations with ReLU or LeakyReLU.",
            "Add residual / skip connections.",
            "Use Xavier or Kaiming weight initialisation.",
            "Add Batch Normalisation to stabilise gradients.",
        ],
        "epoch": latest.get("epoch"),
    }]


def _check_overfitting(history: list[dict]) -> list[dict]:
    """Flag overfitting when val_loss > train_loss by a significant margin
    AND train_loss is trending downward while val_loss trends upward."""
    if len(history) < 5:
        return []

    recent = history[-5:]
    train_losses = [m["train_loss"] for m in recent if m.get("train_loss") is not None]
    val_losses   = [m["val_loss"]   for m in recent if m.get("val_loss")   is not None]

    if len(train_losses) < 3 or len(val_losses) < 3:
        return []

    train_trend = train_losses[-1] - train_losses[0]   # negative = improving
    val_trend   = val_losses[-1]   - val_losses[0]     # positive = worsening
    gap         = val_losses[-1]   - train_losses[-1]

    if train_trend < 0 and val_trend > 0 and gap > OVERFIT_GAP_THRESHOLD:
        return [{
            "issue": "Overfitting",
            "severity": "high",
            "reason": (
                f"Train loss is decreasing ({train_losses[0]:.4f} → {train_losses[-1]:.4f}) "
                f"while val loss is increasing ({val_losses[0]:.4f} → {val_losses[-1]:.4f}). "
                f"Gap = {gap:.4f}."
            ),
            "suggestions": [
                "Add Dropout (p=0.3 – 0.5) between fully-connected layers.",
                "Add weight decay / L2 regularisation (e.g., 1e-4).",
                "Use data augmentation to increase effective dataset size.",
                "Reduce model capacity (fewer layers or smaller hidden sizes).",
                "Implement early stopping based on validation loss.",
            ],
            "epoch": history[-1].get("epoch"),
        }]
    return []


def _check_underfitting(history: list[dict]) -> list[dict]:
    """Flag underfitting when accuracy stays below threshold after sufficient epochs."""
    if len(history) < UNDERFIT_MIN_EPOCHS:
        return []

    latest = history[-1]
    acc = latest.get("accuracy")
    if acc is None or acc > UNDERFIT_ACCURACY_MAX:
        return []

    return [{
        "issue": "Underfitting",
        "severity": "medium",
        "reason": (
            f"Accuracy is only {acc:.2%} after {latest.get('epoch')} epochs, "
            f"below the underfitting threshold of {UNDERFIT_ACCURACY_MAX:.0%}."
        ),
        "suggestions": [
            "Increase model capacity (more layers or wider hidden sizes).",
            "Increase learning rate (try 1e-3 or 3e-3).",
            "Train for more epochs.",
            "Improve feature engineering or input normalisation.",
            "Reduce regularisation if it is too aggressive.",
        ],
        "epoch": latest.get("epoch"),
    }]


def _check_loss_stagnation(history: list[dict]) -> list[dict]:
    """Flag stagnation when loss does not improve by STAGNATION_DELTA for N epochs."""
    if len(history) < STAGNATION_WINDOW:
        return []

    window = history[-STAGNATION_WINDOW:]
    losses = [m["train_loss"] for m in window if m.get("train_loss") is not None]

    if len(losses) < STAGNATION_WINDOW:
        return []

    improvement = losses[0] - losses[-1]
    if improvement < STAGNATION_DELTA:
        return [{
            "issue": "Loss Stagnation",
            "severity": "medium",
            "reason": (
                f"Training loss improved by only {improvement:.5f} over the last "
                f"{STAGNATION_WINDOW} epochs (threshold: {STAGNATION_DELTA}). "
                "The model may be stuck in a local minimum or saddle point."
            ),
            "suggestions": [
                "Apply learning-rate scheduling (ReduceLROnPlateau or cosine annealing).",
                "Try a different optimiser (e.g., AdamW or SGD with momentum).",
                "Add batch normalisation to smooth the loss landscape.",
                "Check for data issues (label noise, class imbalance).",
            ],
            "epoch": history[-1].get("epoch"),
        }]
    return []
