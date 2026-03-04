"""
MLCopilot AI - Anomaly Detection Engine
Detects common ML training problems using rule-based + statistical methods.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Issue:
    """Represents a detected training issue."""
    name: str
    severity: str          # "low", "medium", "high", "critical"
    description: str
    evidence: dict = field(default_factory=dict)
    epoch: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "epoch": self.epoch,
        }


# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "grad_norm_exploding": 10.0,
    "grad_norm_vanishing": 1e-7,
    "overfit_gap": 0.15,              # val_loss - train_loss gap
    "loss_stagnation_epochs": 5,
    "loss_stagnation_delta": 0.005,
    "lr_too_high_loss_spike": 1.5,    # multiplied by previous loss
    "accuracy_underfit": 0.55,        # below this after N epochs → underfitting
    "underfit_min_epochs": 8,
}


class AnomalyDetector:
    """
    Rule-based + statistical anomaly detection for training metrics.
    Analyzes a sequence of metric records and returns detected issues.
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}

    def analyze(self, metrics_history: list[dict], stats: dict) -> list[Issue]:
        """
        Run all detection rules on the given metric history.
        Returns a list of Issues found.
        """
        issues: list[Issue] = []

        if not metrics_history:
            return issues

        latest = metrics_history[-1]
        epoch = latest.get("epoch", len(metrics_history))

        issues.extend(self._check_exploding_gradients(latest, epoch))
        issues.extend(self._check_vanishing_gradients(latest, epoch))
        issues.extend(self._check_overfitting(metrics_history, latest, epoch))
        issues.extend(self._check_underfitting(metrics_history, latest, epoch))
        issues.extend(self._check_loss_stagnation(metrics_history, epoch))
        issues.extend(self._check_lr_too_high(metrics_history, latest, epoch))
        issues.extend(self._check_loss_divergence(metrics_history, latest, epoch))
        issues.extend(self._check_nan_inf(latest, epoch))

        return issues

    # ---------- Individual Detectors ----------

    def _check_exploding_gradients(self, latest: dict, epoch: int) -> list[Issue]:
        grad = latest.get("grad_norm")
        if grad is not None and grad > self.thresholds["grad_norm_exploding"]:
            return [Issue(
                name="Exploding Gradients",
                severity="critical" if grad > self.thresholds["grad_norm_exploding"] * 5 else "high",
                description=(
                    f"Gradient norm is {grad:.4f}, exceeding threshold "
                    f"{self.thresholds['grad_norm_exploding']}."
                ),
                evidence={"grad_norm": grad, "threshold": self.thresholds["grad_norm_exploding"]},
                epoch=epoch,
            )]
        return []

    def _check_vanishing_gradients(self, latest: dict, epoch: int) -> list[Issue]:
        grad = latest.get("grad_norm")
        if grad is not None and grad < self.thresholds["grad_norm_vanishing"]:
            return [Issue(
                name="Vanishing Gradients",
                severity="high",
                description=(
                    f"Gradient norm is {grad:.2e}, below threshold "
                    f"{self.thresholds['grad_norm_vanishing']:.1e}."
                ),
                evidence={"grad_norm": grad, "threshold": self.thresholds["grad_norm_vanishing"]},
                epoch=epoch,
            )]
        return []

    def _check_overfitting(
        self, history: list[dict], latest: dict, epoch: int
    ) -> list[Issue]:
        if len(history) < 3:
            return []
        train_losses = [m["loss"] for m in history[-5:] if "loss" in m and m["loss"] is not None]
        val_losses = [m["val_loss"] for m in history[-5:] if "val_loss" in m and m["val_loss"] is not None]

        if len(train_losses) < 3 or len(val_losses) < 3:
            return []

        train_trend = train_losses[-1] - train_losses[0]
        val_trend = val_losses[-1] - val_losses[0]
        gap = (latest.get("val_loss") or 0) - (latest.get("loss") or 0)

        if train_trend < 0 and val_trend > 0:
            return [Issue(
                name="Overfitting",
                severity="high",
                description=(
                    "Validation loss is increasing while training loss is decreasing. "
                    f"Gap: {gap:.4f}."
                ),
                evidence={
                    "train_loss_trend": round(train_trend, 5),
                    "val_loss_trend": round(val_trend, 5),
                    "gap": round(gap, 5),
                },
                epoch=epoch,
            )]

        if gap > self.thresholds["overfit_gap"]:
            return [Issue(
                name="Overfitting",
                severity="medium",
                description=(
                    f"Train-val loss gap is {gap:.4f}, exceeding threshold "
                    f"{self.thresholds['overfit_gap']}."
                ),
                evidence={"gap": round(gap, 5)},
                epoch=epoch,
            )]

        return []

    def _check_underfitting(
        self, history: list[dict], latest: dict, epoch: int
    ) -> list[Issue]:
        min_epochs = self.thresholds["underfit_min_epochs"]
        if len(history) < min_epochs:
            return []

        acc = latest.get("accuracy")
        if acc is not None and acc < self.thresholds["accuracy_underfit"]:
            return [Issue(
                name="Underfitting",
                severity="medium",
                description=(
                    f"Accuracy is only {acc:.2%} after {epoch} epochs, "
                    f"below threshold {self.thresholds['accuracy_underfit']:.0%}."
                ),
                evidence={"accuracy": acc, "epoch": epoch},
                epoch=epoch,
            )]

        # Also detect if both losses are high and not improving
        losses = [m["loss"] for m in history[-min_epochs:] if "loss" in m and m["loss"] is not None]
        val_losses = [m["val_loss"] for m in history[-min_epochs:] if "val_loss" in m and m["val_loss"] is not None]
        if losses and val_losses:
            if losses[-1] > 1.0 and val_losses[-1] > 1.0:
                loss_delta = abs(losses[-1] - losses[0])
                if loss_delta < self.thresholds["loss_stagnation_delta"] * 2:
                    return [Issue(
                        name="Underfitting",
                        severity="medium",
                        description=(
                            f"Both training loss ({losses[-1]:.4f}) and validation loss "
                            f"({val_losses[-1]:.4f}) remain high with little improvement."
                        ),
                        evidence={"loss": losses[-1], "val_loss": val_losses[-1]},
                        epoch=epoch,
                    )]
        return []

    def _check_loss_stagnation(self, history: list[dict], epoch: int) -> list[Issue]:
        n = self.thresholds["loss_stagnation_epochs"]
        if len(history) < n:
            return []
        recent_losses = [m["loss"] for m in history[-n:] if "loss" in m and m["loss"] is not None]
        if len(recent_losses) < n:
            return []

        max_delta = max(recent_losses) - min(recent_losses)
        if max_delta < self.thresholds["loss_stagnation_delta"]:
            return [Issue(
                name="Loss Stagnation",
                severity="medium",
                description=(
                    f"Loss has barely changed (delta={max_delta:.6f}) over the last "
                    f"{n} epochs. Possible learning plateau."
                ),
                evidence={"delta": max_delta, "window": n},
                epoch=epoch,
            )]
        return []

    def _check_lr_too_high(
        self, history: list[dict], latest: dict, epoch: int
    ) -> list[Issue]:
        if len(history) < 2:
            return []
        prev = history[-2]
        curr_loss = latest.get("loss")
        prev_loss = prev.get("loss")
        if curr_loss is None or prev_loss is None or prev_loss == 0:
            return []

        spike_ratio = curr_loss / prev_loss
        if spike_ratio > self.thresholds["lr_too_high_loss_spike"]:
            return [Issue(
                name="Learning Rate Too High",
                severity="high",
                description=(
                    f"Loss spiked from {prev_loss:.4f} to {curr_loss:.4f} "
                    f"(ratio: {spike_ratio:.2f}x). Learning rate may be too high."
                ),
                evidence={
                    "prev_loss": round(prev_loss, 5),
                    "curr_loss": round(curr_loss, 5),
                    "spike_ratio": round(spike_ratio, 3),
                    "lr": latest.get("lr"),
                },
                epoch=epoch,
            )]
        return []

    def _check_loss_divergence(
        self, history: list[dict], latest: dict, epoch: int
    ) -> list[Issue]:
        loss = latest.get("loss")
        if loss is not None and (loss > 100 or loss < -100):
            return [Issue(
                name="Loss Divergence",
                severity="critical",
                description=f"Loss value is {loss:.4f}, indicating training has diverged.",
                evidence={"loss": loss},
                epoch=epoch,
            )]
        return []

    def _check_nan_inf(self, latest: dict, epoch: int) -> list[Issue]:
        issues = []
        import math
        for key in ("loss", "val_loss", "grad_norm"):
            val = latest.get(key)
            if val is not None and (math.isnan(val) or math.isinf(val)):
                issues.append(Issue(
                    name="NaN/Inf Detected",
                    severity="critical",
                    description=f"{key} is {val}. Training has become numerically unstable.",
                    evidence={"metric": key, "value": str(val)},
                    epoch=epoch,
                ))
        return issues
