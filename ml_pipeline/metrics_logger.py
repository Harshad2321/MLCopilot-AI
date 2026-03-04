"""
MLCopilot AI - Metrics Logger
Captures training signals during model training and sends them to the backend API.
Can also work in offline mode (no server required).
"""

import requests
import json
import time
from typing import Optional


class MetricsLogger:
    """
    Drop-in metrics logger for PyTorch training loops.
    Sends metrics to the MLCopilot backend and prints analysis results.
    """

    def __init__(
        self,
        experiment_name: str,
        api_url: str = "http://localhost:8000",
        config: dict = None,
        offline: bool = False,
    ):
        self.api_url = api_url.rstrip("/")
        self.experiment_name = experiment_name
        self.config = config or {}
        self.offline = offline
        self.experiment_id: Optional[int] = None
        self.logs: list[dict] = []
        self.issues_found: list[dict] = []

        if not offline:
            self._create_experiment()

    def _create_experiment(self):
        """Register a new experiment with the backend."""
        try:
            resp = requests.post(
                f"{self.api_url}/api/experiment",
                json={"name": self.experiment_name, "config": self.config},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            self.experiment_id = data["experiment_id"]
            print(f"[MLCopilot] Experiment created: #{self.experiment_id} - {self.experiment_name}")
        except Exception as e:
            print(f"[MLCopilot] Warning: Could not connect to backend ({e}). Switching to offline mode.")
            self.offline = True

    def log(
        self,
        epoch: int,
        loss: float = None,
        val_loss: float = None,
        accuracy: float = None,
        val_accuracy: float = None,
        grad_norm: float = None,
        lr: float = None,
        batch_size: int = None,
        step: int = 0,
        **extra,
    ) -> dict:
        """
        Log one set of training metrics.
        Returns analysis results (if any issues detected).
        """
        metrics = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "grad_norm": grad_norm,
            "lr": lr,
            "batch_size": batch_size,
            **extra,
        }

        self.logs.append(metrics)

        # Print compact metric line
        parts = [f"Epoch {epoch:3d}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if accuracy is not None:
            parts.append(f"acc={accuracy:.4f}")
        if grad_norm is not None:
            parts.append(f"grad={grad_norm:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.6f}")
        print(f"[MLCopilot] {' | '.join(parts)}")

        if self.offline:
            return {"status": "offline", "metrics_recorded": True}

        # Send to backend
        try:
            payload = {
                "experiment_id": self.experiment_id,
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "val_accuracy": val_accuracy,
                "grad_norm": grad_norm,
                "lr": lr,
                "batch_size": batch_size,
                "extra": extra,
            }
            resp = requests.post(
                f"{self.api_url}/api/metrics",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            result = resp.json()

            # Print real-time alerts
            if result.get("issues_detected", 0) > 0:
                self.issues_found.extend(result.get("issues", []))
                print(f"\n{'='*50}")
                print(f"[!] MLCopilot detected {result['issues_detected']} issue(s)!")
                if "report" in result:
                    print(result["report"])
                print(f"{'='*50}\n")

            return result

        except Exception as e:
            print(f"[MLCopilot] Warning: Could not send metrics ({e})")
            return {"status": "error", "error": str(e)}

    def analyze(self, window: int = 20) -> dict:
        """Request a full analysis of the experiment."""
        if self.offline:
            print("[MLCopilot] Analysis not available in offline mode.")
            return {}

        try:
            resp = requests.post(
                f"{self.api_url}/api/analyze",
                json={"experiment_id": self.experiment_id, "window": window},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            if "report" in result:
                print(result["report"])

            return result

        except Exception as e:
            print(f"[MLCopilot] Error during analysis: {e}")
            return {"error": str(e)}

    def complete(self):
        """Mark the experiment as completed."""
        if self.offline or not self.experiment_id:
            return

        try:
            requests.post(
                f"{self.api_url}/api/experiment/{self.experiment_id}/complete",
                timeout=5,
            )
            print(f"[MLCopilot] Experiment #{self.experiment_id} marked as completed.")
        except Exception:
            pass

    def get_summary(self) -> dict:
        """Return a summary of the training run."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.logs),
            "issues_found": len(self.issues_found),
            "offline_mode": self.offline,
        }
