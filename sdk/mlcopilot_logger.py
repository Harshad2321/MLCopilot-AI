"""
MLCopilot AI — SDK Logger  (sdk/mlcopilot_logger.py)
=====================================================
Drop-in logger that any training script can use to send metrics
to the MLCopilot backend for real-time monitoring.

Minimal integration
-------------------
    from sdk.mlcopilot_logger import MLCopilotLogger

    logger = MLCopilotLogger(run_id="my_run", api_url="http://localhost:8000")

    for epoch in range(num_epochs):
        # ... training code ...
        logger.log(
            epoch       = epoch,
            train_loss  = train_loss,
            val_loss    = val_loss,
            accuracy    = accuracy,
            learning_rate = lr,
            gradient_norm = grad_norm,
        )

    logger.finish()

The logger prints any issues detected by the backend directly to stdout
so the developer sees feedback without opening the dashboard.
"""

from __future__ import annotations
import uuid
import time
import requests
from typing import Optional


# ── Public convenience entry-point (matches the prompt spec) ─────────────────
_default_logger: Optional["MLCopilotLogger"] = None


def start_monitoring(
    run_id: str = None,
    api_url: str = "http://localhost:8000",
    experiment_name: str = "",
) -> "MLCopilotLogger":
    """
    One-liner initialisation.

        import sdk.mlcopilot_logger as mlcopilot
        mlcopilot.start_monitoring()

    Returns the logger instance so callers can also call .log() explicitly.
    """
    global _default_logger
    _default_logger = MLCopilotLogger(
        run_id=run_id or _short_uuid(),
        api_url=api_url,
        experiment_name=experiment_name,
    )
    return _default_logger


def log(**kwargs):
    """Log metrics using the default (most recently started) logger."""
    if _default_logger is None:
        raise RuntimeError("Call mlcopilot.start_monitoring() before mlcopilot.log()")
    return _default_logger.log(**kwargs)


# ── Core logger class ─────────────────────────────────────────────────────────

class MLCopilotLogger:
    """
    Sends epoch metrics to the MLCopilot backend via REST.
    Falls back to offline (local) mode if the server is unreachable.
    """

    def __init__(
        self,
        run_id: str = None,
        api_url: str = "http://localhost:8000",
        experiment_name: str = "",
        offline: bool = False,
    ):
        self.run_id          = run_id or _short_uuid()
        self.api_url         = api_url.rstrip("/")
        self.experiment_name = experiment_name or self.run_id
        self.offline         = offline
        self._local_history: list[dict] = []

        if not self.offline:
            self._check_server()

        print(f"[MLCopilot] Monitoring started — run_id={self.run_id}")
        if not self.offline:
            print(f"[MLCopilot] Dashboard → http://localhost:8501")
            print(f"[MLCopilot] API       → {self.api_url}/docs")

    # ── Core log call ─────────────────────────────────────────────────────────

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
    ) -> list[dict]:
        """
        Log metrics for one epoch.

        Returns the list of issues detected by the backend
        (empty list when offline or no issues found).
        """
        payload = {
            "run_id":         self.run_id,
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "accuracy":       accuracy,
            "learning_rate":  learning_rate,
            "gradient_norm":  gradient_norm,
        }

        # Always keep a local copy for offline analysis
        self._local_history.append(payload)

        # Pretty console print
        parts = [f"Epoch {epoch:3d}", f"loss={train_loss:.4f}"]
        if val_loss       is not None: parts.append(f"val_loss={val_loss:.4f}")
        if accuracy       is not None: parts.append(f"acc={accuracy:.4f}")
        if gradient_norm  is not None: parts.append(f"grad={gradient_norm:.4f}")
        if learning_rate  is not None: parts.append(f"lr={learning_rate:.6f}")
        print(f"[MLCopilot] {' | '.join(parts)}")

        if self.offline:
            return []

        # Send to backend
        try:
            resp = requests.post(
                f"{self.api_url}/metrics",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            # Print any issues detected in real time
            for issue in data.get("issues", []):
                sev = issue.get("severity", "").upper()
                print(
                    f"[MLCopilot] ⚠  {sev}: {issue['issue']} — "
                    f"{issue.get('reason', '')[:80]}"
                )
            return data.get("issues", [])

        except requests.exceptions.ConnectionError:
            # Server went away mid-run — switch to offline silently
            self.offline = True
            print("[MLCopilot] Warning: lost connection to backend, switching to offline mode.")
            return []
        except Exception as e:
            print(f"[MLCopilot] Warning: could not log metrics ({e})")
            return []

    # ── Finish ────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        """Call at the end of training to retrieve the full analysis report."""
        print(f"\n[MLCopilot] Training complete — {len(self._local_history)} epochs logged.")
        if not self.offline:
            print(f"[MLCopilot] Full analysis → {self.api_url}/analysis?run_id={self.run_id}")
            print(f"[MLCopilot] Dashboard    → http://localhost:8501")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_server(self) -> None:
        """Check connectivity once at startup; switch to offline if unreachable."""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=3)
            if resp.status_code != 200:
                raise ConnectionError("Non-200 health check")
        except Exception:
            print(
                f"[MLCopilot] Warning: backend not reachable at {self.api_url}. "
                "Running in offline mode."
            )
            self.offline = True


def _short_uuid() -> str:
    return str(uuid.uuid4())[:8]
