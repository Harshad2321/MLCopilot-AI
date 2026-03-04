"""
MLCopilot AI - Metrics Monitor Service
Tracks and processes training metrics in real time.
"""

import statistics
from typing import Optional


class MetricsMonitor:
    """
    Continuously tracks metrics during model training.
    Maintains a sliding window of recent metrics for analysis.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: list[dict] = []

    def record(self, metrics: dict) -> dict:
        """
        Record a new set of training metrics and return enriched metrics
        with computed statistics.
        """
        self.history.append(metrics)
        # Keep only the last N entries in the sliding window
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2 :]

        enriched = {**metrics}
        enriched["metrics_count"] = len(self.history)
        enriched["stats"] = self._compute_stats()
        return enriched

    def _compute_stats(self) -> dict:
        """Compute rolling statistics over the metric window."""
        recent = self.history[-self.window_size :]
        stats = {}

        for key in ("loss", "val_loss", "grad_norm", "accuracy", "val_accuracy"):
            values = [m[key] for m in recent if key in m and m[key] is not None]
            if len(values) >= 2:
                stats[key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._compute_trend(values),
                    "last": values[-1],
                }
            elif len(values) == 1:
                stats[key] = {
                    "mean": values[0],
                    "std": 0.0,
                    "min": values[0],
                    "max": values[0],
                    "trend": 0.0,
                    "last": values[0],
                }
        return stats

    @staticmethod
    def _compute_trend(values: list[float]) -> float:
        """
        Simple linear trend: positive = increasing, negative = decreasing.
        Returns average change per step over the window.
        """
        if len(values) < 2:
            return 0.0
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return statistics.mean(diffs)

    def get_latest(self) -> Optional[dict]:
        """Return the latest recorded metrics."""
        return self.history[-1] if self.history else None

    def get_history(self, n: Optional[int] = None) -> list[dict]:
        """Return the last n entries or all history."""
        if n is None:
            return list(self.history)
        return list(self.history[-n:])

    def reset(self):
        """Clear all recorded metrics."""
        self.history.clear()
