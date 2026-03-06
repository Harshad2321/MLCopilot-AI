"""
MLCopilot AI — Database Layer  (backend/database.py)
=====================================================
Thin SQLite wrapper.

Tables
------
training_runs  — one row per named run (run_id, name, created_at)
metrics        — one row per epoch (run_id, epoch, losses, accuracy, etc.)
analysis_results — persisted issue reports (optional, used by dashboard)

All functions return plain Python dicts so the API layer stays framework-agnostic.
"""

import sqlite3
import os
from datetime import datetime

# The database file lives in the database/ folder at the project root
_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "mlcopilot.db",
)


# ── Connection helper ─────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not already exist."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id     TEXT PRIMARY KEY,
                name       TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         TEXT    NOT NULL,
                epoch          INTEGER NOT NULL,
                train_loss     REAL,
                val_loss       REAL,
                accuracy       REAL,
                learning_rate  REAL,
                gradient_norm  REAL,
                logged_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT    NOT NULL,
                issue      TEXT,
                severity   TEXT,
                reason     TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
            );
        """)


# ── Writes ────────────────────────────────────────────────────────────────────

def ensure_run(run_id: str, name: str = "") -> None:
    """Insert the run record if it doesn't exist yet."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO training_runs (run_id, name) VALUES (?, ?)",
            (run_id, name or run_id),
        )


def insert_metrics(
    run_id: str,
    epoch: int,
    train_loss: float,
    val_loss: float = None,
    accuracy: float = None,
    learning_rate: float = None,
    gradient_norm: float = None,
) -> None:
    """Persist one epoch's worth of metrics."""
    ensure_run(run_id)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO metrics
                (run_id, epoch, train_loss, val_loss, accuracy, learning_rate, gradient_norm)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, epoch, train_loss, val_loss, accuracy, learning_rate, gradient_norm),
        )


def save_analysis(run_id: str, issues: list[dict]) -> None:
    """Persist detected issues for a run (overwrites previous results)."""
    with _connect() as conn:
        conn.execute("DELETE FROM analysis_results WHERE run_id = ?", (run_id,))
        for issue in issues:
            conn.execute(
                """
                INSERT INTO analysis_results (run_id, issue, severity, reason)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    issue.get("issue", ""),
                    issue.get("severity", ""),
                    issue.get("reason", ""),
                ),
            )


# ── Reads ─────────────────────────────────────────────────────────────────────

def fetch_run_metrics(run_id: str) -> list[dict]:
    """Return all metric rows for a run, ordered by epoch."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY epoch ASC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_all_metrics() -> list[dict]:
    """Return every metric row across all runs."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics ORDER BY run_id, epoch ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_all_runs() -> list[dict]:
    """Return summary info about every training run."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, name, created_at FROM training_runs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
