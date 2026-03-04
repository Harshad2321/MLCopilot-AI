"""
MLCopilot AI - Storage Module
SQLite-backed storage for training logs, metrics, and experiment history.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "mlcopilot.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """Initialize the database with required tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            config TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            val_loss REAL,
            accuracy REAL,
            val_accuracy REAL,
            grad_norm REAL,
            lr REAL,
            batch_size INTEGER,
            extra_metrics TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );

        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            epoch INTEGER,
            issues TEXT,
            root_causes TEXT,
            suggestions TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
    """)

    conn.commit()
    conn.close()


def create_experiment(name: str, config: dict, db_path: str = DB_PATH) -> int:
    """Create a new experiment and return its ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiments (name, config) VALUES (?, ?)",
        (name, json.dumps(config)),
    )
    conn.commit()
    exp_id = cursor.lastrowid
    conn.close()
    return exp_id


def log_metrics(
    experiment_id: int,
    epoch: int,
    metrics: dict,
    step: int = 0,
    db_path: str = DB_PATH,
):
    """Log training metrics for an experiment."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    core_keys = {"loss", "val_loss", "accuracy", "val_accuracy", "grad_norm", "lr", "batch_size"}
    extra = {k: v for k, v in metrics.items() if k not in core_keys}

    cursor.execute(
        """INSERT INTO training_logs
           (experiment_id, epoch, step, loss, val_loss, accuracy, val_accuracy,
            grad_norm, lr, batch_size, extra_metrics)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            experiment_id,
            epoch,
            step,
            metrics.get("loss"),
            metrics.get("val_loss"),
            metrics.get("accuracy"),
            metrics.get("val_accuracy"),
            metrics.get("grad_norm"),
            metrics.get("lr"),
            metrics.get("batch_size"),
            json.dumps(extra) if extra else None,
        ),
    )
    conn.commit()
    conn.close()


def get_metrics(experiment_id: int, db_path: str = DB_PATH) -> list[dict]:
    """Retrieve all metrics for an experiment."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM training_logs WHERE experiment_id = ? ORDER BY epoch, step",
        (experiment_id,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_recent_metrics(
    experiment_id: int, n: int = 10, db_path: str = DB_PATH
) -> list[dict]:
    """Get the last n metric entries for an experiment."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM training_logs
           WHERE experiment_id = ?
           ORDER BY epoch DESC, step DESC
           LIMIT ?""",
        (experiment_id, n),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return list(reversed(rows))


def save_analysis(
    experiment_id: int,
    epoch: int,
    issues: list[dict],
    root_causes: list[dict],
    suggestions: list[dict],
    db_path: str = DB_PATH,
):
    """Save analysis results."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO analysis_results
           (experiment_id, epoch, issues, root_causes, suggestions)
           VALUES (?, ?, ?, ?, ?)""",
        (
            experiment_id,
            epoch,
            json.dumps(issues),
            json.dumps(root_causes),
            json.dumps(suggestions),
        ),
    )
    conn.commit()
    conn.close()


def get_experiment(experiment_id: int, db_path: str = DB_PATH) -> Optional[dict]:
    """Get experiment details."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_experiments(db_path: str = DB_PATH) -> list[dict]:
    """Get all experiments."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def update_experiment_status(
    experiment_id: int, status: str, db_path: str = DB_PATH
):
    """Update experiment status."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE experiments SET status = ? WHERE id = ?",
        (status, experiment_id),
    )
    conn.commit()
    conn.close()


def get_analysis_history(experiment_id: int, db_path: str = DB_PATH) -> list[dict]:
    """Get analysis history for an experiment."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM analysis_results WHERE experiment_id = ? ORDER BY timestamp",
        (experiment_id,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        row["issues"] = json.loads(row["issues"]) if row["issues"] else []
        row["root_causes"] = json.loads(row["root_causes"]) if row["root_causes"] else []
        row["suggestions"] = json.loads(row["suggestions"]) if row["suggestions"] else []
    return rows
