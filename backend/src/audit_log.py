import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import json

DB_PATH = Path(__file__).parent.parent / "data" / "audit_log.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            risk_tier TEXT,
            model_confidence REAL,
            action_status TEXT,
            letter_generated INTEGER,
            citations TEXT,
            similar_cases_used TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_decision(transaction_id, risk_tier, model_confidence, result):
    """Called after every process_dispute() call — one row per decision, no exceptions."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO audit_log
           (transaction_id, timestamp, risk_tier, model_confidence, action_status,
            letter_generated, citations, similar_cases_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            transaction_id,
            datetime.now(timezone.utc).isoformat(),
            risk_tier,
            model_confidence,
            result.get("status"),
            1 if result.get("letter") else 0,
            json.dumps(result.get("key_citations", [])),
            json.dumps(result.get("similar_cases_used", [])),
        ),
    )
    conn.commit()
    conn.close()


def get_audit_summary():
    """Powers the dashboard — counts by action_status, plus recent entries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    by_status = conn.execute(
        "SELECT action_status, COUNT(*) as count FROM audit_log GROUP BY action_status"
    ).fetchall()

    recent = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT 20"
    ).fetchall()

    conn.close()
    return {
        "by_status": [dict(r) for r in by_status],
        "recent": [dict(r) for r in recent],
    }


def get_full_log():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def init_cache_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS result_cache (
            transaction_id TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_cached_result(transaction_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT result_json FROM result_cache WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def cache_result(transaction_id, result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO result_cache (transaction_id, result_json, created_at) VALUES (?, ?, ?)",
        (transaction_id, json.dumps(result), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()