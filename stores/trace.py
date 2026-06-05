"""
Structured run trace -> SQLite (database/db.sqlite).

Every turn becomes a row in `runs`; every node update streamed during that turn becomes a row
in `events`. This is the transparency/observability layer (SATURDAY_MVP_PLAN.md §5): it makes
every run inspectable after the fact and is the data source the frontend will render later.

It supersedes the scattered `print(perf_counter)` lines — those stay as live console output,
but the durable, queryable record lives here.
"""

import json
import sqlite3
from datetime import datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT,
    query      TEXT,
    started_at TEXT,
    ended_at   TEXT,
    status     TEXT,
    response   TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER,
    seq     INTEGER,
    ts      TEXT,
    node    TEXT,
    summary TEXT,
    data    TEXT
);
"""


def _json_default(o):
    # Messages -> "AIMessage: <content>"; PlanStep/pydantic -> dict; everything else -> str.
    if hasattr(o, "content"):
        return f"{type(o).__name__}: {str(o.content)[:300]}"
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


def _summarize(delta: dict) -> tuple[str, str]:
    parts = []
    if delta.get("plan"):
        parts.append("plan=[" + "; ".join(f"{s['status']}:{s['label']}" for s in delta["plan"]) + "]")
    if delta.get("tools_called"):
        parts.append("tools=" + ", ".join(delta["tools_called"]))
    if "iteration" in delta:
        parts.append(f"iter={delta['iteration']}")
    if "messages" in delta:
        parts.append(f"+{len(delta['messages'])}msg")
    summary = " | ".join(parts) or "(update)"
    data = json.dumps(delta, default=_json_default)[:4000]
    return summary, data


class Tracer:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._seq = 0

    def start_run(self, thread_id: str, query: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (thread_id, query, started_at, status) VALUES (?, ?, ?, ?)",
            (thread_id, query, datetime.now().isoformat(), "running"),
        )
        self.conn.commit()
        self._seq = 0
        return cur.lastrowid

    def log_event(self, run_id: int, node: str, delta: dict) -> None:
        self._seq += 1
        summary, data = _summarize(delta or {})
        self.conn.execute(
            "INSERT INTO events (run_id, seq, ts, node, summary, data) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, self._seq, datetime.now().isoformat(), node, summary, data),
        )
        self.conn.commit()

    def end_run(self, run_id: int, status: str, response: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at = ?, status = ?, response = ? WHERE run_id = ?",
            (datetime.now().isoformat(), status, (response or "")[:2000], run_id),
        )
        self.conn.commit()
