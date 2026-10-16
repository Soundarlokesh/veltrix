"""Assistant state in state.db (separate from veltrix.db, which keeps its original schema).

- sessions: which employee owns a session_id. Set by the first request, never changed.
- pending_actions: at most one proposed write per session. A row is claimed (status
  'executing') before a write runs, so a repeated confirmation cannot run it twice.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from app.db.connection import connect_writable

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        employee_id TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS pending_actions (
        session_id TEXT PRIMARY KEY,
        employee_id TEXT NOT NULL,
        action TEXT NOT NULL,
        args_json TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )""",
)

PROPOSAL_TTL = timedelta(minutes=10)


def init_state_db(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(path)
    try:
        for statement in SCHEMA:
            conn.execute(statement)
    finally:
        conn.close()


def claim_session(path: Path, session_id: str, employee_id: str, now: datetime) -> str:
    """Create the session for this employee if it is new. Returns the owning employee ID."""
    conn = connect_writable(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, employee_id, created_at) VALUES (?,?,?)",
            (session_id, employee_id, now.replace(microsecond=0).isoformat()),
        )
        row = conn.execute("SELECT employee_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["employee_id"]
    finally:
        conn.close()


def set_pending(path: Path, session_id: str, employee_id: str, action: str, args: dict, now: datetime) -> None:
    """Store one proposal, replacing any earlier one in this session."""
    conn = connect_writable(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO pending_actions "
            "(session_id, employee_id, action, args_json, status, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                employee_id,
                action,
                json.dumps(args, sort_keys=True),
                "pending",
                now.replace(microsecond=0).isoformat(),
                (now + PROPOSAL_TTL).replace(microsecond=0).isoformat(),
            ),
        )
    finally:
        conn.close()


def get_pending(path: Path, session_id: str) -> dict | None:
    conn = connect_writable(path)
    try:
        row = conn.execute(
            "SELECT session_id, employee_id, action, args_json, status, expires_at "
            "FROM pending_actions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "employee_id": row["employee_id"],
        "action": row["action"],
        "args": json.loads(row["args_json"]),
        "status": row["status"],
        "expires_at": row["expires_at"],
    }


def claim_pending(path: Path, session_id: str, now: datetime) -> dict | None:
    """Atomically move a pending proposal to 'executing'. Returns it, or None if none is claimable."""
    conn = connect_writable(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT employee_id, action, args_json, expires_at FROM pending_actions "
                "WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchone()
            if row is None or row["expires_at"] < now.replace(microsecond=0).isoformat():
                conn.execute("DELETE FROM pending_actions WHERE session_id = ? AND status = 'pending'", (session_id,))
                conn.execute("COMMIT")
                return None
            changed = conn.execute(
                "UPDATE pending_actions SET status = 'executing' WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    if changed != 1:
        return None
    return {
        "employee_id": row["employee_id"],
        "action": row["action"],
        "args": json.loads(row["args_json"]),
    }


def clear_pending(path: Path, session_id: str) -> None:
    conn = connect_writable(path)
    try:
        conn.execute("DELETE FROM pending_actions WHERE session_id = ?", (session_id,))
    finally:
        conn.close()
