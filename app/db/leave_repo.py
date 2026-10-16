"""Data access for employees, balances and leave requests. No business rules live here.

Writes take a connection that is already inside a transaction (BEGIN IMMEDIATE), so the
caller decides the transaction boundary. Every query is parameterised.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: str
    created_at: str


_REQUEST_COLUMNS = "request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at"


def _to_request(row: sqlite3.Row) -> LeaveRequest:
    return LeaveRequest(**{key: row[key] for key in row.keys()})


def get_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str) -> float | None:
    row = conn.execute(
        "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
        (employee_id, leave_type),
    ).fetchone()
    return None if row is None else float(row["balance_days"])


def list_balances(conn: sqlite3.Connection, employee_id: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ? ORDER BY leave_type",
        (employee_id,),
    ).fetchall()
    return {row["leave_type"]: float(row["balance_days"]) for row in rows}


def list_requests(conn: sqlite3.Connection, employee_id: str, statuses: tuple[str, ...] | None = None) -> list[LeaveRequest]:
    sql = f"SELECT {_REQUEST_COLUMNS} FROM leave_requests WHERE employee_id = ?"
    args: list[object] = [employee_id]
    if statuses:
        sql += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
        args.extend(statuses)
    sql += " ORDER BY start_date, request_id"
    return [_to_request(row) for row in conn.execute(sql, args).fetchall()]


def get_request(conn: sqlite3.Connection, request_id: str) -> LeaveRequest | None:
    row = conn.execute(
        f"SELECT {_REQUEST_COLUMNS} FROM leave_requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    return None if row is None else _to_request(row)


def direct_reports(conn: sqlite3.Connection, manager_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT employee_id FROM employees WHERE manager_id = ? ORDER BY employee_id",
        (manager_id,),
    ).fetchall()
    return [row["employee_id"] for row in rows]


def overlapping_requests(
    conn: sqlite3.Connection,
    employee_id: str,
    start_date: str,
    end_date: str,
    statuses: tuple[str, ...] = ("pending", "approved"),
) -> list[LeaveRequest]:
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT {_REQUEST_COLUMNS} FROM leave_requests "
        f"WHERE employee_id = ? AND status IN ({placeholders}) "
        "AND start_date <= ? AND end_date >= ? ORDER BY start_date",
        (employee_id, *statuses, end_date, start_date),
    ).fetchall()
    return [_to_request(row) for row in rows]


def team_overlapping(
    conn: sqlite3.Connection,
    employee_ids: list[str],
    start_date: str,
    end_date: str,
) -> list[LeaveRequest]:
    if not employee_ids:
        return []
    placeholders = ",".join("?" for _ in employee_ids)
    rows = conn.execute(
        f"SELECT {_REQUEST_COLUMNS} FROM leave_requests "
        f"WHERE employee_id IN ({placeholders}) AND status IN ('pending','approved') "
        "AND start_date <= ? AND end_date >= ? ORDER BY employee_id, start_date",
        (*employee_ids, end_date, start_date),
    ).fetchall()
    return [_to_request(row) for row in rows]


def insert_request(conn: sqlite3.Connection, request: LeaveRequest) -> None:
    conn.execute(
        "INSERT INTO leave_requests (request_id, employee_id, leave_type, start_date, end_date, "
        "working_days, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            request.request_id,
            request.employee_id,
            request.leave_type,
            request.start_date,
            request.end_date,
            request.working_days,
            request.status,
            request.created_at,
        ),
    )


def deduct_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, days: int) -> int:
    """Subtract only if the balance covers it. Returns rows changed (0 means rejected)."""
    cursor = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days - ? "
        "WHERE employee_id = ? AND leave_type = ? AND balance_days >= ?",
        (days, employee_id, leave_type, days),
    )
    return cursor.rowcount


def restore_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, days: int) -> int:
    cursor = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id = ? AND leave_type = ?",
        (days, employee_id, leave_type),
    )
    return cursor.rowcount


def mark_cancelled(conn: sqlite3.Connection, request_id: str, employee_id: str) -> int:
    """Only a pending or approved request owned by employee_id can change. Returns rows changed."""
    cursor = conn.execute(
        "UPDATE leave_requests SET status = 'cancelled' WHERE request_id = ? AND employee_id = ? "
        "AND status IN ('pending','approved')",
        (request_id, employee_id),
    )
    return cursor.rowcount
