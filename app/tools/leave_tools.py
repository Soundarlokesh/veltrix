"""The five leave tools (TASK.pdf section 6.1).

Every permission check happens here, using ToolContext.requester, which comes from the
database record for X-Employee-Id. Tool arguments can name a target employee, but they never
grant access: the check decides. Results never contain names, emails or phone numbers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.db import leave_repo
from app.db.connection import connect_readonly
from app.leave import service
from app.leave.service import LEAVE_TYPES, LEAVE_YEAR, LeaveError, parse_iso_date
from app.tools import permissions
from app.tools.permissions import BULK_REFUSAL, FIXED_REFUSAL

logger = logging.getLogger(__name__)

REQUEST_STATUSES = ("pending", "approved", "cancelled")


@dataclass(frozen=True)
class ToolContext:
    """Trusted execution context. Built by the API layer from the database, never from text."""

    requester: Any  # app.policy.models.Requester
    today: date
    db_path: Path
    store_path: Path


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict
    status: str  # success | error
    reason: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success"


def _ok(name: str, args: dict, data: dict) -> ToolResult:
    logger.info("tool=%s status=success", name)
    return ToolResult(name, args, "success", None, data)


def _error(name: str, args: dict, reason: str) -> ToolResult:
    logger.info("tool=%s status=error", name)
    return ToolResult(name, args, "error", reason, {})


def _target(employee_id: object, requester_id: str) -> str | None:
    """The employee ID the tool should act on, or None when the argument is invalid."""
    if employee_id is None:
        return requester_id
    if not isinstance(employee_id, str):
        return None
    cleaned = employee_id.strip()
    return cleaned or requester_id


def get_leave_balance(ctx: ToolContext, employee_id: str | None = None, leave_type: str | None = None) -> ToolResult:
    name = "get_leave_balance"
    args = {"employee_id": employee_id, "leave_type": leave_type}
    if isinstance(employee_id, str) and employee_id.strip().lower() in permissions.BULK_TOKENS:
        return _error(name, args, BULK_REFUSAL)
    target = _target(employee_id, ctx.requester.employee_id)
    if target is None:
        return _error(name, args, FIXED_REFUSAL)
    if leave_type is not None and leave_type not in LEAVE_TYPES:
        return _error(name, args, "Leave type must be annual or sick.")
    conn = connect_readonly(ctx.db_path)
    try:
        if not permissions.can_view_employee(conn, ctx.requester, target):
            return _error(name, args, FIXED_REFUSAL)
        balances = leave_repo.list_balances(conn, target)
    finally:
        conn.close()
    if leave_type is not None:
        if leave_type not in balances:
            return _error(name, args, f"No {leave_type} leave balance is recorded.")
        data = {"employee_id": target, "leave_type": leave_type, "balance_days": balances[leave_type]}
    else:
        data = {"employee_id": target, "balances": balances}
    return _ok(name, args, data)


def list_leave_requests(ctx: ToolContext, employee_id: str | None = None, status: str | None = None) -> ToolResult:
    name = "list_leave_requests"
    args = {"employee_id": employee_id, "status": status}
    if isinstance(employee_id, str) and employee_id.strip().lower() in permissions.BULK_TOKENS:
        return _error(name, args, BULK_REFUSAL)
    target = _target(employee_id, ctx.requester.employee_id)
    if target is None:
        return _error(name, args, FIXED_REFUSAL)
    if status is not None and status not in REQUEST_STATUSES:
        return _error(name, args, "Status must be pending, approved or cancelled.")
    conn = connect_readonly(ctx.db_path)
    try:
        if not permissions.can_view_employee(conn, ctx.requester, target):
            return _error(name, args, FIXED_REFUSAL)
        statuses = (status,) if status else None
        requests = leave_repo.list_requests(conn, target, statuses)
    finally:
        conn.close()
    data = {
        "employee_id": target,
        "requests": [
            {
                "request_id": r.request_id,
                "leave_type": r.leave_type,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "working_days": r.working_days,
                "status": r.status,
            }
            for r in requests
        ],
    }
    return _ok(name, args, data)


def get_team_leave(ctx: ToolContext, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
    name = "get_team_leave"
    args = {"start_date": start_date, "end_date": end_date}
    if not permissions.can_view_team(ctx.requester):
        return _error(name, args, "Only managers can view team leave.")
    try:
        start = parse_iso_date(start_date, "start date")
        end = parse_iso_date(end_date, "end date")
    except LeaveError as exc:
        return _error(name, args, exc.reason)
    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR or end < start:
        return _error(name, args, f"Choose an end date on or after the start date, both in {LEAVE_YEAR}.")
    conn = connect_readonly(ctx.db_path)
    try:
        reports = leave_repo.direct_reports(conn, ctx.requester.employee_id)
        requests = leave_repo.team_overlapping(conn, reports, start.isoformat(), end.isoformat())
    finally:
        conn.close()
    data = {
        "requests": [
            {
                "employee_id": r.employee_id,
                "leave_type": r.leave_type,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "working_days": r.working_days,
                "status": r.status,
            }
            for r in requests
        ]
    }
    return _ok(name, args, data)


def apply_leave(ctx: ToolContext, leave_type: str | None = None, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
    name = "apply_leave"
    args = {"leave_type": leave_type, "start_date": start_date, "end_date": end_date}
    try:
        data = service.apply_leave(
            ctx.db_path, ctx.store_path, ctx.requester, leave_type, start_date, end_date, ctx.today
        )
    except LeaveError as exc:
        return _error(name, args, exc.reason)
    return _ok(name, args, data)


def cancel_leave(ctx: ToolContext, request_id: str | None = None) -> ToolResult:
    name = "cancel_leave"
    args = {"request_id": request_id}
    if not isinstance(request_id, str):
        return _error(name, args, "I can't find a leave request with that ID for you.")
    try:
        uuid.UUID(request_id.strip())
    except ValueError:
        return _error(name, args, "I can't find a leave request with that ID for you.")
    try:
        data = service.cancel_leave(ctx.db_path, ctx.requester, request_id.strip(), ctx.today)
    except LeaveError as exc:
        return _error(name, args, exc.reason)
    return _ok(name, args, data)


TOOL_NAMES = ("get_leave_balance", "list_leave_requests", "get_team_leave", "apply_leave", "cancel_leave")
