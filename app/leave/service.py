"""Leave business rules and transactional writes (TASK.pdf sections 6.2 and 6.3).

This layer does not check who is asking: callers (the tools) must authorise first. It
re-checks every rule inside the write transaction, so a proposal that went stale is rejected.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from app.db import leave_repo
from app.db.connection import connect_writable
from app.leave.calendar import CalendarUnavailableError, applicable_holidays, count_working_days
from app.policy.models import Requester

logger = logging.getLogger(__name__)

LEAVE_TYPES = ("annual", "sick")
LEAVE_YEAR = 2026


class LeaveError(Exception):
    """A rule rejected the action. `reason` is safe to show the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ApplyPlan:
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    balance_before: float


def parse_iso_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise LeaveError(f"The {label} must be a date in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise LeaveError(f"The {label} must be a date in YYYY-MM-DD format.") from exc


def _check_leave_type(leave_type: object) -> str:
    if leave_type not in LEAVE_TYPES:
        raise LeaveError("Leave type must be annual or sick.")
    return str(leave_type)


def _plan(conn, store_path: Path, requester: Requester, leave_type: object, start: object, end: object, today: date) -> ApplyPlan:
    leave_type = _check_leave_type(leave_type)
    start_day = parse_iso_date(start, "start date")
    end_day = parse_iso_date(end, "end date")
    if start_day.year != LEAVE_YEAR or end_day.year != LEAVE_YEAR:
        raise LeaveError(f"Leave can only be requested for dates in {LEAVE_YEAR}.")
    if start_day < today:
        raise LeaveError(f"The start date must be on or after {today.isoformat()}.")
    if end_day < start_day:
        raise LeaveError("The end date must be on or after the start date.")

    try:
        holidays = applicable_holidays(store_path, requester.region, LEAVE_YEAR, today)
    except CalendarUnavailableError as exc:
        raise LeaveError(str(exc)) from exc
    working = count_working_days(start_day, end_day, holidays)
    if working == 0:
        raise LeaveError("The selected dates contain no working days.")

    balance = leave_repo.get_balance(conn, requester.employee_id, leave_type)
    if balance is None:
        raise LeaveError(f"No {leave_type} leave balance is recorded for you.")
    if working > balance:
        raise LeaveError(
            f"This request needs {working} {leave_type} working days, but you have {balance:g} left."
        )

    overlaps = leave_repo.overlapping_requests(
        conn, requester.employee_id, start_day.isoformat(), end_day.isoformat()
    )
    if overlaps:
        first = overlaps[0]
        raise LeaveError(
            f"These dates overlap your existing {first.status} request from {first.start_date} to {first.end_date}."
        )
    return ApplyPlan(leave_type, start_day.isoformat(), end_day.isoformat(), working, balance)


def plan_apply(db_path: Path, store_path: Path, requester: Requester, leave_type, start, end, today: date) -> ApplyPlan:
    """Validate without writing. Used for proposals."""
    conn = connect_writable(db_path)
    try:
        return _plan(conn, store_path, requester, leave_type, start, end, today)
    finally:
        conn.close()


def apply_leave(db_path: Path, store_path: Path, requester: Requester, leave_type, start, end, today: date, now: datetime | None = None) -> dict:
    """Create a pending request and deduct its working days in one transaction."""
    conn = connect_writable(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            plan = _plan(conn, store_path, requester, leave_type, start, end, today)
            request_id = str(uuid.uuid4())
            created = (now or datetime.now()).replace(microsecond=0).isoformat()
            leave_repo.insert_request(
                conn,
                leave_repo.LeaveRequest(
                    request_id=request_id,
                    employee_id=requester.employee_id,
                    leave_type=plan.leave_type,
                    start_date=plan.start_date,
                    end_date=plan.end_date,
                    working_days=plan.working_days,
                    status="pending",
                    created_at=created,
                ),
            )
            if leave_repo.deduct_balance(conn, requester.employee_id, plan.leave_type, plan.working_days) != 1:
                raise LeaveError("Your balance changed while the request was being saved. No changes were made.")
            balance_after = leave_repo.get_balance(conn, requester.employee_id, plan.leave_type)
            conn.execute("COMMIT")
        except LeaveError:
            conn.execute("ROLLBACK")
            raise
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.exception("apply_leave failed for employee=%s", requester.employee_id)
            raise LeaveError("The request could not be saved. No changes were made.") from exc
    finally:
        conn.close()
    logger.info("leave applied employee=%s request=%s days=%d", requester.employee_id, request_id, plan.working_days)
    return {
        "request_id": request_id,
        "leave_type": plan.leave_type,
        "start_date": plan.start_date,
        "end_date": plan.end_date,
        "working_days": plan.working_days,
        "status": "pending",
        "balance_after": balance_after,
    }


def _check_cancellable(request: leave_repo.LeaveRequest | None, requester: Requester, today: date) -> leave_repo.LeaveRequest:
    # Same message for "does not exist" and "belongs to someone else": no existence leak.
    if request is None or request.employee_id != requester.employee_id:
        raise LeaveError("I can't find a leave request with that ID for you.")
    if request.status not in ("pending", "approved"):
        raise LeaveError(f"This request is already {request.status} and cannot be cancelled.")
    if date.fromisoformat(request.start_date) <= today:
        raise LeaveError("Requests that start on or before today cannot be cancelled.")
    return request


def plan_cancel(db_path: Path, requester: Requester, request_id: str, today: date) -> leave_repo.LeaveRequest:
    conn = connect_writable(db_path)
    try:
        request = leave_repo.get_request(conn, request_id)
        return _check_cancellable(request, requester, today)
    finally:
        conn.close()


def cancel_leave(db_path: Path, requester: Requester, request_id: str, today: date) -> dict:
    """Set status to cancelled and restore working days in one transaction."""
    conn = connect_writable(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            request = _check_cancellable(leave_repo.get_request(conn, request_id), requester, today)
            if leave_repo.mark_cancelled(conn, request.request_id, requester.employee_id) != 1:
                raise LeaveError("This request changed while it was being cancelled. No changes were made.")
            if leave_repo.restore_balance(conn, requester.employee_id, request.leave_type, request.working_days) != 1:
                raise LeaveError("Your balance could not be restored. No changes were made.")
            balance_after = leave_repo.get_balance(conn, requester.employee_id, request.leave_type)
            conn.execute("COMMIT")
        except LeaveError:
            conn.execute("ROLLBACK")
            raise
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.exception("cancel_leave failed for employee=%s", requester.employee_id)
            raise LeaveError("The request could not be cancelled. No changes were made.") from exc
    finally:
        conn.close()
    logger.info("leave cancelled employee=%s request=%s", requester.employee_id, request_id)
    return {
        "request_id": request.request_id,
        "leave_type": request.leave_type,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "working_days": request.working_days,
        "status": "cancelled",
        "balance_after": balance_after,
    }
