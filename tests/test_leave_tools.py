"""Phase D: leave calendar, leave rules, transactions and the five tools with their permissions.

Fixed test date: APP_TODAY = 2026-10-14. Employees, balances and requests come from the
supplied database copy. Expected working-day counts were worked out by hand from the calendar
and the weekdays (see comments)."""

import inspect
import threading
from datetime import date
from pathlib import Path

import pytest

from app.db import leave_repo
from app.db.employees import get_requester
from app.leave import service
from app.leave.calendar import applicable_holidays, count_working_days
from app.leave.service import LeaveError
from app.tools import leave_tools
from app.tools.leave_tools import ToolContext
from app.tools.permissions import BULK_REFUSAL, FIXED_REFUSAL

TODAY = date(2026, 10, 14)

# Real employees from the supplied database
E2004 = "E2004"   # India employee, manager E2002
E2005 = "E2005"   # India employee, manager E2002 (sibling of E2004)
E2002 = "E2002"   # India manager, direct reports E2004, E2005, E2006, E2007
E2001 = "E2001"   # India manager, direct reports E2002, E2003, E2012 (indirect: E2004..E2011)
E1001 = "E1001"   # India HR
E4010 = "E4010"   # Germany employee
E2010 = "E2010"   # India employee, no leave requests


@pytest.fixture
def store(ingested):
    return ingested["store_path"]


def ctx_for(db: Path, store: Path, employee_id: str, today: date = TODAY) -> ToolContext:
    requester = get_requester(db, employee_id)
    assert requester is not None, employee_id
    return ToolContext(requester=requester, today=today, db_path=db, store_path=store)


# ---------- working days and holidays ----------

def test_working_days_exclude_weekends_and_holidays():
    holidays = frozenset({date(2026, 11, 9)})  # India public holiday (Deepavali, Monday)
    # Fri 6 Nov, Sat 7, Sun 8, Mon 9 (holiday), Tue 10 -> 2 working days
    assert count_working_days(date(2026, 11, 6), date(2026, 11, 10), holidays) == 2


def test_weekend_only_range_has_zero_working_days():
    assert count_working_days(date(2026, 10, 17), date(2026, 10, 18), frozenset()) == 0


def test_range_spanning_a_weekend_counts_only_weekdays():
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 19), frozenset()) == 6


def test_india_restricted_holiday_is_not_a_public_holiday(store):
    # 20 March 2026 is an India restricted holiday (Id-ul-Fitr), a Friday.
    holidays = applicable_holidays(store, "IN", 2026, date(2026, 3, 1))
    assert date(2026, 3, 20) not in holidays
    assert date(2026, 3, 19) in holidays  # Ugadi is public
    assert count_working_days(date(2026, 3, 20), date(2026, 3, 20), holidays) == 1


def test_germany_company_holiday_reduces_working_days(store):
    holidays = applicable_holidays(store, "DE", 2026, TODAY)
    assert date(2026, 12, 31) in holidays  # Silvester, company holiday
    assert count_working_days(date(2026, 12, 30), date(2026, 12, 31), holidays) == 1


def test_calendar_documents_must_be_in_effect_on_the_reference_date(store):
    # The India 2026 calendar is effective 2026-01-01; nothing is in effect before that.
    assert applicable_holidays(store, "IN", 2026, date(2025, 6, 1)) == frozenset()


# ---------- apply_leave rules (service layer) ----------

def test_apply_creates_pending_request_and_deducts_balance(leave_db, store):
    req = ctx_for(leave_db, store, E2010).requester
    result = service.apply_leave(leave_db, store, req, "annual", "2026-11-06", "2026-11-10", TODAY)
    assert result["status"] == "pending"
    assert result["working_days"] == 2  # Fri 6 Nov and Tue 10 Nov; Mon 9 Nov is a holiday
    assert result["balance_after"] == pytest.approx(8.0)  # 10 - 2
    assert len(result["request_id"]) == 36  # random UUID4 string


def test_apply_request_id_is_not_guessable_or_repeated(leave_db, store):
    req = ctx_for(leave_db, store, E2010).requester
    first = service.apply_leave(leave_db, store, req, "sick", "2026-12-14", "2026-12-14", TODAY)
    second = service.apply_leave(leave_db, store, req, "sick", "2026-12-15", "2026-12-15", TODAY)
    assert first["request_id"] != second["request_id"]


@pytest.mark.parametrize(
    "leave_type, start, end, fragment",
    [
        ("maternity", "2026-12-14", "2026-12-14", "annual or sick"),
        (None, "2026-12-14", "2026-12-14", "annual or sick"),
        ("annual", None, "2026-12-14", "YYYY-MM-DD"),
        ("annual", "2027-01-04", "2027-01-05", "dates in 2026"),
        ("annual", "2026-10-01", "2026-10-02", "on or after 2026-10-14"),
        ("annual", "2026-12-15", "2026-12-14", "on or after the start date"),
        ("annual", "2026-10-17", "2026-10-18", "no working days"),
        ("annual", "2026-12-14", "2026-12-18", "needs 5 annual working days, but you have 3.5"),
    ],
)
def test_apply_rejections_have_specific_reasons(leave_db, store, leave_type, start, end, fragment):
    req = ctx_for(leave_db, store, E2004).requester
    with pytest.raises(LeaveError) as info:
        service.plan_apply(leave_db, store, req, leave_type, start, end, TODAY)
    assert fragment in info.value.reason


def test_apply_rejects_overlap_with_existing_pending_request(leave_db, store):
    # E2004 has a pending annual request for 9-13 Nov 2026
    req = ctx_for(leave_db, store, E2004).requester
    with pytest.raises(LeaveError) as info:
        service.plan_apply(leave_db, store, req, "sick", "2026-11-12", "2026-11-12", TODAY)
    assert "overlap" in info.value.reason and "pending" in info.value.reason


def test_apply_allows_sick_leave_in_a_free_window(leave_db, store):
    req = ctx_for(leave_db, store, E2004).requester
    plan = service.plan_apply(leave_db, store, req, "sick", "2026-12-14", "2026-12-16", TODAY)
    assert plan.working_days == 3
    assert plan.balance_before == pytest.approx(12.0)


def test_apply_rolls_back_when_balance_deduction_fails(leave_db, store, monkeypatch):
    req = ctx_for(leave_db, store, E2010).requester
    before = service.plan_apply  # noqa: F841 - keep reference for clarity
    monkeypatch.setattr(leave_repo, "deduct_balance", lambda *a, **k: 0)
    with pytest.raises(LeaveError):
        service.apply_leave(leave_db, store, req, "annual", "2026-11-06", "2026-11-10", TODAY)
    conn = leave_repo.get_request  # noqa: F841
    import sqlite3
    c = sqlite3.connect(leave_db)
    try:
        assert c.execute("SELECT COUNT(*) FROM leave_requests WHERE employee_id = ?", (E2010,)).fetchone()[0] == 0
        assert c.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type='annual'", (E2010,)).fetchone()[0] == 10.0
    finally:
        c.close()


def test_apply_rolls_back_on_unexpected_error(leave_db, store, monkeypatch):
    req = ctx_for(leave_db, store, E2010).requester

    def boom(*a, **k):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(leave_repo, "insert_request", boom)
    with pytest.raises(LeaveError) as info:
        service.apply_leave(leave_db, store, req, "annual", "2026-11-06", "2026-11-10", TODAY)
    assert "No changes were made" in info.value.reason
    import sqlite3
    c = sqlite3.connect(leave_db)
    try:
        assert c.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type='annual'", (E2010,)).fetchone()[0] == 10.0
    finally:
        c.close()


def test_concurrent_identical_applications_yield_exactly_one_success(leave_db, store):
    req = ctx_for(leave_db, store, E2010).requester
    outcomes = []
    lock = threading.Lock()

    def attempt():
        try:
            service.apply_leave(leave_db, store, req, "annual", "2026-11-06", "2026-11-10", TODAY)
            result = "ok"
        except LeaveError as exc:
            result = exc.reason
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 1
    assert len(outcomes) == 6
    import sqlite3
    c = sqlite3.connect(leave_db)
    try:
        assert c.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type='annual'", (E2010,)).fetchone()[0] == pytest.approx(8.0)
    finally:
        c.close()


# ---------- cancel_leave rules ----------

def test_cancel_pending_future_request_restores_balance(leave_db, store):
    req = ctx_for(leave_db, store, E2004).requester
    request_id = "864fa3f3-eab0-4e9b-a5ed-0de47db4304d"  # pending annual 9-13 Nov, 4 days
    result = service.cancel_leave(leave_db, req, request_id, TODAY)
    assert result["status"] == "cancelled"
    assert result["balance_after"] == pytest.approx(7.5)  # 3.5 + 4


def test_cancel_rejects_started_or_past_requests(leave_db, store):
    req = ctx_for(leave_db, store, E2004).requester
    with pytest.raises(LeaveError) as info:
        service.plan_cancel(leave_db, req, "66160227-1737-4472-ac16-72297608d942", TODAY)  # approved, Feb
    assert "on or before today cannot be cancelled" in info.value.reason


def test_cancel_twice_is_rejected_and_does_not_restore_twice(leave_db, store):
    req = ctx_for(leave_db, store, E2004).requester
    request_id = "864fa3f3-eab0-4e9b-a5ed-0de47db4304d"
    service.cancel_leave(leave_db, req, request_id, TODAY)
    with pytest.raises(LeaveError) as info:
        service.cancel_leave(leave_db, req, request_id, TODAY)
    assert "already cancelled" in info.value.reason
    import sqlite3
    c = sqlite3.connect(leave_db)
    try:
        assert c.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type='annual'", (E2004,)).fetchone()[0] == pytest.approx(7.5)
    finally:
        c.close()


def test_cancel_of_another_employees_request_has_the_same_message_as_unknown(leave_db, store):
    other = ctx_for(leave_db, store, E2005).requester
    with pytest.raises(LeaveError) as foreign:
        service.plan_cancel(leave_db, other, "864fa3f3-eab0-4e9b-a5ed-0de47db4304d", TODAY)
    with pytest.raises(LeaveError) as unknown:
        service.plan_cancel(leave_db, other, "00000000-0000-4000-8000-000000000000", TODAY)
    assert foreign.value.reason == unknown.value.reason


def test_cancel_rolls_back_when_balance_restore_fails(leave_db, store, monkeypatch):
    req = ctx_for(leave_db, store, E2004).requester
    monkeypatch.setattr(leave_repo, "restore_balance", lambda *a, **k: 0)
    with pytest.raises(LeaveError):
        service.cancel_leave(leave_db, req, "864fa3f3-eab0-4e9b-a5ed-0de47db4304d", TODAY)
    import sqlite3
    c = sqlite3.connect(leave_db)
    try:
        assert c.execute("SELECT status FROM leave_requests WHERE request_id='864fa3f3-eab0-4e9b-a5ed-0de47db4304d'").fetchone()[0] == "pending"
    finally:
        c.close()


# ---------- the five tools: permissions ----------

def test_tool_names_and_arguments_match_the_task():
    assert inspect.signature(leave_tools.get_leave_balance).parameters.keys() - {"ctx"} == {"employee_id", "leave_type"}
    assert inspect.signature(leave_tools.list_leave_requests).parameters.keys() - {"ctx"} == {"employee_id", "status"}
    assert inspect.signature(leave_tools.get_team_leave).parameters.keys() - {"ctx"} == {"start_date", "end_date"}
    assert inspect.signature(leave_tools.apply_leave).parameters.keys() - {"ctx"} == {"leave_type", "start_date", "end_date"}
    assert inspect.signature(leave_tools.cancel_leave).parameters.keys() - {"ctx"} == {"request_id"}


def test_apply_and_cancel_have_no_target_employee_argument():
    for fn in (leave_tools.apply_leave, leave_tools.cancel_leave):
        assert "employee_id" not in inspect.signature(fn).parameters


def test_employee_can_read_own_balance(leave_db, store):
    result = leave_tools.get_leave_balance(ctx_for(leave_db, store, E2004))
    assert result.ok
    assert result.data["balances"]["annual"] == pytest.approx(3.5)


def test_employee_reading_another_employee_gets_the_fixed_refusal(leave_db, store):
    ctx = ctx_for(leave_db, store, E2004)
    result = leave_tools.get_leave_balance(ctx, employee_id=E2005)
    assert result.status == "error" and result.reason == FIXED_REFUSAL


def test_unknown_and_forbidden_targets_get_identical_refusals(leave_db, store):
    ctx = ctx_for(leave_db, store, E2004)
    forbidden = leave_tools.get_leave_balance(ctx, employee_id=E2005)
    unknown = leave_tools.get_leave_balance(ctx, employee_id="E9999")
    assert forbidden.reason == unknown.reason == FIXED_REFUSAL


def test_manager_can_read_direct_report(leave_db, store):
    result = leave_tools.list_leave_requests(ctx_for(leave_db, store, E2002), employee_id=E2004)
    assert result.ok
    assert any(r["status"] == "pending" for r in result.data["requests"])


def test_manager_cannot_read_indirect_report(leave_db, store):
    # E2001 manages E2002, who manages E2004: E2004 is indirect for E2001
    result = leave_tools.get_leave_balance(ctx_for(leave_db, store, E2001), employee_id=E2004)
    assert result.reason == FIXED_REFUSAL


def test_hr_can_read_any_single_employee(leave_db, store):
    assert leave_tools.get_leave_balance(ctx_for(leave_db, store, E1001), employee_id=E4010).ok


def test_hr_unknown_employee_gets_the_fixed_refusal(leave_db, store):
    result = leave_tools.get_leave_balance(ctx_for(leave_db, store, E1001), employee_id="E9999")
    assert result.reason == FIXED_REFUSAL


@pytest.mark.parametrize("token", ["all", "everyone", "*", "ALL"])
def test_bulk_requests_are_refused_for_every_role(leave_db, store, token):
    for employee in (E2004, E2001, E1001):
        ctx = ctx_for(leave_db, store, employee)
        assert leave_tools.get_leave_balance(ctx, employee_id=token).reason == BULK_REFUSAL
        assert leave_tools.list_leave_requests(ctx, employee_id=token).reason == BULK_REFUSAL


def test_only_managers_can_call_get_team_leave(leave_db, store):
    for employee in (E2004, E1001):
        result = leave_tools.get_team_leave(ctx_for(leave_db, store, employee), "2026-11-01", "2026-11-30")
        assert result.status == "error"


def test_team_leave_covers_direct_reports_only(leave_db, store):
    team_e2002 = leave_tools.get_team_leave(ctx_for(leave_db, store, E2002), "2026-11-01", "2026-11-30")
    assert team_e2002.ok
    assert "E2004" in {r["employee_id"] for r in team_e2002.data["requests"]}
    team_e2001 = leave_tools.get_team_leave(ctx_for(leave_db, store, E2001), "2026-11-01", "2026-11-30")
    assert team_e2001.ok
    assert "E2004" not in {r["employee_id"] for r in team_e2001.data["requests"]}  # indirect report


def test_team_leave_rejects_dates_outside_2026(leave_db, store):
    result = leave_tools.get_team_leave(ctx_for(leave_db, store, E2002), "2027-01-01", "2027-01-05")
    assert result.status == "error"


def test_tool_results_contain_no_names_emails_or_phones(leave_db, store):
    import sqlite3
    from app.security.pii import redact_text

    ctx = ctx_for(leave_db, store, E2002)
    outputs = [
        leave_tools.get_leave_balance(ctx, employee_id=E2004).data,
        leave_tools.list_leave_requests(ctx, employee_id=E2004).data,
        leave_tools.get_team_leave(ctx, "2026-11-01", "2026-11-30").data,
    ]
    text = repr(outputs)
    conn = sqlite3.connect(leave_db)
    try:
        personal = [row[0] for row in conn.execute("SELECT full_name FROM employees")]
        personal += [row[0] for row in conn.execute("SELECT email FROM employees")]
        personal += [row[0] for row in conn.execute("SELECT phone FROM employees")]
    finally:
        conn.close()
    for value in personal:
        assert value not in text
    assert "@" not in text
    assert redact_text(text) == text  # no phone-like content either


def test_employee_can_apply_and_cancel_only_their_own(leave_db, store):
    ctx = ctx_for(leave_db, store, E2010)
    applied = leave_tools.apply_leave(ctx, "annual", "2026-11-06", "2026-11-10")
    assert applied.ok
    cancel_other = leave_tools.cancel_leave(ctx_for(leave_db, store, E2005), applied.data["request_id"])
    assert cancel_other.status == "error"
    assert cancel_other.reason == "I can't find a leave request with that ID for you."
    cancelled = leave_tools.cancel_leave(ctx, applied.data["request_id"])
    assert cancelled.ok


def test_cancel_tool_rejects_malformed_ids(leave_db, store):
    ctx = ctx_for(leave_db, store, E2004)
    for bad in ["", "not-a-uuid", "1; DROP TABLE leave_requests", None]:
        assert leave_tools.cancel_leave(ctx, bad).status == "error"
