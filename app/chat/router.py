"""One chat turn: confirmation, deterministic routing, tools, documents.

Routing is decided by code (app/chat/intent.py), not by the model. The model only writes
answers to document questions, and only from policy-filtered context.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

from app.chat import intent as I
from app.chat.dates import leave_dates
from app.chat.models import ChatResult, ToolCallRecord
from app.chat.rag import answer_policy_question
from app.db import state_repo
from app.db.connection import connect_readonly
from app.leave import service
from app.leave.service import LeaveError
from app.policy.models import Requester
from app.security.injection import is_manipulation
from app.security.pii import redact_text
from app.tools import leave_tools, permissions
from app.tools.leave_tools import ToolContext, ToolResult

logger = logging.getLogger(__name__)

CHANGE_RE = re.compile(r"\b(instead|make it|change|actually|until|till|through|rather)\b", re.IGNORECASE)
UUID_IN_TEXT_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
REFUSE_MANIPULATION = "I can't help with that. I can answer HR policy questions and help with your own leave."
REFUSE_OFF_TOPIC = "I can only help with HR policies, public holidays and your own leave."
REFUSE_BULK = permissions.BULK_REFUSAL
GREETING_TEXT = "Hello. I can answer HR policy questions and help with your own leave. What would you like to know?"
NOTHING_TO_CONFIRM = "There is nothing waiting to confirm, or the request has expired. Please ask again."


def _record(result: ToolResult) -> ToolCallRecord:
    return ToolCallRecord(result.name, result.args, result.status)


def _fmt_days(value: float) -> str:
    return f"{value:g}"


class Turn:
    """Everything one message needs. Built per request; holds no state between requests."""

    def __init__(self, services, requester: Requester, session_id: str, message: str, today: date, now: datetime) -> None:
        self.s = services
        self.requester = requester
        self.session_id = session_id
        self.message = message.strip()
        self.today = today
        self.now = now
        self.ctx = ToolContext(requester=requester, today=today, db_path=services.db_path, store_path=services.store_path)

    # ----- entry point -----

    def run(self) -> ChatResult:
        pending = state_repo.get_pending(self.s.state_db, self.session_id)
        if pending is not None and pending["employee_id"] != self.requester.employee_id:
            state_repo.clear_pending(self.s.state_db, self.session_id)
            pending = None

        if pending is not None and I.CONFIRM_RE.match(self.message):
            return self._confirm()

        prior = pending
        if prior is not None:
            # Anything other than a confirmation discards the proposal (TASK.pdf 8.2).
            state_repo.clear_pending(self.s.state_db, self.session_id)

        intent = I.classify(self.message, manipulation=is_manipulation(self.message))
        if prior is not None and prior["action"] == "apply" and intent in (I.Intent.OFF_TOPIC, I.Intent.POLICY, I.Intent.APPLY):
            dates = leave_dates(self.message, self.today)
            has_date = dates.start is not None or dates.ambiguous
            changing = bool(I.leave_type_mentions(self.message)) and bool(CHANGE_RE.search(self.message))
            if has_date or changing:
                return self._apply(prior)  # changed details: a new proposal, merged with the old one
        if prior is not None and prior["action"] == "cancel" and UUID_IN_TEXT_RE.search(self.message):
            intent = I.Intent.CANCEL

        if intent is I.Intent.MANIPULATION:
            return ChatResult("refuse", REFUSE_MANIPULATION)
        if intent is I.Intent.BULK:
            return ChatResult("refuse", REFUSE_BULK)
        others = I.employee_ids(self.message) - {self.requester.employee_id}
        if others:
            return self._other_employee(others, intent)
        if intent is I.Intent.GREETING:
            return ChatResult("clarify", GREETING_TEXT)
        if intent is I.Intent.CONFIRM:
            return ChatResult("clarify", NOTHING_TO_CONFIRM)
        if intent is I.Intent.CANCEL:
            return self._cancel()
        if intent is I.Intent.APPLY:
            return self._apply(None)
        if intent is I.Intent.BALANCE:
            return self._balance(policy_too=False)
        if intent is I.Intent.HYBRID:
            return self._balance(policy_too=True)
        if intent is I.Intent.REQUESTS:
            return self._requests()
        if intent is I.Intent.TEAM:
            return self._team()
        if intent is I.Intent.POLICY:
            return self._policy()
        return ChatResult("refuse", REFUSE_OFF_TOPIC)

    # ----- helpers -----

    def _policy(self) -> ChatResult:
        result = answer_policy_question(self.s, self.requester, self.message, self.today)
        citations = tuple(result.citations)
        return ChatResult(result.route, result.answer, citations)

    def _tool_error_reply(self, result: ToolResult) -> ChatResult:
        return ChatResult("tool", result.reason or "The request could not be completed.", (), (_record(result),))

    def _other_employee(self, others: set[str], intent: I.Intent) -> ChatResult:
        target = sorted(others)[0]
        if intent in (I.Intent.BALANCE, I.Intent.REQUESTS):
            conn = connect_readonly(self.s.db_path)
            try:
                allowed = permissions.can_view_employee(conn, self.requester, target)
            finally:
                conn.close()
            if allowed:
                if intent is I.Intent.BALANCE:
                    result = leave_tools.get_leave_balance(self.ctx, employee_id=target)
                else:
                    result = leave_tools.list_leave_requests(self.ctx, employee_id=target)
                if result.ok:
                    return ChatResult("tool", self._summarise(result), (), (_record(result),))
                return ChatResult("refuse", result.reason or permissions.FIXED_REFUSAL)
        return ChatResult("refuse", permissions.FIXED_REFUSAL)

    def _summarise(self, result: ToolResult) -> str:
        data = result.data
        if result.name == "get_leave_balance":
            if "leave_type" in data:
                return f"{data['leave_type'].capitalize()} leave balance: {_fmt_days(data['balance_days'])} days."
            parts = [f"{k} {_fmt_days(v)} days" for k, v in sorted(data["balances"].items())]
            return "Leave balances: " + ", ".join(parts) + "."
        if result.name == "list_leave_requests":
            if not data["requests"]:
                return "No leave requests found."
            lines = [
                f"- {r['start_date']} to {r['end_date']}, {r['leave_type']}, {r['status']}, "
                f"{r['working_days']} working days (ID {r['request_id']})"
                for r in data["requests"]
            ]
            return "Your leave requests:\n" + "\n".join(lines)
        if result.name == "get_team_leave":
            if not data["requests"]:
                return "No pending or approved leave for your direct reports in that range."
            lines = [
                f"- {r['employee_id']}: {r['start_date']} to {r['end_date']}, {r['leave_type']}, {r['status']}"
                for r in data["requests"]
            ]
            return "Your direct reports' leave:\n" + "\n".join(lines)
        return "Done."

    # ----- leave data -----

    def _balance(self, policy_too: bool) -> ChatResult:
        lowered = self.message.lower()
        types = I.leave_type_mentions(self.message)
        leave_type = next(iter(types)) if len(types) == 1 else None
        result = leave_tools.get_leave_balance(self.ctx, employee_id=self.requester.employee_id, leave_type=leave_type)
        if not result.ok:
            return self._tool_error_reply(result)
        text = self._summarise(result)
        if not policy_too:
            return ChatResult("tool", text, (), (_record(result),))
        policy = self._policy()
        if policy.route == "rag":
            return ChatResult("hybrid", f"{text}\n\n{policy.answer}", policy.citations, (_record(result),))
        return ChatResult("tool", f"{text}\n\n{policy.answer}", (), (_record(result),))

    def _requests(self) -> ChatResult:
        status = I.status_mention(self.message)
        result = leave_tools.list_leave_requests(self.ctx, employee_id=self.requester.employee_id, status=status)
        if not result.ok:
            return self._tool_error_reply(result)
        return ChatResult("tool", self._summarise(result), (), (_record(result),))

    def _team(self) -> ChatResult:
        if not permissions.can_view_team(self.requester):
            return ChatResult("refuse", "Only managers can view team leave.")
        dates = leave_dates(self.message, self.today)
        if dates.start is None or dates.end is None:
            return ChatResult("clarify", "Which date range should I check? For example 'next week' or 2026-11-01 to 2026-11-30.")
        result = leave_tools.get_team_leave(self.ctx, dates.start.isoformat(), dates.end.isoformat())
        if not result.ok:
            return self._tool_error_reply(result)
        return ChatResult("tool", self._summarise(result), (), (_record(result),))

    # ----- proposals -----

    def _apply(self, prior: dict | None) -> ChatResult:
        types = I.leave_type_mentions(self.message)
        leave_type = next(iter(types)) if len(types) == 1 else None
        dates = leave_dates(self.message, self.today)
        prior_args = prior["args"] if prior and prior["action"] == "apply" else {}
        if leave_type is None and prior_args.get("leave_type") and not types:
            leave_type = prior_args["leave_type"]
        start = dates.start.isoformat() if dates.start else prior_args.get("start_date")
        end = dates.end.isoformat() if dates.end else prior_args.get("end_date")
        if dates.start and not dates.end:
            end = dates.start.isoformat()
        if leave_type is None and len(types) > 1:
            return ChatResult("clarify", "Which type of leave do you want: annual or sick?")
        if leave_type is None:
            return ChatResult("clarify", "Which type of leave do you want: annual or sick?")
        if dates.ambiguous or not start:
            return ChatResult("clarify", "Which dates do you want? Please give a start date and an end date, for example 2026-11-06 to 2026-11-10.")
        start_day = date.fromisoformat(start)
        if start_day < self.today:
            return ChatResult("clarify", f"That date has already passed. Please give a date on or after {self.today.isoformat()}.")

        args = {"leave_type": leave_type, "start_date": start, "end_date": end}
        try:
            plan = service.plan_apply(self.s.db_path, self.s.store_path, self.requester, leave_type, start, end, self.today)
        except LeaveError as exc:
            return ChatResult("tool", exc.reason, (), (ToolCallRecord("apply_leave", args, "error"),))
        state_repo.set_pending(self.s.state_db, self.session_id, self.requester.employee_id, "apply", args, self.now)
        text = (
            f"Please confirm: {plan.leave_type} leave from {plan.start_date} to {plan.end_date}, "
            f"{plan.working_days} working days. Reply 'yes' to confirm."
        )
        return ChatResult(
            "tool", text, (), (ToolCallRecord("apply_leave", args, "pending_confirmation"),), pending_confirmation=True
        )

    def _cancel(self) -> ChatResult:
        match = UUID_IN_TEXT_RE.search(self.message)
        conn = connect_readonly(self.s.db_path)
        try:
            from app.db import leave_repo

            mine = [
                r for r in leave_repo.list_requests(conn, self.requester.employee_id, ("pending", "approved"))
                if date.fromisoformat(r.start_date) > self.today
            ]
        finally:
            conn.close()
        if match:
            chosen = [r for r in mine if r.request_id == match.group(0).lower()]
        else:
            dates = leave_dates(self.message, self.today)
            if dates.start is not None:
                chosen = [r for r in mine if r.start_date == dates.start.isoformat()]
            else:
                chosen = mine
        if not chosen:
            return ChatResult("clarify", "I could not find a leave request you can cancel. Please give its request ID or start date.")
        if len(chosen) > 1:
            options = "; ".join(f"{r.start_date} to {r.end_date} ({r.leave_type}, ID {r.request_id})" for r in chosen)
            return ChatResult("clarify", f"Which request should I cancel? {options}")
        target = chosen[0]
        args = {"request_id": target.request_id}
        try:
            service.plan_cancel(self.s.db_path, self.requester, target.request_id, self.today)
        except LeaveError as exc:
            return ChatResult("tool", exc.reason, (), (ToolCallRecord("cancel_leave", args, "error"),))
        state_repo.set_pending(self.s.state_db, self.session_id, self.requester.employee_id, "cancel", args, self.now)
        text = (
            f"Please confirm: cancel your {target.leave_type} leave from {target.start_date} to "
            f"{target.end_date} ({target.working_days} working days). Reply 'yes' to confirm."
        )
        return ChatResult(
            "tool", text, (), (ToolCallRecord("cancel_leave", args, "pending_confirmation"),), pending_confirmation=True
        )

    def _confirm(self) -> ChatResult:
        claimed = state_repo.claim_pending(self.s.state_db, self.session_id, self.now)
        if claimed is None:
            return ChatResult("clarify", NOTHING_TO_CONFIRM)
        try:
            if claimed["employee_id"] != self.requester.employee_id:
                return ChatResult("refuse", "I can't confirm that request.")
            args = claimed["args"]
            if claimed["action"] == "apply":
                result = leave_tools.apply_leave(self.ctx, **args)
            elif claimed["action"] == "cancel":
                result = leave_tools.cancel_leave(self.ctx, **args)
            else:
                return ChatResult("refuse", "I can't confirm that request.")
        finally:
            state_repo.clear_pending(self.s.state_db, self.session_id)
        if result.ok:
            data = result.data
            if result.name == "apply_leave":
                text = (
                    f"Your {data['leave_type']} leave from {data['start_date']} to {data['end_date']} is pending "
                    f"approval ({data['working_days']} working days). Request ID: {data['request_id']}."
                )
            else:
                text = f"Your leave from {data['start_date']} to {data['end_date']} has been cancelled."
        else:
            text = result.reason or "The request could not be completed."
        return ChatResult("tool", text, (), (_record(result),))


def handle_message(services, requester: Requester, session_id: str, message: str, today: date, now: datetime) -> ChatResult:
    turn = Turn(services, requester, session_id, message, today, now)
    result = turn.run()
    # Final output filter: personal data never leaves the service in an answer.
    return ChatResult(
        route=result.route,
        answer=redact_text(result.answer),
        citations=result.citations,
        tool_calls=result.tool_calls,
        pending_confirmation=result.pending_confirmation,
    )

