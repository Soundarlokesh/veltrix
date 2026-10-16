"""Gap-filling tests: state changes between proposal and confirmation, concurrent confirmation,
hybrid ordering, cross-employee reads, fail-closed roles, and the Groq error mapping."""

import dataclasses
import sqlite3
import threading
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.db import state_repo
from app.llm.groq_provider import GroqProvider
from app.llm.mock import ScriptedProvider
from app.llm.provider import LLMError, LLMTimeoutError
from app.policy.access import can_read
from app.policy.models import Requester
from app.db.connection import connect_readonly
from app.tools.permissions import can_view_employee
from app.tools import leave_tools
from app.tools.leave_tools import ToolContext
from app.db.employees import get_requester
from tests.test_chat_api import (  # reuse the shared fixtures and helpers
    E2001, E2002, E2004, E2005, E2010, E4010, TODAY, chat, make_client, rules_responder, subset_index,
)

from tests.helpers import CORPUS_DIR, HashingEmbedder


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("APP_TODAY", TODAY.isoformat())


def _sql(db, sql, args=()):
    conn = sqlite3.connect(db)
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _one(db, sql, args=()):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


def test_balance_drop_between_proposal_and_confirmation_is_rejected(make_client, leave_db):
    with make_client() as client:
        chat(client, E2010, "I want to take annual leave from 2026-11-06 to 2026-11-10")  # needs 2 days
        _sql(leave_db, "UPDATE leave_balances SET balance_days = 1 WHERE employee_id = ? AND leave_type = 'annual'", (E2010,))
        result = chat(client, E2010, "yes")
    assert result["tool_calls"][0]["status"] == "error"
    assert "you have 1 left" in result["answer"]
    assert _one(leave_db, "SELECT COUNT(*) FROM leave_requests WHERE employee_id = ?", (E2010,)) == 0
    assert _one(leave_db, "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = 'annual'", (E2010,)) == 1.0


def test_overlap_created_between_proposal_and_confirmation_is_rejected(make_client, leave_db):
    with make_client() as client:
        chat(client, E2010, "I want to take annual leave from 2026-11-06 to 2026-11-10")
        _sql(
            leave_db,
            "INSERT INTO leave_requests VALUES ('33333333-3333-4333-8333-333333333333', ?, 'sick', '2026-11-09', '2026-11-09', 1, 'approved', '2026-10-02T09:00:00')",
            (E2010,),
        )
        result = chat(client, E2010, "yes")
    assert result["tool_calls"][0]["status"] == "error"
    assert "overlap" in result["answer"]
    assert _one(leave_db, "SELECT COUNT(*) FROM leave_requests WHERE employee_id = ?", (E2010,)) == 1


def test_cancelled_request_cannot_be_cancelled_again_after_proposal(make_client, leave_db):
    with make_client() as client:
        chat(client, E2004, "Cancel my leave on 2026-11-09")
        _sql(leave_db, "UPDATE leave_requests SET status = 'cancelled' WHERE request_id = '864fa3f3-eab0-4e9b-a5ed-0de47db4304d'")
        result = chat(client, E2004, "yes")
    assert result["tool_calls"][0]["status"] == "error"
    assert _one(leave_db, "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = 'annual'", (E2004,)) == pytest.approx(3.5)


def test_concurrent_confirmations_claim_the_proposal_once(tmp_path):
    state = tmp_path / "state.db"
    state_repo.init_state_db(state)
    from datetime import datetime

    now = datetime(2026, 10, 14, 9, 0, 0)
    state_repo.set_pending(state, "race", E2010, "apply", {"leave_type": "annual", "start_date": "2026-11-06", "end_date": "2026-11-10"}, now)
    claims = []
    lock = threading.Lock()

    def claim():
        result = state_repo.claim_pending(state, "race", now)
        with lock:
            claims.append(result)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for c in claims if c is not None) == 1


def test_expired_proposal_cannot_be_confirmed(tmp_path):
    from datetime import datetime, timedelta

    state = tmp_path / "state.db"
    state_repo.init_state_db(state)
    created = datetime(2026, 10, 14, 9, 0, 0)
    state_repo.set_pending(state, "late", E2010, "cancel", {"request_id": "x"}, created)
    assert state_repo.claim_pending(state, "late", created + timedelta(minutes=11)) is None


def test_hybrid_answer_lists_the_tool_call_first_and_cites_documents(make_client, corpus_available):
    docs = ["TBL-LEAVE-ACCRUAL", "POL-LEAVE-GLOBAL-V2"]
    values = {"TBL-LEAVE-ACCRUAL": "1.5 per month", "POL-LEAVE-GLOBAL-V2": "carry forward 10 days"}
    provider = ScriptedProvider(rules_responder(values, "carry forward or accrual"))
    with make_client(provider=provider, index=subset_index(docs)) as client:
        result = chat(client, E2004, "How many annual leave days will I have by December?")
    assert result["route"] == "hybrid"
    assert result["tool_calls"][0]["name"] == "get_leave_balance"
    assert result["citations"]


def test_employee_cannot_list_another_employees_requests(leave_db, ingested):
    ctx = ToolContext(get_requester(leave_db, E2005), TODAY, leave_db, ingested["store_path"])
    assert leave_tools.list_leave_requests(ctx, employee_id=E2004).status == "error"


def test_manager_cannot_list_indirect_report_requests(leave_db, ingested):
    ctx = ToolContext(get_requester(leave_db, E2001), TODAY, leave_db, ingested["store_path"])
    assert leave_tools.list_leave_requests(ctx, employee_id=E2004).status == "error"


def test_unknown_role_fails_closed(leave_db):
    stranger = Requester(E2004, "contractor", "IN")
    assert can_read("contractor", "all") is False
    conn = connect_readonly(leave_db)
    try:
        assert can_view_employee(conn, stranger, "E2005") is False
    finally:
        conn.close()


def test_no_answer_never_reveals_a_restricted_title(make_client, corpus_available):
    provider = ScriptedProvider(rules_responder({"HR-COMP-BANDS-2026": "band"}, "band"))
    with make_client(provider=provider, index=subset_index(["HR-COMP-BANDS-2026", "TBL-LEAVE-ACCRUAL"])) as client:
        result = chat(client, E2004, "What are the compensation bands for senior staff?")
    assert result["route"] == "no_answer"
    lowered = result["answer"].lower()
    assert "compensation" not in lowered and "hr-comp" not in lowered and "band" not in lowered


class _FakeGroqError(Exception):
    pass


def test_groq_provider_maps_timeouts_and_errors(monkeypatch):
    import groq

    class FakeTimeout(groq.APITimeoutError):
        def __init__(self):
            Exception.__init__(self, "timeout")

    class FakeApiError(groq.APIError):
        def __init__(self):
            Exception.__init__(self, "server")

    class FakeCompletions:
        def __init__(self, error):
            self.error = error

        def create(self, **kwargs):
            raise self.error

    class FakeClient:
        error = None

        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": FakeCompletions(FakeClient.error)})()

    monkeypatch.setattr(groq, "Groq", FakeClient)
    provider = GroqProvider(api_key="test-key-not-real", model="configured-model-id")

    FakeClient.error = FakeTimeout()
    with pytest.raises(LLMTimeoutError):
        provider.complete("sys", "user", 5)

    FakeClient.error = FakeApiError()
    with pytest.raises(LLMError):
        provider.complete("sys", "user", 5)
