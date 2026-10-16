"""Precedence on the supplied corpus. Metadata comes from manifest.json; the candidate values are
oracle values read by hand from the documents. The app code never sees these values."""

from datetime import date

import pytest

from app.ingestion.manifest import load_manifest
from app.policy.models import Candidate, DocMeta, Requester, Status
from app.policy.precedence import resolve, upcoming_changes
from tests.helpers import MANIFEST_PATH

TODAY = date(2026, 10, 14)  # fixed reference date for deterministic tests

IN_EMP = Requester("T-IN", "employee", "IN")
US_EMP = Requester("T-US", "employee", "US")
DE_EMP = Requester("T-DE", "employee", "DE")
IN_MGR = Requester("T-INM", "manager", "IN")
HR = Requester("T-HR", "hr", "IN")


@pytest.fixture
def metas(corpus_available):
    return {entry.doc_id: DocMeta.from_entry(entry) for entry in load_manifest(MANIFEST_PATH)}


def c(metas, doc_id, value):
    return Candidate(metas[doc_id], value, doc_id)


def sick_leave(metas):
    # Values as stated in each document.
    return [
        c(metas, "POL-LEAVE-GLOBAL-V1", "10"),         # v1, superseded by v2
        c(metas, "POL-LEAVE-GLOBAL-V2", "12"),
        c(metas, "POL-LEAVE-GLOBAL-V2-SUMMARY", "12"),
        c(metas, "FAQ-LEAVE", "10"),                   # out of date; faq loses on authority
        c(metas, "ADD-LEAVE-DE", "30"),                # Germany addendum
    ]


def test_sick_leave_india_is_twelve_days_from_v2(metas):
    result = resolve(sick_leave(metas), IN_EMP, TODAY)
    assert result.status is Status.ANSWER
    assert result.value == "12"
    assert {c.meta.doc_id for c in result.winners} == {"POL-LEAVE-GLOBAL-V2", "POL-LEAVE-GLOBAL-V2-SUMMARY"}
    assert "FAQ-LEAVE" in {c.meta.doc_id for c in result.eligible}  # in effect and readable
    assert "FAQ-LEAVE" not in {c.meta.doc_id for c in result.winners}  # loses on authority
    assert result.decided_by == "recency"


def test_sick_leave_germany_is_thirty_days_by_scope(metas):
    result = resolve(sick_leave(metas), DE_EMP, TODAY)
    assert result.value == "30"
    assert result.decided_by == "scope"


def test_sick_leave_us_is_twelve_days(metas):
    assert resolve(sick_leave(metas), US_EMP, TODAY).value == "12"


def test_sick_leave_on_historical_date_uses_only_v1(metas):
    result = resolve(sick_leave(metas), IN_EMP, date(2025, 8, 1))
    assert result.value == "10"
    assert [c.meta.doc_id for c in result.winners] == ["POL-LEAVE-GLOBAL-V1"]


def test_v2_is_not_yet_in_effect_on_its_day_minus_one(metas):
    result = resolve(sick_leave(metas), IN_EMP, date(2026, 3, 31))
    assert result.value == "10"  # v2 starts 1 April 2026


def test_medical_certificate_threshold_india_uses_global_v2(metas):
    candidates = [
        c(metas, "POL-LEAVE-GLOBAL-V1", "2 consecutive working days"),
        c(metas, "POL-LEAVE-GLOBAL-V2", "3 consecutive working days"),
        c(metas, "FAQ-LEAVE", "2 working days in a row"),
    ]
    result = resolve(candidates, IN_EMP, TODAY)
    assert result.value == "3 consecutive working days"


def test_medical_certificate_germany_addendum_wins_by_scope(metas):
    candidates = [
        c(metas, "POL-LEAVE-GLOBAL-V2", "3 consecutive working days"),
        c(metas, "ADD-LEAVE-DE", "from the fourth calendar day of illness"),
    ]
    assert resolve(candidates, DE_EMP, TODAY).decided_by == "scope"


def test_carry_forward_follows_regional_addendum(metas):
    carry = [
        c(metas, "POL-LEAVE-GLOBAL-V2", "10"),
        c(metas, "ADD-LEAVE-IN", "15"),
        c(metas, "ADD-LEAVE-US", "5"),
        c(metas, "ADD-LEAVE-DE", "10"),
    ]
    assert resolve(carry, IN_EMP, TODAY).value == "15"
    assert resolve(carry, US_EMP, TODAY).value == "5"
    de = resolve(carry, DE_EMP, TODAY)
    assert de.value == "10"
    assert [c.meta.doc_id for c in de.winners] == ["ADD-LEAVE-DE"]


def test_accrual_india_upcoming_v3_is_mentioned(metas):
    candidates = [
        c(metas, "TBL-LEAVE-ACCRUAL", "2.0 per month (5 years or more)"),
        c(metas, "POL-LEAVE-GLOBAL-V3", "2.25 per month (5 years or more)"),
    ]
    changes = upcoming_changes(candidates, IN_EMP, TODAY)
    assert [c.candidate.meta.doc_id for c in changes] == ["POL-LEAVE-GLOBAL-V3"]
    assert changes[0].candidate.meta.effective_date == date(2027, 1, 1)
    assert changes[0].current_value == "2.0 per month (5 years or more)"


def test_accrual_germany_v3_is_not_mentioned_because_addendum_wins_on_scope(metas):
    candidates = [
        c(metas, "TBL-LEAVE-ACCRUAL", "2.0 per month (5 years or more)"),
        c(metas, "ADD-LEAVE-DE", "2.5 per month"),
        c(metas, "POL-LEAVE-GLOBAL-V3", "2.25 per month (5 years or more)"),
    ]
    assert upcoming_changes(candidates, DE_EMP, TODAY) == []


def test_manager_handbook_is_visible_to_manager_only(metas):
    candidates = [
        c(metas, "POL-LEAVE-GLOBAL-V2", "line manager approves in the HR system"),
        c(metas, "HB-MANAGER", "manager approves in the manager portal"),
    ]
    employee = resolve(candidates, IN_EMP, TODAY)
    assert "HB-MANAGER" not in {c.meta.doc_id for c in employee.eligible}
    manager = resolve(candidates, IN_MGR, TODAY)
    assert "HB-MANAGER" in {c.meta.doc_id for c in manager.eligible}


def test_hr_only_document_gives_no_answer_for_an_employee(metas):
    candidates = [c(metas, "HR-COMP-BANDS-2026", "band L4: 90,000 to 110,000")]
    assert resolve(candidates, IN_EMP, TODAY).status is Status.NO_ANSWER
    assert resolve(candidates, HR, TODAY).status is Status.ANSWER
