"""Precedence and eligibility on synthetic documents. The worked example in TASK.pdf section 5.2
is reproduced exactly, with its own expected answers (3, 5, 2 days and a v3 mention)."""

from datetime import date

from app.policy.models import Candidate, DocMeta, Requester, Status
from app.policy.eligibility import is_eligible, reference_date
from app.policy.precedence import authority_rank, resolve, upcoming_changes

TODAY = date(2026, 9, 15)
IN_EMPLOYEE = Requester("T-IN", "employee", "IN")
DE_EMPLOYEE = Requester("T-DE", "employee", "DE")
US_EMPLOYEE = Requester("T-US", "employee", "US")


def doc(doc_id, doc_type, regions, effective, access="all"):
    return DocMeta(doc_id, doc_id, doc_type, tuple(regions), date.fromisoformat(effective), access)


def cand(meta, value):
    return Candidate(meta, str(value), meta.doc_id)


# TASK.pdf section 5.2 worked example
GLOBAL_V1 = cand(doc("EX-V1", "policy", ["GLOBAL"], "2025-01-01"), 2)
GLOBAL_V2 = cand(doc("EX-V2", "policy", ["GLOBAL"], "2026-04-01"), 3)
FAQ = cand(doc("EX-FAQ", "faq", ["GLOBAL"], "2026-05-01"), 2)
DE_ADDENDUM = cand(doc("EX-DE", "addendum", ["DE"], "2025-06-01"), 5)
GLOBAL_V3 = cand(doc("EX-V3", "policy", ["GLOBAL"], "2027-01-01"), 4)
EXAMPLE = [GLOBAL_V1, GLOBAL_V2, FAQ, DE_ADDENDUM, GLOBAL_V3]


def test_worked_example_india_employee_gets_three_days():
    result = resolve(EXAMPLE, IN_EMPLOYEE, TODAY)
    assert result.status is Status.ANSWER
    assert result.value == "3"
    assert [c.meta.doc_id for c in result.winners] == ["EX-V2"]
    assert result.decided_by == "recency"


def test_worked_example_india_upcoming_v3_is_mentioned():
    changes = upcoming_changes(EXAMPLE, IN_EMPLOYEE, TODAY)
    assert [c.candidate.meta.doc_id for c in changes] == ["EX-V3"]
    assert changes[0].candidate.value == "4"
    assert changes[0].current_value == "3"
    assert changes[0].candidate.meta.effective_date == date(2027, 1, 1)


def test_worked_example_germany_employee_gets_five_days_by_scope():
    result = resolve(EXAMPLE, DE_EMPLOYEE, TODAY)
    assert result.value == "5"
    assert result.decided_by == "scope"


def test_worked_example_germany_upcoming_v3_is_not_mentioned():
    assert upcoming_changes(EXAMPLE, DE_EMPLOYEE, TODAY) == []


def test_worked_example_historical_date_gives_two_days():
    result = resolve(EXAMPLE, IN_EMPLOYEE, date(2025, 8, 1))
    assert result.value == "2"
    assert [c.meta.doc_id for c in result.winners] == ["EX-V1"]


def test_authority_outranks_newer_faq():
    faq_newer = cand(doc("F", "faq", ["GLOBAL"], "2026-06-01"), "A")
    policy_older = cand(doc("P", "policy", ["GLOBAL"], "2025-01-01"), "B")
    result = resolve([faq_newer, policy_older], IN_EMPLOYEE, TODAY)
    assert result.value == "B"
    assert result.decided_by == "authority"


def test_wiki_and_faq_rank_below_every_authority_type():
    for authority in ("policy", "addendum", "memo", "handbook", "calendar"):
        assert authority_rank(authority) > authority_rank("faq")
        assert authority_rank(authority) > authority_rank("wiki")


def test_regional_addendum_outranks_newer_global_policy():
    global_newer = cand(doc("G", "policy", ["GLOBAL"], "2026-02-01"), "global")
    addendum_older = cand(doc("A", "addendum", ["IN"], "2025-01-01"), "india")
    result = resolve([global_newer, addendum_older], IN_EMPLOYEE, TODAY)
    assert result.value == "india"
    assert result.decided_by == "scope"


def test_recency_picks_later_global_policy():
    old = cand(doc("OLD", "policy", ["GLOBAL"], "2025-01-01"), "old")
    new = cand(doc("NEW", "policy", ["GLOBAL"], "2026-01-01"), "new")
    result = resolve([old, new], IN_EMPLOYEE, TODAY)
    assert result.value == "new"
    assert result.decided_by == "recency"


def test_unresolved_tie_reports_both_values_and_does_not_pick():
    a = cand(doc("TIE-A", "policy", ["GLOBAL"], "2026-01-01"), "10")
    b = cand(doc("TIE-B", "policy", ["GLOBAL"], "2026-01-01"), "12")
    result = resolve([a, b], IN_EMPLOYEE, TODAY)
    assert result.status is Status.CONFLICT
    assert result.value is None
    assert {c.value for c in result.winners} == {"10", "12"}
    assert result.decided_by == "unresolved"


def test_same_value_from_several_documents_is_agreement_not_conflict():
    a = cand(doc("SAME-A", "policy", ["GLOBAL"], "2026-01-01"), "12")
    b = cand(doc("SAME-B", "policy", ["GLOBAL"], "2026-01-01"), "12")
    result = resolve([a, b], IN_EMPLOYEE, TODAY)
    assert result.status is Status.ANSWER
    assert result.value == "12"


def test_no_eligible_document_is_no_answer():
    future = cand(doc("FUT", "policy", ["GLOBAL"], "2027-01-01"), "x")
    result = resolve([future], IN_EMPLOYEE, TODAY)
    assert result.status is Status.NO_ANSWER
    assert result.winners == ()


def test_wrong_region_document_is_not_eligible():
    us_only = cand(doc("US", "addendum", ["US"], "2025-01-01"), "us")
    assert resolve([us_only], IN_EMPLOYEE, TODAY).status is Status.NO_ANSWER


def test_documents_with_multiple_regions_apply_to_each_of_them():
    shared = doc("SHARED", "memo", ["IN", "US"], "2026-07-01")
    assert is_eligible(shared, IN_EMPLOYEE, date(2026, 6, 30)) is False  # the day before it takes effect
    assert is_eligible(shared, IN_EMPLOYEE, date(2026, 7, 1)) is True
    assert is_eligible(shared, US_EMPLOYEE, date(2026, 7, 1)) is True
    assert is_eligible(shared, DE_EMPLOYEE, date(2026, 7, 1)) is False


def test_reference_date_uses_asked_date_when_given():
    assert reference_date(TODAY) == TODAY
    assert reference_date(TODAY, date(2025, 8, 1)) == date(2025, 8, 1)


def test_upcoming_change_hidden_from_requester_without_access():
    hidden_future = cand(doc("HR-FUT", "policy", ["GLOBAL"], "2027-01-01", access="hr"), "secret")
    current = cand(doc("CUR", "policy", ["GLOBAL"], "2025-01-01"), "public")
    assert upcoming_changes([current, hidden_future], IN_EMPLOYEE, TODAY) == []
    hr = Requester("T-HR", "hr", "IN")
    assert [c.candidate.meta.doc_id for c in upcoming_changes([current, hidden_future], hr, TODAY)] == ["HR-FUT"]


def test_upcoming_change_for_another_region_is_not_mentioned():
    us_future = cand(doc("US-FUT", "addendum", ["US"], "2027-01-01"), "us-value")
    current = cand(doc("CUR", "policy", ["GLOBAL"], "2025-01-01"), "global")
    assert upcoming_changes([current, us_future], IN_EMPLOYEE, TODAY) == []


def test_restricted_document_cannot_change_the_visible_answer():
    visible = cand(doc("VIS", "policy", ["GLOBAL"], "2025-01-01"), "visible")
    hidden = cand(doc("HID", "policy", ["GLOBAL"], "2026-01-01", access="managers"), "hidden")
    employee = resolve([visible, hidden], IN_EMPLOYEE, TODAY)
    assert employee.value == "visible"
    assert all(c.meta.doc_id != "HID" for c in employee.eligible)
    manager = resolve([visible, hidden], Requester("T-M", "manager", "IN"), TODAY)
    assert manager.value == "hidden"
