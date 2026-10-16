"""Precedence (TASK.pdf section 5.1). Deterministic: no LLM, no text matching.

Tests run in this order, and a test is applied only if the previous ones left more
than one distinct value:
  1. Authority: policy, addendum, memo, handbook and calendar outrank faq and wiki.
  2. Scope: a document for specific regions outranks a GLOBAL document.
  3. Recency: the later effective_date wins.
If values still differ after all three, the result is a CONFLICT: both values and all
their documents are returned, and nothing is picked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from app.policy.eligibility import is_eligible, is_readable, region_applies
from app.policy.models import GLOBAL_REGION, Candidate, Requester, Resolution, Status, UpcomingChange

AUTHORITATIVE_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})
LOWER_AUTHORITY_TYPES = frozenset({"faq", "wiki"})


def authority_rank(doc_type: str) -> int:
    if doc_type in AUTHORITATIVE_TYPES:
        return 1
    return 0  # faq, wiki, and anything unknown rank lowest (fail safe)


def _distinct_values(candidates: Sequence[Candidate]) -> int:
    return len({candidate.value.casefold() for candidate in candidates})


def _keep_top_authority(candidates: list[Candidate]) -> list[Candidate]:
    top = max(authority_rank(c.meta.doc_type) for c in candidates)
    return [c for c in candidates if authority_rank(c.meta.doc_type) == top]


def _keep_specific_scope(candidates: list[Candidate]) -> list[Candidate]:
    specific = [c for c in candidates if GLOBAL_REGION not in c.meta.regions]
    return specific if specific else candidates


def _keep_latest(candidates: list[Candidate]) -> list[Candidate]:
    latest: date = max(c.meta.effective_date for c in candidates)
    return [c for c in candidates if c.meta.effective_date == latest]


_STEPS: tuple[tuple[str, Callable[[list[Candidate]], list[Candidate]]], ...] = (
    ("authority", _keep_top_authority),
    ("scope", _keep_specific_scope),
    ("recency", _keep_latest),
)


def narrow(eligible: Sequence[Candidate]) -> tuple[list[Candidate], str]:
    """Apply the precedence tests in order and return (cited documents, the test that decided).

    The decided label is the first test after which one value remains. All three tests still
    run, so an agreeing but overridden document (for example a GLOBAL policy under a regional
    addendum that states the same number) is not cited as a source.
    """
    remaining = list(eligible)
    initially_single = _distinct_values(remaining) == 1
    decided = ("only_candidate" if len(remaining) == 1 else "agreement") if initially_single else "unresolved"
    for name, step in _STEPS:
        narrowed = step(remaining)
        if len(narrowed) < len(remaining):
            remaining = narrowed
            if not initially_single and decided == "unresolved" and _distinct_values(remaining) == 1:
                decided = name
    if _distinct_values(remaining) != 1:
        decided = "unresolved"
    return remaining, decided


def resolve(candidates: Sequence[Candidate], requester: Requester, reference: date) -> Resolution:
    """Resolve one rule for one requester on one reference date.

    Only eligible candidates take part. Candidates the requester cannot read are dropped
    here, so they can never affect the result.
    """
    eligible = [c for c in candidates if is_eligible(c.meta, requester, reference)]
    if not eligible:
        return Resolution(Status.NO_ANSWER, (), "none", ())
    remaining, decided = narrow(eligible)
    status = Status.ANSWER if _distinct_values(remaining) == 1 else Status.CONFLICT
    return Resolution(status, tuple(remaining), decided, tuple(eligible))


def upcoming_changes(candidates: Sequence[Candidate], requester: Requester, reference: date) -> list[UpcomingChange]:
    """Documents that take effect after `reference`, that the requester may read, that apply to
    their region, and that would win under the precedence rules on their effective date.

    Documents that would not win (for example a later GLOBAL policy under a regional addendum)
    are not mentioned.
    """
    readable = [
        c for c in candidates
        if is_readable(c.meta, requester) and region_applies(c.meta, requester.region)
    ]
    current = resolve(readable, requester, reference)
    changes: list[UpcomingChange] = []
    future = sorted(
        (c for c in readable if c.meta.effective_date > reference),
        key=lambda c: (c.meta.effective_date, c.meta.doc_id),
    )
    for candidate in future:
        on_date = candidate.meta.effective_date
        at_date = [c for c in readable if c.meta.effective_date <= on_date]
        outcome = resolve(at_date, requester, on_date)
        if outcome.status is not Status.ANSWER or candidate not in outcome.winners:
            continue
        if current.status is Status.ANSWER and current.value.casefold() == outcome.value.casefold():
            continue  # the same value, so not a change
        changes.append(UpcomingChange(candidate=candidate, current_value=current.value))
    return changes
