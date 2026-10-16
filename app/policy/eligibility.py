"""Eligibility: which documents may be used for a requester on a reference date.

A document is eligible only if all three hold (TASK.pdf section 5.1):
  1. the requester's role can read its access_level,
  2. its regions include the requester's region or GLOBAL,
  3. its effective_date is on or before the reference date.
"""

from __future__ import annotations

from datetime import date

from app.policy.access import can_read
from app.policy.models import GLOBAL_REGION, DocMeta, Requester


def is_readable(meta: DocMeta, requester: Requester) -> bool:
    return can_read(requester.role, meta.access_level)


def region_applies(meta: DocMeta, region: str) -> bool:
    return GLOBAL_REGION in meta.regions or region in meta.regions


def is_in_effect(meta: DocMeta, reference: date) -> bool:
    return meta.effective_date <= reference


def is_eligible(meta: DocMeta, requester: Requester, reference: date) -> bool:
    return (
        is_readable(meta, requester)
        and region_applies(meta, requester.region)
        and is_in_effect(meta, reference)
    )


def reference_date(today: date, asked_date: date | None = None) -> date:
    """The date the answer must reflect: the date the user asked about, else today (APP_TODAY)."""
    return asked_date if asked_date is not None else today
