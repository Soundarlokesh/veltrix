"""Deterministic date phrases (TASK.pdf section 8.4).

- A date without a year means 2026.
- "next week" means Monday to Friday of the following week.
- "next <weekday>" means the first such weekday after today.
- Plain weekday names and vague phrases are ambiguous and must be clarified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
DEFAULT_LEAVE_YEAR = 2026

ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\b(?:,?\s+(\d{4}))?",
    re.IGNORECASE,
)
MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?",
    re.IGNORECASE,
)
NEXT_WEEKDAY_RE = re.compile(r"\bnext\s+(" + "|".join(WEEKDAYS) + r")\b", re.IGNORECASE)
NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class DateMention:
    value: date
    explicit_year: bool


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def explicit_dates(text: str) -> list[tuple[int, date | None, bool]]:
    """(position, date, has_year) for each explicit calendar date, in text order. date is None
    when the phrase is not a real date (for example 31 February)."""
    found: list[tuple[int, date | None, bool]] = []
    for match in ISO_RE.finditer(text):
        found.append((match.start(), _safe_date(int(match[1]), int(match[2]), int(match[3])), True))
    for match in DAY_MONTH_RE.finditer(text):
        year = int(match[3]) if match[3] else DEFAULT_LEAVE_YEAR
        found.append((match.start(), _safe_date(year, MONTHS[match[2].lower()], int(match[1])), bool(match[3])))
    for match in MONTH_DAY_RE.finditer(text):
        year = int(match[3]) if match[3] else DEFAULT_LEAVE_YEAR
        found.append((match.start(), _safe_date(year, MONTHS[match[1].lower()], int(match[2])), bool(match[3])))
    found.sort(key=lambda item: item[0])
    # The two regexes can match the same phrase; keep the first match at each position.
    unique: list[tuple[int, date | None, bool]] = []
    for item in found:
        if not any(abs(item[0] - seen[0]) < 3 for seen in unique):
            unique.append(item)
    return unique


def next_weekday(today: date, weekday: int) -> date:
    days = (weekday - today.weekday()) % 7 or 7
    return today + timedelta(days=days)


def next_week_range(today: date) -> tuple[date, date]:
    this_monday = today - timedelta(days=today.weekday())
    monday = this_monday + timedelta(days=7)
    return monday, monday + timedelta(days=4)


@dataclass(frozen=True)
class LeaveDates:
    start: date | None
    end: date | None
    ambiguous: bool  # a phrase is present but cannot be resolved without asking


def leave_dates(text: str, today: date) -> LeaveDates:
    if NEXT_WEEK_RE.search(text):
        start, end = next_week_range(today)
        return LeaveDates(start, end, ambiguous=False)
    weekday_match = NEXT_WEEKDAY_RE.search(text)
    if weekday_match:
        day = next_weekday(today, WEEKDAYS[weekday_match[1].lower()])
        return LeaveDates(day, day, ambiguous=False)
    dates = [value for _, value, _ in explicit_dates(text)]
    if any(value is None for value in dates):
        return LeaveDates(None, None, ambiguous=True)
    dates = [d for d in dates if d is not None]
    if not dates:
        plain_weekday = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", text, re.IGNORECASE)
        return LeaveDates(None, None, ambiguous=bool(plain_weekday))
    if len(dates) == 1:
        return LeaveDates(dates[0], dates[0], ambiguous=False)
    return LeaveDates(dates[0], dates[1], ambiguous=False)


def asked_policy_date(text: str, today: date) -> date | None:
    """The date a policy question is about, or None for today (APP_TODAY).

    - An explicit date with a year (for example "on 2025-08-01") is used as given.
    - A bare past year (for example "in 2025") means the last day of that year.
    - The current or a later year means today.
    """
    for _, value, has_year in explicit_dates(text):
        if value is not None and has_year:
            return value
    for match in YEAR_RE.finditer(text):
        year = int(match[1])
        if year < today.year:
            return date(year, 12, 31)
    return None
