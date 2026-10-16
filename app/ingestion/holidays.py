"""Extract holiday dates from calendar documents (doc_type == "calendar").

A holiday is a table row that has a full date, a weekday name and a holiday name.
Category comes from the section heading and the row text, never from fixed dates:
  restricted - the section heading or name says "restricted" (optional holidays, not closures)
  company    - the name says "company holiday"
  public     - any other holiday row
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from app.ingestion.models import ParsedDocument

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
WEEKDAYS = frozenset({"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"})
DATE_RE = re.compile(r"^(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})$")


@dataclass(frozen=True)
class Holiday:
    doc_id: str
    region: str
    holiday_date: date
    name: str
    category: str
    weekday_matches: bool
    locator: str


def parse_calendar_date(value: str) -> date | None:
    match = DATE_RE.match(value.strip())
    if not match:
        return None
    return datetime.strptime(" ".join(match.groups()), "%d %B %Y").date()


def classify(heading: str | None, name: str) -> str:
    if "restricted" in (heading or "").lower() or "restricted" in name.lower():
        return "restricted"
    if "company holiday" in name.lower():
        return "company"
    return "public"


def extract_holidays(parsed: ParsedDocument) -> list[Holiday]:
    if parsed.entry.doc_type != "calendar":
        return []
    region = ",".join(parsed.entry.regions)
    holidays: list[Holiday] = []
    for segment in parsed.segments:
        if segment.kind != "table":
            continue
        for row in segment.rows:
            cells = [cell.strip() for cell in row if cell and cell.strip()]
            weekday = next((cell for cell in cells if cell in WEEKDAYS), None)
            if weekday is None:
                continue  # metadata or header rows have no weekday column
            dates = [parse_calendar_date(cell) for cell in cells]
            holiday_date = next((d for d in dates if d is not None), None)
            if holiday_date is None:
                continue
            names = [cell for cell, d in zip(cells, dates) if d is None and cell not in WEEKDAYS]
            if not names:
                continue
            name = names[0]
            holidays.append(
                Holiday(
                    doc_id=parsed.entry.doc_id,
                    region=region,
                    holiday_date=holiday_date,
                    name=name,
                    category=classify(segment.heading, name),
                    weekday_matches=holiday_date.strftime("%A") == weekday,
                    locator=segment.heading or parsed.entry.title,
                )
            )
    return holidays
