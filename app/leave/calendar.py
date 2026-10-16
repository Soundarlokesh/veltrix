"""Working days: weekdays that are not public or company holidays for the requester's region.

Holiday dates come from the ingestion store (calendar documents), never from code. Restricted
holidays are not public holidays and do not reduce working days (TASK.pdf section 6.2 and the
India addendum). Company holidays are treated as office closures (assumption; see README).
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

from app.db.connection import connect_readonly

NON_WORKING_CATEGORIES = ("public", "company")


class CalendarUnavailableError(Exception):
    """The ingestion store (holiday calendars) is missing or unreadable."""


def applicable_holidays(store_path: Path, region: str, year: int, reference: date) -> frozenset[date]:
    if not Path(store_path).is_file():
        raise CalendarUnavailableError("The holiday calendar has not been loaded.")
    connection = connect_readonly(store_path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT h.holiday_date FROM holidays h "
            "JOIN documents d ON d.doc_id = h.doc_id "
            "WHERE h.region = ? AND h.category IN (?, ?) "
            "AND h.holiday_date BETWEEN ? AND ? AND d.effective_date <= ?",
            (
                region,
                *NON_WORKING_CATEGORIES,
                date(year, 1, 1).isoformat(),
                date(year, 12, 31).isoformat(),
                reference.isoformat(),
            ),
        ).fetchall()
    except sqlite3.Error as exc:
        raise CalendarUnavailableError("The holiday calendar could not be read.") from exc
    finally:
        connection.close()
    return frozenset(date.fromisoformat(row[0]) for row in rows)


def count_working_days(start: date, end: date, holidays: frozenset[date] | set[date]) -> int:
    """Inclusive count of Monday-to-Friday dates that are not holidays."""
    total = 0
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            total += 1
        day += timedelta(days=1)
    return total
