"""Ingestion store: document metadata and extracted holidays in their own SQLite file.

This file lives under VECTOR_DIR, not in veltrix.db, so the assessment database is
never written to by ingestion.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from app.ingestion.holidays import Holiday
from app.ingestion.models import ParsedDocument

# One statement per entry: executescript() would commit the surrounding transaction.
SCHEMA_STATEMENTS = (
    """CREATE TABLE documents (
        doc_id TEXT PRIMARY KEY,
        file TEXT NOT NULL,
        title TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        regions TEXT NOT NULL,
        effective_date TEXT NOT NULL,
        access_level TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        page_count INTEGER,
        ocr_pages TEXT NOT NULL,
        chunk_count INTEGER NOT NULL
    )""",
    """CREATE TABLE holidays (
        doc_id TEXT NOT NULL,
        region TEXT NOT NULL,
        holiday_date TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        weekday_matches INTEGER NOT NULL,
        locator TEXT NOT NULL,
        PRIMARY KEY (doc_id, holiday_date, name)
    )""",
    "CREATE INDEX holidays_region_date ON holidays (region, holiday_date)",
    "CREATE TABLE ingest_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)


def write_store(
    path: Path,
    documents: list[ParsedDocument],
    chunk_counts: dict[str, int],
    holidays: list[Holiday],
    meta: dict[str, str],
) -> None:
    """Replace the whole store in one transaction, so a failed run never leaves a half-written store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN")
        for table in ("documents", "holidays", "ingest_meta"):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    doc.entry.doc_id,
                    doc.entry.file,
                    doc.entry.title,
                    doc.entry.doc_type,
                    json.dumps(list(doc.entry.regions)),
                    doc.entry.effective_date.isoformat(),
                    doc.entry.access_level,
                    doc.sha256,
                    doc.page_count,
                    json.dumps(list(doc.ocr_pages)),
                    chunk_counts.get(doc.entry.doc_id, 0),
                )
                for doc in documents
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO holidays VALUES (?,?,?,?,?,?,?)",
            [
                (
                    h.doc_id,
                    h.region,
                    h.holiday_date.isoformat(),
                    h.name,
                    h.category,
                    int(h.weekday_matches),
                    h.locator,
                )
                for h in holidays
            ],
        )
        connection.executemany(
            "INSERT INTO ingest_meta VALUES (?,?)",
            sorted(meta.items()),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def list_holidays(path: Path, region: str, year: int | None = None, category: str | None = None) -> list[dict]:
    """Holidays for one region, optionally filtered by year and category."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        sql = "SELECT doc_id, region, holiday_date, name, category, weekday_matches, locator FROM holidays WHERE region = ?"
        args: list[object] = [region]
        if year is not None:
            sql += " AND holiday_date >= ? AND holiday_date <= ?"
            args += [date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()]
        if category is not None:
            sql += " AND category = ?"
            args.append(category)
        sql += " ORDER BY holiday_date, name"
        rows = connection.execute(sql, args).fetchall()
    finally:
        connection.close()
    keys = ("doc_id", "region", "holiday_date", "name", "category", "weekday_matches", "locator")
    return [dict(zip(keys, row)) for row in rows]
