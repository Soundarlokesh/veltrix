"""Load and validate manifest.json (TASK.pdf section 2.2)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.ingestion.errors import ManifestError
from app.ingestion.models import ACCESS_LEVELS, DOC_TYPES, ManifestEntry

REQUIRED_FIELDS = ("doc_id", "file", "title", "doc_type", "regions", "effective_date", "access_level")


def load_manifest(path: Path) -> list[ManifestEntry]:
    try:
        # utf-8-sig accepts an optional byte-order mark, which Windows editors and PowerShell add.
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ManifestError("manifest must be a JSON array")

    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        entry = _parse_entry(item, index)
        if entry.doc_id in seen:
            raise ManifestError(f"duplicate doc_id in manifest: {entry.doc_id}")
        seen.add(entry.doc_id)
        entries.append(entry)
    return entries


def _parse_entry(item: object, index: int) -> ManifestEntry:
    if not isinstance(item, dict):
        raise ManifestError(f"manifest entry {index} is not an object")
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise ManifestError(f"manifest entry {index} is missing fields: {', '.join(missing)}")

    doc_id = str(item["doc_id"]).strip()
    if not doc_id:
        raise ManifestError(f"manifest entry {index} has an empty doc_id")

    regions = item["regions"]
    if not isinstance(regions, list) or not regions or not all(isinstance(r, str) and r for r in regions):
        raise ManifestError(f"{doc_id}: regions must be a non-empty list of strings")

    if item["doc_type"] not in DOC_TYPES:
        raise ManifestError(f"{doc_id}: unknown doc_type {item['doc_type']!r}")
    if item["access_level"] not in ACCESS_LEVELS:
        raise ManifestError(f"{doc_id}: unknown access_level {item['access_level']!r}")

    try:
        effective = date.fromisoformat(str(item["effective_date"]))
    except ValueError as exc:
        raise ManifestError(f"{doc_id}: effective_date must be YYYY-MM-DD") from exc

    return ManifestEntry(
        doc_id=doc_id,
        file=str(item["file"]),
        title=str(item["title"]),
        doc_type=str(item["doc_type"]),
        regions=tuple(regions),
        effective_date=effective,
        access_level=str(item["access_level"]),
    )
