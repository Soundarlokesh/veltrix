import json
from datetime import date

import pytest

from app.ingestion.errors import ManifestError
from app.ingestion.manifest import load_manifest
from tests.helpers import CORPUS_DIR, MANIFEST_PATH


def _write(tmp_path, data):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _entry(**overrides):
    base = {
        "doc_id": "POL-TEST",
        "file": "test.md",
        "title": "Test",
        "doc_type": "policy",
        "regions": ["GLOBAL"],
        "effective_date": "2025-01-01",
        "access_level": "all",
    }
    base.update(overrides)
    return base


def test_supplied_manifest_loads_every_entry(corpus_available):
    entries = load_manifest(MANIFEST_PATH)
    assert len(entries) == 30
    assert len({entry.doc_id for entry in entries}) == 30


def test_every_manifest_file_exists_in_corpus(corpus_available):
    missing = [entry.file for entry in load_manifest(MANIFEST_PATH) if not (CORPUS_DIR / entry.file).is_file()]
    assert missing == []


def test_metadata_fields_are_typed(tmp_path):
    entries = load_manifest(_write(tmp_path, [_entry(regions=["IN", "US"], effective_date="2026-07-01")]))
    entry = entries[0]
    assert entry.regions == ("IN", "US")
    assert entry.effective_date == date(2026, 7, 1)
    assert entry.access_level == "all"


@pytest.mark.parametrize(
    "override",
    [
        {"doc_type": "notes"},
        {"access_level": "public"},
        {"effective_date": "01/01/2025"},
        {"regions": []},
        {"regions": "GLOBAL"},
    ],
)
def test_rejects_invalid_values(tmp_path, override):
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, [_entry(**override)]))


def test_rejects_missing_field(tmp_path):
    data = _entry()
    del data["title"]
    with pytest.raises(ManifestError, match="title"):
        load_manifest(_write(tmp_path, [data]))


def test_rejects_duplicate_doc_id(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [_entry(), _entry()]))


def test_rejects_non_array(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, {"doc_id": "x"}))


def test_missing_manifest_is_a_clear_error(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_manifest_with_utf8_byte_order_mark_loads(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes("﻿".encode("utf-8") + json.dumps([_entry()]).encode("utf-8"))
    assert load_manifest(path)[0].doc_id == "POL-TEST"
