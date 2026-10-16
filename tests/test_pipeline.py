import dataclasses
import hashlib
import json
import shutil
import sqlite3

import pytest

from app.ingestion.errors import OCRUnavailableError
from app.ingestion.manifest import load_manifest
from app.ingestion.pipeline import STORE_FILE, run_ingestion
from app.ingestion.store import list_holidays
from tests.helpers import CORPUS_DIR, HashingEmbedder, MANIFEST_PATH, MEMO_FILE, ORIGINAL_DB, tesseract_available


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_without_memo(tmp_path, source=MANIFEST_PATH):
    data = [item for item in json.loads(source.read_text(encoding="utf-8")) if item["file"] != MEMO_FILE]
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def _store_rows(vector_dir, sql, args=()):
    connection = sqlite3.connect(f"file:{vector_dir / STORE_FILE}?mode=ro", uri=True)
    try:
        return connection.execute(sql, args).fetchall()
    finally:
        connection.close()


def test_scanned_memo_without_tesseract_fails_without_writing_store(corpus_available, settings_for_tmp, tmp_path):
    if tesseract_available():
        pytest.skip("Tesseract is installed; the failure path is covered by test_parsers")
    settings = dataclasses.replace(settings_for_tmp, tesseract_cmd=str(tmp_path / "missing" / "tesseract.exe"))
    with pytest.raises(OCRUnavailableError):
        run_ingestion(settings, embedder=HashingEmbedder())
    assert not (settings.vector_dir / STORE_FILE).exists(), "a failed run must not leave a store behind"


def test_all_manifest_entries_except_scanned_are_processed(corpus_available, settings_for_tmp, tmp_path):
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    report = run_ingestion(settings, embedder=HashingEmbedder())
    manifest = load_manifest(settings.manifest_path)
    assert report.documents == len(manifest) == 29
    stored = {row[0] for row in _store_rows(settings.vector_dir, "SELECT doc_id FROM documents")}
    assert stored == {entry.doc_id for entry in manifest}


def test_stored_metadata_matches_manifest(corpus_available, settings_for_tmp, tmp_path):
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    run_ingestion(settings, embedder=HashingEmbedder())
    rows = {row[0]: row for row in _store_rows(
        settings.vector_dir,
        "SELECT doc_id, title, doc_type, regions, effective_date, access_level, file FROM documents",
    )}
    for entry in load_manifest(settings.manifest_path):
        doc_id, title, doc_type, regions, effective, access, file_name = rows[entry.doc_id]
        assert title == entry.title
        assert doc_type == entry.doc_type
        assert json.loads(regions) == list(entry.regions)
        assert effective == entry.effective_date.isoformat()
        assert access == entry.access_level
        assert file_name == entry.file


def test_holidays_are_readable_by_region_and_year(corpus_available, settings_for_tmp, tmp_path):
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    run_ingestion(settings, embedder=HashingEmbedder())
    store = settings.vector_dir / STORE_FILE
    india_2026_public = list_holidays(store, "IN", year=2026, category="public")
    assert len(india_2026_public) == 12
    assert all(h["holiday_date"].startswith("2026-") for h in india_2026_public)
    assert list_holidays(store, "IN", year=2026, category="company") == []
    assert {h["region"] for h in list_holidays(store, "DE", year=2026)} == {"DE"}


def test_rerun_reuses_index_and_force_rebuilds(corpus_available, settings_for_tmp, tmp_path):
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    embedder = HashingEmbedder()
    first = run_ingestion(settings, embedder=embedder)
    second = run_ingestion(settings, embedder=embedder)
    forced = run_ingestion(settings, embedder=embedder, force_rebuild=True)
    assert first.index_rebuilt is True
    assert second.index_rebuilt is False
    assert forced.index_rebuilt is True
    assert second.index.size == first.index.size == first.chunks


def test_modified_corpus_and_metadata_trigger_rebuild(corpus_available, settings_for_tmp, tmp_path):
    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS_DIR, corpus_copy, ignore=shutil.ignore_patterns(MEMO_FILE))
    manifest_copy = _manifest_without_memo(tmp_path)
    settings = dataclasses.replace(
        settings_for_tmp,
        corpus_dir=corpus_copy,
        manifest_path=manifest_copy,
        vector_dir=tmp_path / "index",
    )
    embedder = HashingEmbedder()
    run_ingestion(settings, embedder=embedder)

    # Modified corpus: add a new Markdown document and register it in the manifest.
    manifest = json.loads(manifest_copy.read_text(encoding="utf-8"))
    manifest.append({
        "doc_id": "TEST-MODIFIED",
        "file": "test_modified.md",
        "title": "Test Modified",
        "doc_type": "policy",
        "regions": ["GLOBAL"],
        "effective_date": "2025-01-01",
        "access_level": "all",
    })
    manifest_copy.write_text(json.dumps(manifest), encoding="utf-8")
    (corpus_copy / "test_modified.md").write_text("# Sick leave\n\nSick leave is 14 working days.\n", encoding="utf-8")

    report = run_ingestion(settings, embedder=embedder)
    assert report.index_rebuilt is True
    texts = [c.text for c in report.index.chunks if c.doc_id == "TEST-MODIFIED"]
    assert any("14 working days" in text for text in texts)

    manifest[-1]["effective_date"] = "2026-01-01"
    manifest_copy.write_text(json.dumps(manifest), encoding="utf-8")
    again = run_ingestion(settings, embedder=embedder)
    assert again.index_rebuilt is True, "a metadata change must rebuild the index"
    effective = [c.effective_date for c in again.index.chunks if c.doc_id == "TEST-MODIFIED"]
    assert effective == ["2026-01-01"]


def test_original_assessment_database_is_never_modified(corpus_available, settings_for_tmp, tmp_path):
    if not ORIGINAL_DB.is_file():
        pytest.skip("original database not present under data/")
    before = _sha256(ORIGINAL_DB)
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    run_ingestion(settings, embedder=HashingEmbedder())
    assert _sha256(ORIGINAL_DB) == before


def test_report_counts_corpus_tokens(corpus_available, settings_for_tmp, tmp_path):
    settings = dataclasses.replace(settings_for_tmp, manifest_path=_manifest_without_memo(tmp_path))
    report = run_ingestion(settings, embedder=HashingEmbedder())
    assert report.corpus_tokens == sum(chunk.tokens for chunk in report.index.chunks)
    assert report.corpus_tokens > 0


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract OCR is not installed or not configured")
def test_full_corpus_including_scanned_memo(corpus_available, settings_for_tmp):
    report = run_ingestion(settings_for_tmp, embedder=HashingEmbedder())
    assert report.documents == 30
    assert report.ocr_doc_ids == ("MEMO-HYBRID-2026",)
    memo_chunks = [c for c in report.index.chunks if c.doc_id == "MEMO-HYBRID-2026"]
    assert memo_chunks and all(c.ocr for c in memo_chunks)
