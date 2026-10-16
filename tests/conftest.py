import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.pipeline import run_ingestion
from tests.helpers import CORPUS_DIR, DATA_DIR, MANIFEST_PATH, MEMO_FILE, HashingEmbedder

WORKING_DB = DATA_DIR / "veltrix.db"


@pytest.fixture
def corpus_available():
    if not (CORPUS_DIR.is_dir() and MANIFEST_PATH.is_file()):
        pytest.skip("assessment corpus not present under data/")


@pytest.fixture
def settings_for_tmp(tmp_path: Path) -> Settings:
    """Settings that read the supplied corpus but write only into tmp_path."""
    return dataclasses.replace(
        Settings.from_env(),
        corpus_dir=CORPUS_DIR,
        manifest_path=MANIFEST_PATH,
        vector_dir=tmp_path / "index",
        tesseract_cmd=None,
    )


@pytest.fixture(scope="session")
def ingested(tmp_path_factory):
    """Ingestion output (ingest.db and FAISS index) for the supplied corpus, built once.

    The scanned memo is left out because Tesseract may be missing. Tests that need it skip.
    """
    if not (CORPUS_DIR.is_dir() and MANIFEST_PATH.is_file()):
        pytest.skip("assessment corpus not present under data/")
    base = tmp_path_factory.mktemp("ingest")
    manifest = base / "manifest.json"
    entries = [e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if e["file"] != MEMO_FILE]
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    settings = dataclasses.replace(
        Settings.from_env(),
        corpus_dir=CORPUS_DIR,
        manifest_path=manifest,
        vector_dir=base / "index",
        tesseract_cmd=None,
    )
    embedder = HashingEmbedder()
    report = run_ingestion(settings, embedder=embedder)
    return {
        "store_path": base / "index" / "ingest.db",
        "index": report.index,
        "embedder": embedder,
        "documents": report.documents,
    }


@pytest.fixture
def leave_db(tmp_path) -> Path:
    """A private copy of the working database. Tests may write to it; the original is never used."""
    if not WORKING_DB.is_file():
        pytest.skip("working database copy not present under data/")
    target = tmp_path / "veltrix_test.db"
    shutil.copy(WORKING_DB, target)
    return target
