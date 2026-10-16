from datetime import date

from app.ingestion.chunking import MAX_CHUNK_TOKENS, chunk_document
from app.ingestion.models import ManifestEntry, ParsedDocument, Segment
from app.ingestion.parsers import parse_document
from app.tokens import count_tokens
from tests.helpers import CORPUS_DIR, MANIFEST_PATH, MEMO_FILE
from app.ingestion.manifest import load_manifest

ENTRY = ManifestEntry("POL-SYN", "syn.md", "Synthetic Policy", "policy", ("IN", "US"), date(2025, 6, 1), "managers")


def _doc(segments, ocr_pages=()):
    return ParsedDocument(ENTRY, "0" * 64, tuple(segments), None, tuple(ocr_pages))


def test_chunk_never_spans_two_headings():
    doc = _doc([
        Segment("text", "1. Alpha\nalpha body", "1. Alpha", 1),
        Segment("text", "2. Beta\nbeta body", "2. Beta", 1),
    ])
    chunks = chunk_document(doc)
    assert [c.heading for c in chunks] == ["1. Alpha", "2. Beta"]
    assert "beta body" not in chunks[0].text
    assert "alpha body" not in chunks[1].text


def test_chunk_keeps_document_metadata_and_locator():
    doc = _doc([Segment("text", "some words here", "1. Section", 2)])
    chunk = chunk_document(doc)[0]
    assert chunk.doc_id == "POL-SYN"
    assert chunk.regions == ("IN", "US")
    assert chunk.effective_date == "2025-06-01"
    assert chunk.access_level == "managers"
    assert chunk.locator == "1. Section"
    assert chunk.page_start == 2 and chunk.page_end == 2


def test_locator_falls_back_to_page_then_title():
    chunks = chunk_document(_doc([Segment("text", "no heading text", None, 3)]))
    assert chunks[0].locator == "p. 3"
    chunks = chunk_document(_doc([Segment("text", "no heading or page", None, None)]))
    assert chunks[0].locator == "Synthetic Policy"


def test_long_text_is_split_within_token_budget():
    long_text = "\n".join(f"Line {i} with several ordinary words about leave." for i in range(400))
    chunks = chunk_document(_doc([Segment("text", long_text, "1. Long", 1)]))
    assert len(chunks) > 1
    assert all(c.tokens <= MAX_CHUNK_TOKENS for c in chunks)
    assert all(c.tokens == count_tokens(c.text) for c in chunks)
    assert all(c.heading == "1. Long" for c in chunks)


def test_single_overlong_line_is_split_by_tokens():
    line = " ".join(["leave"] * 1000)
    chunks = chunk_document(_doc([Segment("text", line, None, 1)]))
    assert len(chunks) > 1
    assert all(c.tokens <= MAX_CHUNK_TOKENS for c in chunks)


def test_ocr_flag_follows_page_source():
    doc = _doc([Segment("text", "scanned words", None, 1)], ocr_pages=(1,))
    assert chunk_document(doc)[0].ocr is True


def test_every_supplied_document_chunks_within_budget(corpus_available):
    entries = load_manifest(MANIFEST_PATH)
    for entry in entries:
        if entry.file == MEMO_FILE:
            continue  # needs OCR; covered in test_parsers
        parsed = parse_document(entry, CORPUS_DIR, None)
        chunks = chunk_document(parsed)
        assert chunks, entry.doc_id
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), entry.doc_id
        for chunk in chunks:
            assert chunk.doc_id == entry.doc_id
            assert chunk.regions == entry.regions
            assert chunk.effective_date == entry.effective_date.isoformat()
            assert chunk.access_level == entry.access_level
            assert chunk.locator
            assert chunk.tokens <= MAX_CHUNK_TOKENS
