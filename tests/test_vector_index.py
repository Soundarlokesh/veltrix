import dataclasses
from datetime import date

import numpy as np

from app.ingestion.chunking import chunk_document
from app.ingestion.models import Chunk, ManifestEntry, ParsedDocument, Segment
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.vector_index import build_index, compute_fingerprint, load_or_build
from tests.helpers import HashingEmbedder


def _chunk(doc_id: str, text: str, access: str = "all") -> Chunk:
    entry = ManifestEntry(doc_id, f"{doc_id}.md", doc_id, "policy", ("GLOBAL",), date(2025, 1, 1), access)
    parsed = ParsedDocument(entry, "0" * 64, (Segment("text", text, "1. Heading", 1),), None, ())
    return chunk_document(parsed)[0]


CHUNKS = [
    _chunk("POL-SICK", "sick leave allowance is twelve working days each year"),
    _chunk("POL-TRAVEL", "hotel limits per night for business travel"),
    _chunk("HR-SECRET", "compensation band salary ranges for senior staff", access="hr"),
]


def test_search_returns_the_most_similar_chunk():
    embedder = HashingEmbedder()
    idx = build_index(CHUNKS, embedder, compute_fingerprint(CHUNKS, embedder.name))
    results = idx.search_text("how many sick leave days", embedder, k=2)
    assert results[0][1].doc_id == "POL-SICK"
    assert len(results) == 2


def test_faiss_results_match_brute_force_numpy_search():
    """Guards the Windows OpenMP workaround in app/__init__.py: FAISS must return the same ranking."""
    embedder = HashingEmbedder()
    idx = build_index(CHUNKS, embedder, compute_fingerprint(CHUNKS, embedder.name))
    vectors = embedder.encode([c.text for c in CHUNKS])
    for text in ["sick leave days", "hotel per night", "salary bands"]:
        query = embedder.encode([text])[0]
        expected_scores = np.sort(vectors @ query)[::-1]
        results = idx.search(query, len(CHUNKS))
        actual_scores = np.array([score for score, _ in results])
        # Compare scores, not ids: tied chunks may come back in either order.
        assert np.allclose(actual_scores, expected_scores, atol=1e-5)
        assert len({chunk.chunk_id for _, chunk in results}) == len(CHUNKS)


def test_access_predicate_is_applied_before_ranking():
    embedder = HashingEmbedder()
    idx = build_index(CHUNKS, embedder, compute_fingerprint(CHUNKS, embedder.name))
    allowed = lambda chunk: chunk.access_level != "hr"  # noqa: E731
    results = idx.search_text("salary bands", embedder, k=3, allow=allowed)
    assert all(chunk.access_level != "hr" for _, chunk in results)
    assert len(results) == 2  # the filtered chunk does not take a slot


def test_index_is_saved_and_reloaded_identically(tmp_path):
    embedder = HashingEmbedder()
    first, rebuilt = load_or_build(CHUNKS, embedder, tmp_path)
    assert rebuilt is True
    second, rebuilt_again = load_or_build(CHUNKS, embedder, tmp_path)
    assert rebuilt_again is False
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]
    query = embedder.encode(["travel hotel"])[0]
    assert first.search(query, 3)[0][1].chunk_id == second.search(query, 3)[0][1].chunk_id


def test_changed_chunk_text_forces_rebuild(tmp_path):
    embedder = HashingEmbedder()
    load_or_build(CHUNKS, embedder, tmp_path)
    changed = [dataclasses.replace(CHUNKS[0], text="sick leave allowance is ten working days each year")] + CHUNKS[1:]
    _, rebuilt = load_or_build(changed, embedder, tmp_path)
    assert rebuilt is True


def test_changed_metadata_forces_rebuild(tmp_path):
    embedder = HashingEmbedder()
    load_or_build(CHUNKS, embedder, tmp_path)
    changed = [dataclasses.replace(CHUNKS[0], effective_date="2026-04-01")] + CHUNKS[1:]
    _, rebuilt = load_or_build(changed, embedder, tmp_path)
    assert rebuilt is True


def test_empty_index_returns_no_results(tmp_path):
    embedder = HashingEmbedder()
    index, _ = load_or_build([], embedder, tmp_path)
    assert index.size == 0
    assert index.search(embedder.encode(["anything"])[0], 3) == []


def test_real_sentence_transformer_embeddings_are_normalised_cpu_vectors():
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.encode(["annual leave carry forward", "sick leave certificate"])
    assert vectors.shape == (2, embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
