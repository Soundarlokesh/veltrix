"""Restricted documents must never reach the LLM context: filtering happens before context
construction, and before vector search results are returned."""

import dataclasses
from datetime import date

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.manifest import load_manifest
from app.ingestion.parsers import parse_document
from app.llm.context import CORPUS_TOKEN_BUDGET, build_context
from app.policy.filtering import EligibleChunk, filter_eligible_chunks
from app.policy.models import Requester
from app.retrieval.vector_index import build_index, compute_fingerprint
from app.tokens import count_tokens
from tests.helpers import CORPUS_DIR, MANIFEST_PATH, MEMO_FILE, HashingEmbedder

TODAY = date(2026, 10, 14)
IN_EMP = Requester("E2004", "employee", "IN")   # real IDs from the database
IN_MGR = Requester("E2001", "manager", "IN")
HR = Requester("E1001", "hr", "IN")
DE_EMP = Requester("E4005", "employee", "DE")


@pytest.fixture(scope="module")
def corpus_chunks():
    if not (CORPUS_DIR.is_dir() and MANIFEST_PATH.is_file()):
        pytest.skip("assessment corpus not present under data/")
    chunks = []
    for entry in load_manifest(MANIFEST_PATH):
        if entry.file == MEMO_FILE:
            continue  # OCR needed; not part of these security checks
        chunks.extend(chunk_document(parse_document(entry, CORPUS_DIR, None)))
    return chunks


def doc_ids(items):
    return {item.chunk.doc_id for item in items}


def test_employee_receives_no_manager_or_hr_chunks(corpus_chunks):
    eligible = filter_eligible_chunks(corpus_chunks, IN_EMP, TODAY)
    assert eligible
    assert {c.chunk.access_level for c in eligible} == {"all"}
    ids = doc_ids(eligible)
    assert "HB-MANAGER" not in ids
    assert "HR-COMP-BANDS-2026" not in ids
    assert "HR-PIP-GUIDE" not in ids


def test_manager_receives_manager_documents_but_not_hr(corpus_chunks):
    ids = doc_ids(filter_eligible_chunks(corpus_chunks, IN_MGR, TODAY))
    assert "HB-MANAGER" in ids
    assert "HR-COMP-BANDS-2026" not in ids


def test_hr_receives_all_access_levels(corpus_chunks):
    ids = doc_ids(filter_eligible_chunks(corpus_chunks, HR, TODAY))
    assert {"HB-MANAGER", "HR-COMP-BANDS-2026", "HR-PIP-GUIDE"} <= ids


def test_region_is_taken_from_the_requester_record(corpus_chunks):
    de = doc_ids(filter_eligible_chunks(corpus_chunks, DE_EMP, TODAY))
    india = doc_ids(filter_eligible_chunks(corpus_chunks, IN_EMP, TODAY))
    assert "ADD-LEAVE-DE" in de and "ADD-LEAVE-DE" not in india
    assert "ADD-LEAVE-IN" in india and "ADD-LEAVE-IN" not in de
    assert "CAL-HOL-IN-2026" not in de


def test_future_documents_are_excluded_on_the_reference_date(corpus_chunks):
    ids = doc_ids(filter_eligible_chunks(corpus_chunks, IN_EMP, TODAY))
    assert "POL-LEAVE-GLOBAL-V3" not in ids  # effective 2027-01-01
    past = doc_ids(filter_eligible_chunks(corpus_chunks, IN_EMP, date(2025, 8, 1)))
    assert "POL-LEAVE-GLOBAL-V2" not in past
    assert "POL-LEAVE-GLOBAL-V1" in past


def test_eligible_chunk_cannot_be_built_by_hand(corpus_chunks):
    with pytest.raises(TypeError):
        EligibleChunk(corpus_chunks[0])


def test_llm_context_rejects_raw_chunks(corpus_chunks):
    with pytest.raises(TypeError):
        build_context([corpus_chunks[0]])


def test_restricted_chunks_never_enter_llm_context(corpus_chunks):
    eligible = filter_eligible_chunks(corpus_chunks, IN_EMP, TODAY)
    context = build_context(eligible)
    assert "HB-MANAGER" not in context.doc_ids
    assert "HR-COMP-BANDS-2026" not in context.doc_ids
    assert "compensation bands" not in context.text.lower()


def test_vector_search_never_returns_a_restricted_chunk(corpus_chunks):
    embedder = HashingEmbedder()
    fingerprint = compute_fingerprint(corpus_chunks, embedder.name)
    index = build_index(corpus_chunks, embedder, fingerprint)
    from app.policy.filtering import chunk_allowed

    query = "compensation bands salary ranges and leave approval by the line manager"
    results = index.search_text(query, embedder, k=8, allow=chunk_allowed(IN_EMP, TODAY))
    assert results
    assert all(chunk.access_level == "all" for _, chunk in results)
    assert "HR-COMP-BANDS-2026" not in {chunk.doc_id for _, chunk in results}


def test_context_respects_the_3000_token_budget(corpus_chunks):
    eligible = filter_eligible_chunks(corpus_chunks, HR, TODAY)  # the largest eligible set
    context = build_context(eligible)
    assert context.tokens <= CORPUS_TOKEN_BUDGET
    assert count_tokens(context.text) == context.tokens
    assert context.dropped > 0, "the full corpus cannot fit in 3,000 tokens"


def test_context_budget_can_be_checked_with_a_small_limit(corpus_chunks):
    eligible = filter_eligible_chunks(corpus_chunks, IN_EMP, TODAY)
    context = build_context(eligible[:5], max_tokens=150)
    assert context.tokens <= 150


def test_modified_access_level_in_metadata_is_enforced(corpus_chunks):
    changed = [
        dataclasses.replace(c, access_level="hr") if c.doc_id == "POL-LEAVE-GLOBAL-V2" else c
        for c in corpus_chunks
    ]
    ids = doc_ids(filter_eligible_chunks(changed, IN_EMP, TODAY))
    assert "POL-LEAVE-GLOBAL-V2" not in ids


def test_modified_effective_date_in_metadata_is_enforced(corpus_chunks):
    changed = [
        dataclasses.replace(c, effective_date="2027-06-01") if c.doc_id == "FAQ-LEAVE" else c
        for c in corpus_chunks
    ]
    assert "FAQ-LEAVE" not in doc_ids(filter_eligible_chunks(changed, IN_EMP, TODAY))
