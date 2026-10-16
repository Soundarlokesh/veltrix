"""Regression: the hybrid-work memo overrides the global remote-work limit for India and the US.

Facts used (read from the documents):
- POL-REMOTE-WORK (policy, GLOBAL, 2025-01-01): up to 2 days per week working from home.
- MEMO-HYBRID-2026 (memo, IN + US, effective 2026-07-01): up to 3 days per week for those two
  offices only. It "replaces the limit of 2 days per week ... for these offices only".

Metadata for the memo comes from manifest.json. Its sentence is taken from the OCR output of the
scanned PDF (verified inside the Docker container). The chat-level test uses that sentence as
text so it does not depend on Tesseract being installed on the host.
Expected values are hand-read oracle values; the application never reads them.
"""

from datetime import date

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.models import ParsedDocument, Segment
from app.ingestion.parsers import parse_document
from app.policy.models import Candidate, DocMeta, Requester, Status
from app.policy.precedence import resolve
from tests.helpers import CORPUS_DIR, MANIFEST_PATH, HashingEmbedder
from tests.test_chat_api import make_client  # noqa: F401 - shared fixture
from app.ingestion.manifest import load_manifest

TODAY = date(2026, 10, 14)
MEMO_TEXT = (
    "With effect from 1 July 2026, employees in these two offices may work from home for up to 3 days per week. "
    "This replaces the limit of 2 days per week in the Remote and Hybrid Work Policy for these offices only."
)
IN_EMP = Requester("E2004", "employee", "IN")
US_EMP = Requester("E3004", "employee", "US")
DE_EMP = Requester("E4010", "employee", "DE")


@pytest.fixture
def metas(corpus_available):
    return {e.doc_id: DocMeta.from_entry(e) for e in load_manifest(MANIFEST_PATH)}


def _candidates(metas):
    return [
        Candidate(metas["POL-REMOTE-WORK"], "2 days per week", "2. Working from home"),
        Candidate(metas["MEMO-HYBRID-2026"], "3 days per week", "p. 1"),
    ]


def test_memo_overrides_remote_policy_for_india_on_reference_date(metas):
    result = resolve(_candidates(metas), IN_EMP, TODAY)
    assert result.status is Status.ANSWER
    assert result.value == "3 days per week"
    assert [c.meta.doc_id for c in result.winners] == ["MEMO-HYBRID-2026"]
    assert result.decided_by == "scope"


def test_memo_overrides_remote_policy_for_the_us(metas):
    result = resolve(_candidates(metas), US_EMP, TODAY)
    assert result.value == "3 days per week"
    assert result.decided_by == "scope"


def test_germany_keeps_the_global_limit(metas):
    result = resolve(_candidates(metas), DE_EMP, TODAY)
    assert result.value == "2 days per week"
    assert [c.meta.doc_id for c in result.winners] == ["POL-REMOTE-WORK"]


def test_memo_is_not_in_effect_the_day_before_1_july_2026(metas):
    result = resolve(_candidates(metas), IN_EMP, date(2026, 6, 30))
    assert result.value == "2 days per week"
    assert "MEMO-HYBRID-2026" not in {c.meta.doc_id for c in result.eligible}


def test_memo_metadata_matches_the_assessment_facts(metas):
    memo = metas["MEMO-HYBRID-2026"]
    assert memo.doc_type == "memo"
    assert set(memo.regions) == {"IN", "US"}
    assert memo.effective_date == date(2026, 7, 1)
    assert memo.access_level == "all"


def test_chat_answer_uses_the_memo_for_india_with_citation(make_client, corpus_available):
    from tests.test_chat_api import DATA_ENTRIES, chat, rules_responder
    from app.llm.mock import ScriptedProvider

    remote = parse_document(DATA_ENTRIES["POL-REMOTE-WORK"], CORPUS_DIR, None)
    memo_entry = DATA_ENTRIES["MEMO-HYBRID-2026"]
    memo = ParsedDocument(memo_entry, "0" * 64, (Segment("text", MEMO_TEXT, None, 1),), None, ())
    chunks = chunk_document(remote) + chunk_document(memo)
    embedder = HashingEmbedder()
    from app.retrieval.vector_index import build_index, compute_fingerprint

    index = build_index(chunks, embedder, compute_fingerprint(chunks, embedder.name))
    values = {"POL-REMOTE-WORK": "2 days per week", "MEMO-HYBRID-2026": "3 days per week"}
    provider = ScriptedProvider(rules_responder(values, "remote work days per week"))
    with make_client(provider=provider, index=index) as client:
        result = chat(client, "E2004", "How many days a week can I work from home?")
    assert result["route"] == "rag"
    assert "3 days per week" in result["answer"]
    assert "MEMO-HYBRID-2026" in {c["doc_id"] for c in result["citations"]}
