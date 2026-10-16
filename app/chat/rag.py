"""Document answers: retrieve -> filter -> budgeted context -> model -> validate -> precedence.

Deterministic controls around the model:
- Only policy-filtered chunks (EligibleChunk / UpcomingChunk) can enter the context.
- Chunks that contain instructions aimed at an AI assistant are quarantined.
- The model's citations and rules must refer to chunk ids that were in the context.
- Citation locators come from the chunk metadata, never from model text.
- When the model gives rules, precedence (app/policy/precedence.py) decides the value.
  Conflicts are reported, not resolved by picking one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.chat.dates import asked_policy_date
from app.chat.models import Citation
from app.ingestion.models import Chunk
from app.llm.context import build_context
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.llm.provider import LLMError, LLMTimeoutError
from app.policy.eligibility import reference_date
from app.policy.filtering import chunk_allowed, filter_eligible_chunks, filter_upcoming_chunks, upcoming_allowed
from app.policy.models import Candidate, DocMeta, Requester, Status
from app.policy.precedence import resolve, upcoming_changes
from app.security.injection import contains_untrusted_instruction
from app.security.pii import redact_text

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "The HR documents available to you do not cover this question. Please contact the HR team."
UNPARSEABLE_TEXT = "I could not produce a reliable answer from the documents. Please contact the HR team."
TIMEOUT_TEXT = "The assistant took too long to respond. Please try again, or contact the HR team."
UNAVAILABLE_TEXT = "The assistant is unavailable right now. Please try again later."
MAX_VALUE_CHARS = 200


@dataclass(frozen=True)
class PolicyResult:
    route: str  # rag | no_answer
    answer: str
    citations: tuple[Citation, ...]
    llm_calls: int


@dataclass(frozen=True)
class _Rule:
    topic: str
    value: str
    chunk_id: str


@dataclass(frozen=True)
class _ModelOutput:
    status: str
    answer: str
    cited_chunk_ids: tuple[str, ...]
    rules: tuple[_Rule, ...]


def parse_model_output(raw: str) -> _ModelOutput | None:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    status = data.get("status")
    answer = data.get("answer", "")
    cited = data.get("cited_chunk_ids", [])
    rules_raw = data.get("rules", [])
    if status not in ("answer", "no_answer") or not isinstance(answer, str):
        return None
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        return None
    if not isinstance(rules_raw, list):
        return None
    rules: list[_Rule] = []
    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        topic, value, chunk_id = item.get("topic"), item.get("value"), item.get("chunk_id")
        if all(isinstance(v, str) for v in (topic, value, chunk_id)):
            rules.append(_Rule(topic, value, chunk_id))
    return _ModelOutput(status, answer, tuple(cited), tuple(rules))


def _topic_key(topic: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", topic.lower()).split())


def _value_key(value: str) -> str:
    """Display form of a value: whitespace collapsed, case kept. Precedence compares case-insensitively."""
    return " ".join(value.split())[:MAX_VALUE_CHARS]


def _quarantined(chunk: Chunk) -> bool:
    if contains_untrusted_instruction(chunk.text):
        logger.warning("quarantined chunk doc_id=%s (instruction-like text)", chunk.doc_id)
        return True
    return False


def _cite(chunk: Chunk) -> Citation:
    return Citation(doc_id=chunk.doc_id, locator=chunk.locator)


def _unique(citations: list[Citation]) -> tuple[Citation, ...]:
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for citation in citations:
        key = (citation.doc_id, citation.locator)
        if key not in seen:
            seen.add(key)
            out.append(citation)
    return tuple(out)


def answer_policy_question(services, requester: Requester, message: str, today) -> PolicyResult:
    reference = reference_date(today, asked_policy_date(message, today))
    question = redact_text(message)

    top_k = services.settings.rag_top_k
    in_hits = services.index.search_text(
        question, services.embedder, top_k, allow=chunk_allowed(requester, reference)
    )
    up_hits = services.index.search_text(
        question, services.embedder, top_k, allow=upcoming_allowed(requester, reference)
    )
    in_chunks = [c for _, c in in_hits if not _quarantined(c)]
    up_chunks = [c for _, c in up_hits if not _quarantined(c)]
    eligible = filter_eligible_chunks(in_chunks, requester, reference)
    upcoming = filter_upcoming_chunks(up_chunks, requester, reference)
    if not eligible:
        return PolicyResult("no_answer", NO_ANSWER_TEXT, (), 0)

    context = build_context(list(eligible) + list(upcoming))
    try:
        raw = services.provider.complete(SYSTEM_PROMPT, build_user_prompt(question, context.text), services.settings.llm_timeout_seconds)
    except LLMTimeoutError:
        return PolicyResult("no_answer", TIMEOUT_TEXT, (), 1)
    except LLMError:
        return PolicyResult("no_answer", UNAVAILABLE_TEXT, (), 1)

    output = parse_model_output(raw)
    if output is None:
        logger.warning("model reply was not valid JSON; no answer given")
        return PolicyResult("no_answer", UNPARSEABLE_TEXT, (), 1)

    in_by_id = {item.chunk.chunk_id: item.chunk for item in eligible}
    up_by_id = {item.chunk.chunk_id: item.chunk for item in upcoming}
    if output.status == "no_answer":
        return PolicyResult("no_answer", NO_ANSWER_TEXT, (), 1)

    if output.rules:
        answer, citations = _compose_from_rules(output.rules, in_by_id, up_by_id, requester, reference)
        if not citations:
            return PolicyResult("no_answer", NO_ANSWER_TEXT, (), 1)
        return PolicyResult("rag", redact_text(answer), citations, 1)

    cited = [in_by_id[c] for c in output.cited_chunk_ids if c in in_by_id]
    if not cited or not output.answer.strip():
        return PolicyResult("no_answer", NO_ANSWER_TEXT, (), 1)
    return PolicyResult("rag", redact_text(output.answer.strip()), _unique([_cite(c) for c in cited]), 1)


def _compose_from_rules(rules, in_by_id, up_by_id, requester, reference) -> tuple[str, tuple[Citation, ...]]:
    current: dict[str, tuple[str, list[Candidate]]] = {}
    upcoming_cands: dict[str, list[Candidate]] = {}
    dropped = 0
    for rule in rules:
        key = _topic_key(rule.topic)
        if not key:
            dropped += 1
            continue
        if rule.chunk_id in in_by_id:
            chunk = in_by_id[rule.chunk_id]
            candidate = Candidate(DocMeta.from_chunk(chunk), _value_key(rule.value), chunk.locator)
            current.setdefault(key, (rule.topic.strip(), []))[1].append(candidate)
        elif rule.chunk_id in up_by_id:
            chunk = up_by_id[rule.chunk_id]
            upcoming_cands.setdefault(key, []).append(Candidate(DocMeta.from_chunk(chunk), _value_key(rule.value), chunk.locator))
        else:
            dropped += 1  # the model cited a chunk that was not in its context
    if dropped:
        logger.info("dropped %d invalid rule(s)", dropped)

    lines: list[str] = []
    citations: list[Citation] = []
    for key, (display, candidates) in current.items():
        outcome = resolve(candidates, requester, reference)
        if outcome.status is Status.ANSWER:
            winner = outcome.winners[0]
            sources = "; ".join(f"{c.meta.title}, {c.locator}" for c in outcome.winners)
            lines.append(f"{display}: {winner.value} (source: {sources}).")
        elif outcome.status is Status.CONFLICT:
            values = "; ".join(f"{c.value} ({c.meta.title}, {c.locator})" for c in outcome.winners)
            lines.append(f"{display}: the documents conflict and I cannot choose between them: {values}.")
        else:
            continue
        for winner in outcome.winners:  # cite only the documents that decided the value
            citations.append(Citation(doc_id=winner.meta.doc_id, locator=winner.locator))
        for upcoming_change in upcoming_changes(candidates + upcoming_cands.get(key, []), requester, reference):
            meta = upcoming_change.candidate.meta
            lines.append(
                f"Upcoming change: {display} will be {upcoming_change.candidate.value} from "
                f"{meta.effective_date.isoformat()} (source: {meta.title}, {upcoming_change.candidate.locator})."
            )
            citations.append(Citation(doc_id=meta.doc_id, locator=upcoming_change.candidate.locator))
    return "\n".join(lines), _unique(citations) if lines else ()
