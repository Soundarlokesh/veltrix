"""Filter retrieved chunks before anything can reach the LLM (TASK.pdf section 7.2).

Two wrapper types exist, and both can only be created in this module:
  EligibleChunk  - readable, applies to the region, and in effect on the reference date.
  UpcomingChunk  - readable, applies to the region, and takes effect after the reference date
                   (used only to mention upcoming changes).
build_context() in app/llm/context.py accepts nothing else, so a raw Chunk cannot reach the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date

from app.ingestion.models import Chunk
from app.policy.eligibility import is_eligible, is_readable, region_applies
from app.policy.models import DocMeta, Requester

_CREATE_TOKEN = object()


class EligibleChunk:
    """A chunk the requester may read and that is in effect on the reference date."""

    __slots__ = ("chunk",)
    status = "IN_EFFECT"

    def __init__(self, chunk: Chunk, _token: object = None) -> None:
        if _token is not _CREATE_TOKEN:
            raise TypeError("EligibleChunk is created only by filter_eligible_chunks()")
        self.chunk = chunk

    def __repr__(self) -> str:
        return f"EligibleChunk({self.chunk.chunk_id})"


class UpcomingChunk:
    """A chunk the requester may read that is not yet in effect and applies to their region."""

    __slots__ = ("chunk",)
    status = "UPCOMING"

    def __init__(self, chunk: Chunk, _token: object = None) -> None:
        if _token is not _CREATE_TOKEN:
            raise TypeError("UpcomingChunk is created only by filter_upcoming_chunks()")
        self.chunk = chunk

    def __repr__(self) -> str:
        return f"UpcomingChunk({self.chunk.chunk_id})"


def chunk_allowed(requester: Requester, reference: date) -> Callable[[Chunk], bool]:
    """Search predicate: readable, region-applicable and in effect. Restricted chunks never rank."""

    def allowed(chunk: Chunk) -> bool:
        return is_eligible(DocMeta.from_chunk(chunk), requester, reference)

    return allowed


def upcoming_allowed(requester: Requester, reference: date) -> Callable[[Chunk], bool]:
    def allowed(chunk: Chunk) -> bool:
        meta = DocMeta.from_chunk(chunk)
        return is_readable(meta, requester) and region_applies(meta, requester.region) and meta.effective_date > reference

    return allowed


def filter_eligible_chunks(chunks: Iterable[Chunk], requester: Requester, reference: date) -> list[EligibleChunk]:
    allowed = chunk_allowed(requester, reference)
    return [EligibleChunk(chunk, _CREATE_TOKEN) for chunk in chunks if allowed(chunk)]


def filter_upcoming_chunks(chunks: Iterable[Chunk], requester: Requester, reference: date) -> list[UpcomingChunk]:
    allowed = upcoming_allowed(requester, reference)
    return [UpcomingChunk(chunk, _CREATE_TOKEN) for chunk in chunks if allowed(chunk)]
