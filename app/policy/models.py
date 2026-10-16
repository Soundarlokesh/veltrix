"""Data types for eligibility and precedence. Policy code works on these, not on raw text."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from app.ingestion.models import Chunk, ManifestEntry

GLOBAL_REGION = "GLOBAL"


@dataclass(frozen=True)
class Requester:
    """Identity as stored in the database. Never built from message text."""

    employee_id: str
    role: str
    region: str


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    @classmethod
    def from_entry(cls, entry: ManifestEntry) -> "DocMeta":
        return cls(
            doc_id=entry.doc_id,
            title=entry.title,
            doc_type=entry.doc_type,
            regions=entry.regions,
            effective_date=entry.effective_date,
            access_level=entry.access_level,
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "DocMeta":
        return cls(
            doc_id=chunk.doc_id,
            title=chunk.title,
            doc_type=chunk.doc_type,
            regions=chunk.regions,
            effective_date=date.fromisoformat(chunk.effective_date),
            access_level=chunk.access_level,
        )


@dataclass(frozen=True)
class Candidate:
    """One document's statement of a rule, e.g. its sick-leave allowance.

    Deciding which candidates state the same rule is done before this layer. The
    value is already normalised text, so two candidates agree only if values are equal.
    """

    meta: DocMeta
    value: str
    locator: str = ""


class Status(str, Enum):
    ANSWER = "answer"
    CONFLICT = "conflict"
    NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class Resolution:
    status: Status
    winners: tuple[Candidate, ...]
    decided_by: str  # authority | scope | recency | agreement | only_candidate | unresolved | none
    eligible: tuple[Candidate, ...]

    @property
    def value(self) -> str | None:
        return self.winners[0].value if self.status is Status.ANSWER else None


@dataclass(frozen=True)
class UpcomingChange:
    candidate: Candidate
    current_value: str | None
