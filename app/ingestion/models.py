"""Plain data types shared by ingestion, retrieval and policy code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

DOC_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar", "faq", "wiki"})
ACCESS_LEVELS = frozenset({"all", "managers", "hr"})


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str


@dataclass(frozen=True)
class Segment:
    """A run of text or one table, tagged with the section heading and page it came from."""

    kind: str  # "text" or "table"
    text: str
    heading: str | None
    page: int | None
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    entry: ManifestEntry
    sha256: str
    segments: tuple[Segment, ...]
    page_count: int | None
    ocr_pages: tuple[int, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: str
    access_level: str
    heading: str | None
    page_start: int | None
    page_end: int | None
    ocr: bool
    text: str
    tokens: int
    locator: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["regions"] = list(self.regions)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        values = dict(data)
        values["regions"] = tuple(values["regions"])
        return cls(**values)
