"""Section-aware chunking. A chunk never spans two section headings, and every chunk
keeps its document metadata and a locator (section heading, else page, else title)."""

from __future__ import annotations

from app.ingestion.models import Chunk, ParsedDocument
from app.tokens import count_tokens, get_encoding

# Kept below the 256-token input limit of the default sentence-transformers model.
MAX_CHUNK_TOKENS = 200


def chunk_document(parsed: ParsedDocument, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    entry = parsed.entry
    pieces: list[tuple[str | None, int | None, str]] = []
    for segment in parsed.segments:
        for text, page in _split_to_fit(segment.text, segment.page, max_tokens):
            pieces.append((segment.heading, page, text))

    chunks: list[Chunk] = []
    lines: list[str] = []
    pages: list[int | None] = []
    current_heading: str | None = None

    def flush() -> None:
        if not lines:
            return
        known_pages = [page for page in pages if page is not None]
        page_start = min(known_pages) if known_pages else None
        page_end = max(known_pages) if known_pages else None
        text = "\n".join(lines)
        if current_heading:
            locator = current_heading
        elif page_start is not None:
            locator = f"p. {page_start}"
        else:
            locator = entry.title
        chunks.append(
            Chunk(
                chunk_id=f"{entry.doc_id}#{len(chunks):03d}",
                doc_id=entry.doc_id,
                title=entry.title,
                doc_type=entry.doc_type,
                regions=entry.regions,
                effective_date=entry.effective_date.isoformat(),
                access_level=entry.access_level,
                heading=current_heading,
                page_start=page_start,
                page_end=page_end,
                ocr=any(page in parsed.ocr_pages for page in known_pages),
                text=text,
                tokens=count_tokens(text),
                locator=locator,
            )
        )
        lines.clear()
        pages.clear()

    for heading, page, text in pieces:
        if lines and (heading != current_heading or count_tokens("\n".join(lines + [text])) > max_tokens):
            flush()
        if not lines:
            current_heading = heading
        lines.append(text)
        pages.append(page)
    flush()
    return chunks


def _split_to_fit(text: str, page: int | None, max_tokens: int) -> list[tuple[str, int | None]]:
    if count_tokens(text) <= max_tokens:
        return [(text, page)]
    parts: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if count_tokens(line) > max_tokens:
            if current:
                parts.append("\n".join(current))
                current = []
            parts.extend(_split_tokens(line, max_tokens))
            continue
        candidate = "\n".join(current + [line])
        if current and count_tokens(candidate) > max_tokens:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return [(part, page) for part in parts]


def _split_tokens(text: str, max_tokens: int) -> list[str]:
    encoding = get_encoding()
    ids = encoding.encode(text, disallowed_special=())
    return [encoding.decode(ids[start:start + max_tokens]) for start in range(0, len(ids), max_tokens)]
