"""Turn each supported document format into ordered (kind, payload, page) items,
then group them into Segments that keep their section heading and page number.

Supported: PDF (native text, tables, scanned pages via OCR), DOCX, Markdown, HTML.
Nothing in this module knows about specific corpus documents.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import docx
import fitz
from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import Comment
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from app.ingestion.errors import IngestionError
from app.ingestion.models import ManifestEntry, ParsedDocument, Segment
from app.ingestion.ocr import ocr_page_text, is_scanned_page, require_tesseract

# "6. Carrying annual leave forward" or "2.1 The document corpus" are headings;
# "15 January 2026" and "10 working days" are not.
HEADING_RE = re.compile(r"^(?:\d+\.\d+\.?|\d+\.)\s+[A-Z]")
MAX_HEADING_CHARS = 90
PDF_NOISE_RE = re.compile(r"^(?:Veltrix Systems\s*[|·].*|Page \d+)$")
MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
MD_SEPARATOR_RE = re.compile(r":?-{2,}:?")
HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
HTML_TEXT_BLOCKS = {"p", "li", "dt", "dd", "pre", "figcaption"}
HTML_INLINE = {"span", "strong", "em", "b", "i", "a", "code", "small", "sup", "sub", "br", "u", "mark"}
HTML_SKIP = {"script", "style", "footer", "nav", "head", "title", "noscript"}

Item = tuple[str, object, int | None]  # (kind, payload, page); kind is heading | text | table


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_scanned_pages(path: Path) -> list[int]:
    """Pages without a text layer, found without running OCR."""
    with fitz.open(path) as pdf:
        return [number for number, page in enumerate(pdf, start=1) if is_scanned_page(page)]


def parse_document(entry: ManifestEntry, corpus_dir: Path, ocr_cmd: str | None = None) -> ParsedDocument:
    path = Path(corpus_dir) / entry.file
    if not path.is_file():
        raise IngestionError(f"{entry.doc_id}: file {entry.file!r} not found in {corpus_dir}")

    suffix = path.suffix.lower()
    page_count: int | None = None
    ocr_pages: list[int] = []
    if suffix == ".pdf":
        items, page_count, ocr_pages = _pdf_items(path, ocr_cmd)
    elif suffix == ".docx":
        items = _docx_items(path)
    elif suffix == ".md":
        items = _markdown_items(path.read_text(encoding="utf-8"))
    elif suffix in {".html", ".htm"}:
        items = _html_items(path.read_text(encoding="utf-8"))
    else:
        raise IngestionError(f"{entry.doc_id}: unsupported file type {suffix!r}")

    segments = build_segments(items)
    if not segments:
        raise IngestionError(f"{entry.doc_id}: no text could be extracted from {entry.file}")
    return ParsedDocument(
        entry=entry,
        sha256=sha256_file(path),
        segments=tuple(segments),
        page_count=page_count,
        ocr_pages=tuple(ocr_pages),
    )


def build_segments(items: list[Item]) -> list[Segment]:
    """Group items into segments. A heading starts a new segment; so does a page change."""
    segments: list[Segment] = []
    heading: str | None = None
    buffer: list[str] = []
    buffer_heading: str | None = None
    buffer_page: int | None = None

    def flush() -> None:
        if buffer:
            segments.append(Segment("text", "\n".join(buffer), buffer_heading, buffer_page))
            buffer.clear()

    for kind, payload, page in items:
        if kind == "table":
            flush()
            rows = tuple(tuple(cell for cell in row) for row in payload if any(row))  # type: ignore[union-attr]
            if rows:
                text = "\n".join(" | ".join(cell for cell in row if cell) for row in rows)
                segments.append(Segment("table", text, heading, page, rows))
            continue
        if kind == "heading":
            flush()
            heading = str(payload)
        elif buffer and page != buffer_page:
            flush()
        if not buffer:
            buffer_heading = heading
            buffer_page = page
        buffer.append(str(payload))
    flush()
    return segments


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


# ---------- PDF ----------

def _pdf_items(path: Path, ocr_cmd: str | None) -> tuple[list[Item], int, list[int]]:
    items: list[Item] = []
    ocr_pages: list[int] = []
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            if is_scanned_page(page):
                require_tesseract(ocr_cmd)
                text = ocr_page_text(page)
                if not text.strip():
                    raise IngestionError(f"OCR returned no text for page {page_no} of {path.name}")
                ocr_pages.append(page_no)
                for line in text.splitlines():
                    _add_text_item(items, line, page_no)
                continue
            for _, kind, payload in _native_page_layout(page):
                if kind == "table":
                    rows = [[_clean(cell) for cell in row] for row in payload]  # type: ignore[union-attr]
                    items.append(("table", rows, page_no))
                else:
                    _add_text_item(items, str(payload), page_no)
        page_count = pdf.page_count
    return items, page_count, ocr_pages


def _add_text_item(items: list[Item], raw_line: str, page: int) -> None:
    line = _clean(raw_line)
    if not line or PDF_NOISE_RE.match(line):
        return
    is_heading = bool(HEADING_RE.match(line)) and len(line) <= MAX_HEADING_CHARS
    items.append(("heading" if is_heading else "text", line, page))


def _native_page_layout(page: fitz.Page) -> list[tuple[float, str, object]]:
    """Text lines and tables of one page, ordered top to bottom. Table text is not repeated as lines."""
    found = page.find_tables().tables
    rects = [fitz.Rect(table.bbox) for table in found]
    layout: list[tuple[float, str, object]] = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:
            continue
        center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(rect.contains(center) for rect in rects):
            continue
        for line in text.splitlines():
            layout.append((y0, "line", line))
    for table, rect in zip(found, rects):
        layout.append((rect.y0, "table", table.extract()))
    layout.sort(key=lambda entry: entry[0])
    return layout


# ---------- DOCX ----------

def _docx_items(path: Path) -> list[Item]:
    document = docx.Document(str(path))
    items: list[Item] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = _clean(paragraph.text)
            if not text:
                continue
            # A paragraph whose style is missing or unknown has style None; treat it as body text.
            style_name = paragraph.style.name if paragraph.style is not None else ""
            kind = "heading" if style_name.startswith("Heading") else "text"
            items.append((kind, text, None))
        elif tag == "tbl":
            table = DocxTable(child, document)
            rows = [[_clean(cell.text) for cell in row.cells] for row in table.rows]
            items.append(("table", rows, None))
    return items


# ---------- Markdown ----------

def _markdown_items(text: str) -> list[Item]:
    items: list[Item] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        if table_rows:
            items.append(("table", list(table_rows), None))
            table_rows.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            cells = [_clean(cell) for cell in line.strip("|").split("|")]
            if cells and all(MD_SEPARATOR_RE.fullmatch(cell) for cell in cells if cell):
                continue
            table_rows.append(cells)
            continue
        flush_table()
        if not line:
            continue
        heading = MD_HEADING_RE.match(line)
        if heading:
            items.append(("heading", _clean(heading.group(1)), None))
        else:
            items.append(("text", _clean(line), None))
    flush_table()
    return items


# ---------- HTML ----------

def _html_items(text: str) -> list[Item]:
    soup = BeautifulSoup(text, "lxml")
    items: list[Item] = []
    _walk_html(soup.body or soup, items)
    return items


def _walk_html(node: Tag, items: list[Item]) -> None:
    inline: list[str] = []

    def flush() -> None:
        value = _clean(" ".join(inline))
        if value:
            items.append(("text", value, None))
        inline.clear()

    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in HTML_SKIP:
            continue
        if name in HTML_HEADINGS:
            flush()
            value = _clean(child.get_text(" "))
            if value:
                items.append(("heading", value, None))
        elif name == "table":
            flush()
            rows = [
                [_clean(cell.get_text(" ")) for cell in row.find_all(["th", "td"], recursive=False)]
                for row in child.find_all("tr")
            ]
            items.append(("table", rows, None))
        elif name in HTML_TEXT_BLOCKS:
            flush()
            value = _clean(child.get_text(" "))
            if value:
                items.append(("text", value, None))
        elif name in HTML_INLINE:
            inline.append(child.get_text(" "))
        else:
            flush()
            _walk_html(child, items)
    flush()
