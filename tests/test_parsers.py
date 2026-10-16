import docx
import pytest
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.config import Settings
from app.ingestion.errors import IngestionError, OCRUnavailableError
from app.ingestion.manifest import load_manifest
from app.ingestion.models import ManifestEntry
from app.ingestion.ocr import detect_scanned_pages
from app.ingestion.parsers import parse_document
from tests.helpers import CORPUS_DIR, MANIFEST_PATH, MEMO_FILE, tesseract_available

from datetime import date


def _entry_for(file_name: str, doc_type: str = "policy") -> ManifestEntry:
    for entry in load_manifest(MANIFEST_PATH):
        if entry.file == file_name:
            return entry
    raise AssertionError(f"{file_name} not in manifest")


def _all_text(parsed):
    return "\n".join(segment.text for segment in parsed.segments)


# ---- PDF ----

def test_pdf_keeps_headings_pages_and_tables(corpus_available):
    parsed = parse_document(_entry_for("leave_policy_global_v2.pdf"), CORPUS_DIR, None)
    assert parsed.page_count == 2
    assert parsed.ocr_pages == ()

    headings = [s.heading for s in parsed.segments if s.heading]
    assert any(h.startswith("6. Carrying annual leave forward") for h in headings)

    pages = {s.page for s in parsed.segments}
    assert pages == {1, 2}

    tables = [s for s in parsed.segments if s.kind == "table"]
    assert tables, "the comparison table must be extracted as a table segment"
    sick_rows = [row for t in tables for row in t.rows if row and row[0] == "Sick leave per year"]
    assert sick_rows == [("Sick leave per year", "10 working days", "12 working days")]


def test_pdf_header_and_page_number_lines_are_removed(corpus_available):
    parsed = parse_document(_entry_for("leave_policy_global_v2.pdf"), CORPUS_DIR, None)
    text = _all_text(parsed)
    assert "Veltrix Systems  |" not in text
    assert "POL-LEAVE-GLOBAL-V2  |" not in text
    assert "\nPage 2\n" not in text


def test_pdf_without_text_is_detected_as_scanned_without_ocr(corpus_available):
    assert detect_scanned_pages(CORPUS_DIR / MEMO_FILE) == [1]
    assert detect_scanned_pages(CORPUS_DIR / "leave_policy_global_v2.pdf") == []


def test_missing_tesseract_gives_clear_actionable_error(corpus_available, tmp_path):
    missing_cmd = str(tmp_path / "no-such-folder" / "tesseract.exe")
    with pytest.raises(OCRUnavailableError) as info:
        parse_document(_entry_for(MEMO_FILE), CORPUS_DIR, missing_cmd)
    message = str(info.value)
    assert "TESSERACT_CMD" in message
    assert "tesseract" in message.lower()


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract OCR is not installed or not configured")
def test_scanned_memo_is_read_with_ocr(corpus_available):
    parsed = parse_document(_entry_for(MEMO_FILE), CORPUS_DIR, Settings.from_env().tesseract_cmd)
    assert parsed.ocr_pages == (1,)
    text = _all_text(parsed).lower()
    assert len(text) > 200
    assert "hybrid" in text


# ---- DOCX ----

def test_docx_headings_and_tables(corpus_available):
    parsed = parse_document(_entry_for("leave_policy_global_v3.docx"), CORPUS_DIR, None)
    headings = [s.heading for s in parsed.segments if s.heading]
    assert "1. What changes" in headings
    rows = [row for s in parsed.segments if s.kind == "table" for row in s.rows]
    assert any(row[0] == "Carry-forward limit" and row[1] == "10 days" and row[2] == "12 days" for row in rows)


def test_docx_paragraph_with_missing_style_is_kept_as_text(tmp_path):
    path = tmp_path / "orphan.docx"
    document = docx.Document()
    paragraph = document.add_paragraph("Orphan paragraph with a missing style.")
    pPr = paragraph._p.get_or_add_pPr()
    style = OxmlElement("w:pStyle")
    style.set(qn("w:val"), "NoSuchStyleXYZ")
    pPr.append(style)
    document.save(path)

    entry = ManifestEntry("POL-ORPHAN", "orphan.docx", "Orphan", "policy", ("GLOBAL",), date(2025, 1, 1), "all")
    parsed = parse_document(entry, tmp_path, None)
    texts = [s for s in parsed.segments if "Orphan paragraph" in s.text]
    assert len(texts) == 1
    assert texts[0].heading is None


# ---- Markdown ----

def test_markdown_headings_and_tables(corpus_available):
    parsed = parse_document(_entry_for("it_security_policy.md"), CORPUS_DIR, None)
    assert any(s.heading for s in parsed.segments)
    tables = [s for s in parsed.segments if s.kind == "table"]
    assert tables
    for table in tables:
        widths = {len(row) for row in table.rows}
        assert len(widths) == 1, "every row within one table should have the same number of cells"
    assert all("---" not in cell for t in tables for row in t.rows for cell in row)


# ---- HTML ----

def test_html_drops_footer_and_keeps_faq_text(corpus_available):
    parsed = parse_document(_entry_for("employee_leave_faq.html"), CORPUS_DIR, None)
    text = _all_text(parsed)
    assert "10 paid sick days" in text
    assert "· Internal" not in text
    assert "Veltrix Intranet" in text or "Employee Leave FAQ" in text


def test_html_table_rows_are_extracted(corpus_available):
    parsed = parse_document(_entry_for("holiday_calendar_us_2026.html"), CORPUS_DIR, None)
    rows = [row for s in parsed.segments if s.kind == "table" for row in s.rows]
    assert any(row[-1] == "Labor Day" for row in rows)
    assert any(s.heading == "Paid holidays 2026" for s in parsed.segments)


def test_unsupported_extension_is_rejected(tmp_path):
    (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
    entry = ManifestEntry("POL-X", "x.txt", "X", "policy", ("GLOBAL",), date(2025, 1, 1), "all")
    with pytest.raises(IngestionError, match="unsupported"):
        parse_document(entry, tmp_path, None)


def test_missing_file_is_rejected(tmp_path):
    entry = ManifestEntry("POL-NONE", "absent.md", "Absent", "policy", ("GLOBAL",), date(2025, 1, 1), "all")
    with pytest.raises(IngestionError, match="not found"):
        parse_document(entry, tmp_path, None)
