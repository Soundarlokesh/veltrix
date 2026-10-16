from collections import Counter
from datetime import date

from app.ingestion.holidays import classify, extract_holidays, parse_calendar_date
from app.ingestion.manifest import load_manifest
from app.ingestion.models import ManifestEntry, ParsedDocument, Segment
from app.ingestion.parsers import parse_document
from tests.helpers import CORPUS_DIR, MANIFEST_PATH

# Expected counts were read by hand from the calendar documents, not taken from code output.
EXPECTED_2026 = {
    "CAL-HOL-IN-2026": Counter({"public": 12, "restricted": 7}),
    "CAL-HOL-US-2026": Counter({"public": 10}),
    "CAL-HOL-DE-2026": Counter({"public": 10, "company": 2}),
}


def _calendar_doc(doc_id: str) -> ParsedDocument:
    entry = next(e for e in load_manifest(MANIFEST_PATH) if e.doc_id == doc_id)
    return parse_document(entry, CORPUS_DIR, None)


def test_supplied_calendars_produce_expected_counts(corpus_available):
    for doc_id, expected in EXPECTED_2026.items():
        holidays = extract_holidays(_calendar_doc(doc_id))
        assert Counter(h.category for h in holidays) == expected, doc_id


def test_restricted_holidays_are_not_public(corpus_available):
    holidays = extract_holidays(_calendar_doc("CAL-HOL-IN-2026"))
    restricted_names = {h.name for h in holidays if h.category == "restricted"}
    public_names = {h.name for h in holidays if h.category == "public"}
    assert "Holi" in restricted_names
    assert "Republic Day" in public_names
    assert restricted_names.isdisjoint(public_names)


def test_company_holidays_are_flagged(corpus_available):
    holidays = extract_holidays(_calendar_doc("CAL-HOL-DE-2026"))
    company = {h.holiday_date for h in holidays if h.category == "company"}
    assert company == {date(2026, 12, 24), date(2026, 12, 31)}


def test_dates_agree_with_the_weekday_column(corpus_available):
    for doc_id in EXPECTED_2026:
        for holiday in extract_holidays(_calendar_doc(doc_id)):
            assert holiday.weekday_matches, f"{doc_id}: {holiday.name} {holiday.holiday_date}"


def test_india_2025_calendar_is_a_separate_year(corpus_available):
    holidays = extract_holidays(_calendar_doc("CAL-HOL-IN-2025"))
    assert holidays
    assert {h.holiday_date.year for h in holidays} == {2025}


def test_non_calendar_documents_yield_no_holidays(corpus_available):
    entry = next(e for e in load_manifest(MANIFEST_PATH) if e.doc_id == "POL-LEAVE-GLOBAL-V2")
    assert extract_holidays(parse_document(entry, CORPUS_DIR, None)) == []


def test_parse_calendar_date_rejects_non_dates():
    assert parse_calendar_date("15 January 2026") == date(2026, 1, 15)
    assert parse_calendar_date("Effective date") is None
    assert parse_calendar_date("2026-01-15") is None


def test_classification_uses_heading_and_name_only():
    assert classify("2. Restricted holidays 2026", "Holi") == "restricted"
    assert classify("1. Public holidays 2026", "Heiligabend (company holiday)") == "company"
    assert classify("1. Public holidays 2026", "Republic Day") == "public"


def test_metadata_rows_without_weekday_are_ignored():
    entry = ManifestEntry("CAL-SYN", "x.md", "Syn", "calendar", ("IN",), date(2026, 1, 1), "all")
    rows = (("Effective date", "1 January 2026"), ("Date", "Day", "Holiday"), ("2 March 2026", "Monday", "Test Day"))
    doc = ParsedDocument(entry, "0" * 64, (Segment("table", "x", "1. Public", None, rows),), None, ())
    holidays = extract_holidays(doc)
    assert [(h.holiday_date, h.name) for h in holidays] == [(date(2026, 3, 2), "Test Day")]
