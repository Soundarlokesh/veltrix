class IngestionError(Exception):
    """Ingestion failed. The service must stay not-ready (/health 503) when raised."""


class ManifestError(IngestionError):
    """manifest.json is missing, malformed or inconsistent with the corpus."""


class OCRUnavailableError(IngestionError):
    """A scanned document needs OCR but Tesseract is not installed or not configured."""
