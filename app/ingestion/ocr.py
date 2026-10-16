"""OCR for scanned PDF pages: PyMuPDF renders the page, Tesseract reads it.

A page is treated as scanned when its text layer is (almost) empty but it contains
an image. OCR never fails silently: missing Tesseract raises OCRUnavailableError and
empty OCR output raises IngestionError.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from app.ingestion.errors import OCRUnavailableError

SCANNED_MIN_CHARS = 25
OCR_DPI = 300
OCR_LANG = "eng"

INSTALL_HELP = (
    "Tesseract OCR is needed to read scanned PDFs but was not found. "
    "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (run the installer as "
    "administrator), then set TESSERACT_CMD in .env to the full path of tesseract.exe, for example "
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe. "
    "Linux or Docker: apt-get install tesseract-ocr tesseract-ocr-eng."
)


def is_scanned_page(page: fitz.Page) -> bool:
    has_text = len(page.get_text().strip()) >= SCANNED_MIN_CHARS
    return not has_text and bool(page.get_images(full=True))


def detect_scanned_pages(pdf_path: Path) -> list[int]:
    """1-based page numbers that have no text layer but contain an image."""
    with fitz.open(pdf_path) as pdf:
        return [number for number, page in enumerate(pdf, start=1) if is_scanned_page(page)]


def require_tesseract(cmd: str | None) -> None:
    """Raise OCRUnavailableError with install instructions unless Tesseract works."""
    if cmd:
        if not Path(cmd).is_file():
            raise OCRUnavailableError(f"TESSERACT_CMD points to a file that does not exist: {cmd}. {INSTALL_HELP}")
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages(config="")
    except Exception as exc:  # pytesseract raises several exception types when the binary is missing
        raise OCRUnavailableError(INSTALL_HELP) from exc
    if OCR_LANG not in languages:
        raise OCRUnavailableError(f"Tesseract is installed but the '{OCR_LANG}' language data is missing. {INSTALL_HELP}")


def ocr_page_text(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image, lang=OCR_LANG)
