"""End-to-end ingestion: manifest -> parse -> chunk -> holidays -> store -> vector index.

Any failure raises IngestionError (or a subclass). Nothing is reported as ready unless
the whole run succeeds, so /health must stay 503 when this raises.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import chunk_document
from app.ingestion.errors import IngestionError
from app.ingestion.holidays import extract_holidays
from app.ingestion.manifest import load_manifest
from app.ingestion.models import ManifestEntry, ParsedDocument
from app.ingestion.parsers import parse_document
from app.ingestion.store import write_store
from app.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from app.retrieval.vector_index import PARSER_VERSION, VectorIndex, load_or_build

logger = logging.getLogger(__name__)
STORE_FILE = "ingest.db"


@dataclass(frozen=True)
class IngestionReport:
    documents: int
    ocr_doc_ids: tuple[str, ...]
    chunks: int
    corpus_tokens: int
    holidays: int
    index_rebuilt: bool
    index: VectorIndex


def parse_all(entries: list[ManifestEntry], corpus_dir: Path, ocr_cmd: str | None) -> list[ParsedDocument]:
    documents: list[ParsedDocument] = []
    for entry in entries:
        parsed = parse_document(entry, corpus_dir, ocr_cmd)
        logger.info(
            "parsed doc_id=%s format=%s pages=%s segments=%d ocr_pages=%s",
            entry.doc_id,
            Path(entry.file).suffix.lower(),
            parsed.page_count,
            len(parsed.segments),
            list(parsed.ocr_pages),
        )
        documents.append(parsed)
    return documents


def run_ingestion(
    settings: Settings,
    embedder: Embedder | None = None,
    force_rebuild: bool = False,
    manifest_path: Path | None = None,
) -> IngestionReport:
    entries = load_manifest(manifest_path or settings.manifest_path)
    documents = parse_all(entries, settings.corpus_dir, settings.tesseract_cmd)

    chunks = []
    holidays = []
    for document in documents:
        chunks.extend(chunk_document(document))
        holidays.extend(extract_holidays(document))
    chunk_counts = Counter(chunk.doc_id for chunk in chunks)
    corpus_tokens = sum(chunk.tokens for chunk in chunks)
    logger.info("corpus chunks=%d corpus_tokens=%d holidays=%d", len(chunks), corpus_tokens, len(holidays))

    store_meta = {"parser": PARSER_VERSION, "documents": str(len(documents)), "chunks": str(len(chunks))}
    write_store(settings.vector_dir / STORE_FILE, documents, chunk_counts, holidays, store_meta)

    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    index, rebuilt = load_or_build(chunks, embedder, settings.vector_dir, force=force_rebuild)
    logger.info("vector index size=%d rebuilt=%s", index.size, rebuilt)

    return IngestionReport(
        documents=len(documents),
        ocr_doc_ids=tuple(doc.entry.doc_id for doc in documents if doc.ocr_pages),
        chunks=len(chunks),
        corpus_tokens=corpus_tokens,
        holidays=len(holidays),
        index_rebuilt=rebuilt,
        index=index,
    )


__all__ = ["IngestionError", "IngestionReport", "parse_all", "run_ingestion"]
