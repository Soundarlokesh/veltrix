"""Command line: python -m app.ingestion.cli [--force]"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import Settings
from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import run_ingestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the HR corpus and rebuild the vector index.")
    parser.add_argument("--force", action="store_true", help="rebuild the vector index even if nothing changed")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        report = run_ingestion(settings, force_rebuild=args.force)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"documents processed : {report.documents}")
    print(f"ocr documents       : {', '.join(report.ocr_doc_ids) or 'none'}")
    print(f"chunks              : {report.chunks}")
    print(f"corpus tokens       : {report.corpus_tokens}")
    print(f"holidays extracted  : {report.holidays}")
    print(f"index rebuilt       : {report.index_rebuilt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
