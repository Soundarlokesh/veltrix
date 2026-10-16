PYTHON ?= python

.PHONY: test run ingest

test:
	$(PYTHON) -m pytest

ingest:
	$(PYTHON) -m app.ingestion.cli

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000
