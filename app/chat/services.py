"""The objects one running service shares: database paths, the vector index, the embedder and
the chat provider. Built once at startup, after ingestion has finished."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Services:
    settings: object  # app.config.Settings
    index: object     # app.retrieval.vector_index.VectorIndex
    embedder: object  # app.retrieval.embeddings.Embedder
    provider: object  # app.llm.provider.ChatProvider
    db_path: Path
    store_path: Path
    state_db: Path
