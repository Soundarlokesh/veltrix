"""Local FAISS vector index with a JSON-lines chunk store and a fingerprint.

The fingerprint covers every chunk's text and metadata, the embedder name and the parser
and chunking versions. If any of them change, the index is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from app.ingestion.chunking import MAX_CHUNK_TOKENS
from app.ingestion.models import Chunk
from app.retrieval.embeddings import Embedder

PARSER_VERSION = "phase-b-1"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


def compute_fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    payload = {
        "chunks": [chunk.to_dict() for chunk in chunks],
        "embedder": embedder_name,
        "parser": PARSER_VERSION,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VectorIndex:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], fingerprint: str) -> None:
        self.index = index
        self.chunks = chunks
        self.fingerprint = fingerprint

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(
        self,
        query: np.ndarray,
        k: int,
        allow: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[float, Chunk]]:
        """Top-k chunks by cosine similarity. `allow` is applied before ranking, so a
        filtered-out chunk can never occupy one of the k slots."""
        if self.index.ntotal == 0 or k <= 0:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(vector, self.index.ntotal)
        results: list[tuple[float, Chunk]] = []
        for score, position in zip(scores[0], ids[0]):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            if allow is not None and not allow(chunk):
                continue
            results.append((float(score), chunk))
            if len(results) == k:
                break
        return results

    def search_text(
        self,
        text: str,
        embedder: Embedder,
        k: int,
        allow: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[float, Chunk]]:
        return self.search(embedder.encode([text])[0], k, allow)

    def save(self, directory: Path, embedder_name: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta = {
            "fingerprint": self.fingerprint,
            "embedder": embedder_name,
            "parser": PARSER_VERSION,
            "chunk_count": len(self.chunks),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, fingerprint: str) -> "VectorIndex | None":
        """Return the saved index only if it matches `fingerprint`; otherwise None."""
        directory = Path(directory)
        paths = [directory / name for name in (INDEX_FILE, CHUNKS_FILE, META_FILE)]
        if not all(path.is_file() for path in paths):
            return None
        meta = json.loads(paths[2].read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            return None
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in paths[1].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(faiss.read_index(str(paths[0])), chunks, fingerprint)


def build_index(chunks: list[Chunk], embedder: Embedder, fingerprint: str) -> VectorIndex:
    index = faiss.IndexFlatIP(embedder.dim)
    if chunks:
        vectors = embedder.encode([chunk.text for chunk in chunks])
        if vectors.shape != (len(chunks), embedder.dim):
            raise ValueError(f"embedder returned shape {vectors.shape}, expected {(len(chunks), embedder.dim)}")
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return VectorIndex(index, list(chunks), fingerprint)


def load_or_build(
    chunks: list[Chunk],
    embedder: Embedder,
    directory: Path,
    force: bool = False,
) -> tuple[VectorIndex, bool]:
    """Reuse the saved index when nothing changed. Returns (index, rebuilt)."""
    fingerprint = compute_fingerprint(chunks, embedder.name)
    if not force:
        existing = VectorIndex.load(directory, fingerprint)
        if existing is not None:
            return existing, False
    index = build_index(chunks, embedder, fingerprint)
    index.save(directory, embedder.name)
    return index, True
