"""Test doubles. HashingEmbedder is deterministic and needs no model download, so index
logic can be tested everywhere. Real-model tests use SentenceTransformerEmbedder."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFEST_PATH = DATA_DIR / "manifest.json"
ORIGINAL_DB = DATA_DIR / "veltrix.original.db"
MEMO_FILE = "memo_hybrid_work_update.pdf"


class HashingEmbedder:
    """Bag-of-words embedding with stable hashing (unlike Python's salted hash())."""

    name = "test-hashing-embedder"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[row, bucket] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


def tesseract_available() -> bool:
    try:
        from app.ingestion.ocr import require_tesseract
        from app.config import Settings

        require_tesseract(Settings.from_env().tesseract_cmd)
        return True
    except Exception:
        return False
