"""Local CPU embeddings. Any embedder with name, dim and encode() works; the default is a
sentence-transformers model loaded lazily so importing the app never downloads anything."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)
