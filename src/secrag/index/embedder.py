"""Embedding backends. Both return L2-normalised float32 matrices."""
from __future__ import annotations

import threading
import zlib

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()  # one encode at a time: safe under concurrent eval workers

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Dense retrieval needs sentence-transformers: pip install -e '.[dense]' "
                    "(or set EMBEDDER=hash for a no-download smoke test)"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        with self._lock:
            vecs = self._encode(texts)
        return np.asarray(vecs, dtype=np.float32)

    def _encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )


class HashEmbedder:
    """Bag-of-words hashing 'embedding'. No downloads; for tests and smoke runs only."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hash-{self.dim}"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        from .store import tokenize

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
