"""Hybrid index: BM25 (lexical) + dense (semantic), fused with Reciprocal Rank Fusion.

RRF score(d) = sum over rankers r of 1 / (k + rank_r(d)),  k = 60 by default.
It uses only ranks, so BM25 scores and cosine similarities (different scales)
never need normalising. Optional cross-encoder reranking on the fused top-N.
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what "
    "which with how did does do during their our we".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def rrf(rank_lists: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Weighted RRF: score(d) = sum_r w_r / (k + rank_r(d)). Equal weights = classic RRF."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rank_lists):
        for rank, idx in enumerate(ranking):
            scores[idx] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


@dataclass
class Hit:
    chunk: Chunk
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return np.asarray(self._model.predict([(query, t) for t in texts]))


class HybridIndex:
    def __init__(self, index_dir: Path | None, embedder, rrf_k: int = 60, reranker=None, bm25_weight: float = 1.0):
        self.index_dir = Path(index_dir) if index_dir else None
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.reranker = reranker
        self.chunks: list[Chunk] = []
        self.emb: np.ndarray | None = None
        self._ids: set[str] = set()
        self._accessions: set[str] = set()
        self._bm25 = None
        self._meta_arrays: dict[str, np.ndarray] | None = None
        self._lock = threading.Lock()  # guards lazy BM25 / metadata builds under concurrent queries

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, index_dir: Path, embedder, **kw) -> "HybridIndex":
        idx = cls(index_dir, embedder, **kw)
        d = Path(index_dir)
        if (d / "chunks.jsonl").exists():
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            if meta.get("embedder") and meta["embedder"] != embedder.name:
                raise ValueError(
                    f"Index was built with embedder '{meta['embedder']}' but '{embedder.name}' is configured. "
                    "Use the same EMBED_MODEL or delete data/index and re-ingest."
                )
            with open(d / "chunks.jsonl", encoding="utf-8") as fh:
                idx.chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
            idx.emb = np.load(d / "embeddings.npy")
            idx._ids = {c.chunk_id for c in idx.chunks}
            idx._accessions = {c.accession for c in idx.chunks}
        return idx

    def save(self) -> None:
        if self.index_dir is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict()) + "\n")
        if self.emb is not None:
            np.save(self.index_dir / "embeddings.npy", self.emb)
        (self.index_dir / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}))

    # ------------------------------------------------------------ building
    def add(self, chunks: list[Chunk]) -> int:
        new = [c for c in chunks if c.chunk_id not in self._ids]
        if not new:
            return 0
        vecs = self.embedder.encode([c.indexed_text() for c in new])
        self.emb = vecs if self.emb is None else np.vstack([self.emb, vecs])
        self.chunks.extend(new)
        self._ids.update(c.chunk_id for c in new)
        self._accessions.update(c.accession for c in new)
        self._bm25 = None
        self._meta_arrays = None
        return len(new)

    def has_accession(self, accession: str) -> bool:
        return accession in self._accessions

    def catalog(self) -> list[dict]:
        """One entry per ingested filing (for the Planner and the eval generator)."""
        seen: dict[str, dict] = {}
        for c in self.chunks:
            if c.accession not in seen:
                seen[c.accession] = {
                    "ticker": c.ticker,
                    "company": c.company,
                    "form": c.form,
                    "fiscal_year": c.fiscal_year,
                    "report_date": c.report_date,
                    "accession": c.accession,
                }
        return sorted(seen.values(), key=lambda f: (f["ticker"], f["report_date"]), reverse=False)

    def _bm25_model(self):
        with self._lock:
            if self._bm25 is None:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi([tokenize(c.indexed_text()) for c in self.chunks])
            return self._bm25

    def _arrays(self) -> dict[str, np.ndarray]:
        with self._lock:
            return self._build_arrays()

    def _build_arrays(self) -> dict[str, np.ndarray]:
        if self._meta_arrays is None:
            self._meta_arrays = {
                "ticker": np.array([c.ticker for c in self.chunks]),
                "form": np.array([c.form for c in self.chunks]),
                "year": np.array([c.fiscal_year for c in self.chunks]),
            }
        return self._meta_arrays

    # ------------------------------------------------------------ search
    def _mask(self, filters: dict | None) -> np.ndarray:
        n = len(self.chunks)
        mask = np.ones(n, dtype=bool)
        if not filters or n == 0:
            return mask
        a = self._arrays()
        tickers = filters.get("tickers")
        if tickers:
            mask &= np.isin(a["ticker"], [t.upper() for t in tickers])
        if filters.get("form"):
            mask &= a["form"] == filters["form"]
        fy = filters.get("fiscal_year")
        if fy:
            mask &= np.isin(a["year"], fy if isinstance(fy, (list, tuple, set)) else [fy])
        return mask

    def _bm25_rank(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        scores = np.asarray(self._bm25_model().get_scores(tokenize(query)), dtype=float)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_rank(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self.emb @ q
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:n]
        return [int(i) for i in order if np.isfinite(sims[i])]

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        filters: dict | None = None,
        n_candidates: int = 50,
        rerank: bool = False,
        bm25_weight: float | None = None,
    ) -> list[Hit]:
        if not self.chunks:
            return []
        mask = self._mask(filters)
        if not mask.any():
            return []
        if mode == "routed":
            from .router import route

            mode = route(query)
        if mode == "bm25":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._bm25_rank(query, mask, n_candidates))]
        elif mode == "dense":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._dense_rank(query, mask, n_candidates))]
        elif mode == "hybrid":
            ranked = rrf(
                [self._bm25_rank(query, mask, n_candidates), self._dense_rank(query, mask, n_candidates)],
                self.rrf_k,
                weights=[self.bm25_weight if bm25_weight is None else bm25_weight, 1.0],
            )
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        if rerank and self.reranker is not None and ranked:
            top = ranked[: max(20, k)]
            scores = self.reranker.score(query, [self.chunks[i].indexed_text() for i, _ in top])
            ranked = sorted(zip([i for i, _ in top], scores.tolist()), key=lambda x: -x[1])
        return [Hit(self.chunks[i], float(s)) for i, s in ranked[:k]]
