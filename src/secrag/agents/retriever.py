"""Retriever agent: runs pending sub-queries against the hybrid index with metadata filters."""
from __future__ import annotations

from ..index.store import rrf


def _mode_for(ctx, sub_query: str, question: str) -> str:
    """With routed retrieval, a sub-query is numeric if it or the user's question asks for a figure."""
    if ctx.retrieval_mode != "routed":
        return ctx.retrieval_mode
    from ..index.router import classify

    numeric = classify(sub_query) == "numeric" or classify(question) == "numeric"
    return "dense" if numeric else "hybrid"


def search_relaxing(ctx, query: str, filters: dict, mode: str | None = None) -> list:
    """Search with filters; if nothing matches, relax year, then form (never the user's ticker restriction)."""
    attempts = [dict(filters)]
    if filters.get("fiscal_year"):
        attempts.append({**filters, "fiscal_year": None})
    if filters.get("form"):
        attempts.append({**filters, "fiscal_year": None, "form": None})
    for f in attempts:
        hits = ctx.index.search(
            query,
            k=ctx.settings.top_k,
            mode=mode or ctx.retrieval_mode,
            filters=f,
            n_candidates=ctx.settings.n_candidates,
            rerank=ctx.rerank,
        )
        if hits:
            return hits
    return []


def make_retriever(ctx):
    def retriever(state):
        evidence = dict(state.get("evidence") or {})
        restrict = state.get("tickers") or None
        added = 0
        # RAG-Fusion: run every sub-query, then merge their rankings with RRF so chunks found by
        # several queries come first. The user's original question goes first, so it wins ties.
        pending = sorted(state.get("pending_queries") or [], key=lambda s: not s.get("original"))
        rankings: list[list[str]] = []
        found: dict[str, dict] = {}
        for sq in pending:
            filters = {
                "tickers": [sq["ticker"]] if sq.get("ticker") else restrict,
                "form": sq.get("form"),
                "fiscal_year": sq.get("fiscal_year"),
            }
            mode = _mode_for(ctx, sq["query"], state.get("question", ""))
            hits = search_relaxing(ctx, sq["query"], filters, mode)
            rankings.append([h.chunk.chunk_id for h in hits])
            for h in hits:
                found.setdefault(h.chunk.chunk_id, h.chunk.to_dict())
        for cid, _score in rrf(rankings, k=ctx.settings.rrf_k):
            if cid not in evidence:
                evidence[cid] = found[cid]
                added += 1

        # Cap the context: drop the oldest evidence that no SUPPORTED claim relies on.
        protected = {
            cid
            for v in state.get("verdicts") or []
            if v.get("verdict") == "SUPPORTED"
            for cid in v.get("citations", [])
        }
        items = list(evidence.items())
        while len(items) > ctx.max_evidence:
            drop = next((i for i, (cid, _) in enumerate(items) if cid not in protected), None)
            if drop is None:
                break
            items.pop(drop)
        evidence = dict(items)
        return {
            "evidence": evidence,
            "pending_queries": [],
            "trace": [f"retriever: +{added} new chunks, {len(evidence)} in context"],
        }

    return retriever
