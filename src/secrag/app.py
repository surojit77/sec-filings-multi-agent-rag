"""Wiring helpers: build index/LLM/context from Settings, and the ingestion loop."""
from __future__ import annotations

import os

from .agents.graph import AgentContext
from .config import Settings
from .edgar.client import EdgarClient
from .edgar.parser import parse_filing_html
from .edgar.xbrl import FactStore, extract_facts
from .index.chunker import chunk_blocks
from .index.embedder import HashEmbedder, SentenceTransformerEmbedder
from .index.store import CrossEncoderReranker, HybridIndex
from .llm import OpenAICompatLLM


def make_embedder(settings: Settings):
    if os.environ.get("EMBEDDER", "").lower() == "hash" or settings.embed_model == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embed_model, settings.embed_query_prefix)


def load_index(settings: Settings, rerank: bool = False) -> HybridIndex:
    reranker = CrossEncoderReranker(settings.rerank_model) if rerank else None
    return HybridIndex.load(settings.index_dir, make_embedder(settings), rrf_k=settings.rrf_k, reranker=reranker,
                            bm25_weight=settings.bm25_weight)


def make_llm(settings: Settings, judge: bool = False) -> OpenAICompatLLM:
    if judge:
        return OpenAICompatLLM(
            os.environ.get("JUDGE_BASE_URL", settings.llm_base_url),
            os.environ.get("JUDGE_API_KEY", settings.llm_api_key),
            os.environ.get("JUDGE_MODEL", settings.llm_model),
        )
    return OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)


def build_context(settings: Settings, mode: str = "hybrid", rerank: bool = False, llm=None, index=None,
                  use_xbrl: bool = True) -> AgentContext:
    return AgentContext(
        llm=llm or make_llm(settings),
        index=index or load_index(settings, rerank=rerank),
        facts=FactStore(settings.xbrl_dir) if use_xbrl else None,
        settings=settings,
        retrieval_mode=mode,
        rerank=rerank,
    )


def ingest(
    tickers: list[str],
    forms: tuple[str, ...],
    limit: int,
    settings: Settings,
    index: HybridIndex | None = None,
    client: EdgarClient | None = None,
    log=print,
) -> dict:
    index = index or load_index(settings)
    client = client or EdgarClient(settings.sec_user_agent, settings.raw_dir)
    facts_store = FactStore(settings.xbrl_dir)
    stats = {"filings": 0, "skipped": 0, "chunks": 0, "tables": 0, "facts": 0}
    for t in tickers:
        t = t.upper()
        for f in client.list_filings(t, forms, limit):
            if index.has_accession(f.accession):
                stats["skipped"] += 1
                continue
            blocks = parse_filing_html(client.fetch_document(f), f.form)
            chunks = chunk_blocks(blocks, f, settings.chunk_words, settings.chunk_overlap)
            n = index.add(chunks)
            n_tables = sum(c.kind == "table" for c in chunks)
            stats["filings"] += 1
            stats["chunks"] += n
            stats["tables"] += n_tables
            log(f"  {t} {f.form} {f.report_date}: {n} chunks ({n_tables} tables)")
        cik, _ = client.lookup(t)
        facts = extract_facts(t, client.company_facts(cik))
        facts_store.save(t, facts)
        stats["facts"] += len(facts)
        index.save()  # checkpoint after every company
        log(f"{t}: done ({len(facts)} XBRL facts)")
    return stats
