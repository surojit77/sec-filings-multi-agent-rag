"""Planner agent: decomposes the question into filtered sub-queries."""
from __future__ import annotations

import re
from collections import defaultdict

from .prompts import PLANNER_SYSTEM, PLANNER_USER

MAX_SUBQUERIES = 4


def render_catalog(catalog: list[dict], tickers: list[str] | None = None) -> str:
    by_t: dict[str, list[dict]] = defaultdict(list)
    names: dict[str, str] = {}
    for f in catalog:
        if tickers and f["ticker"] not in tickers:
            continue
        by_t[f["ticker"]].append(f)
        names[f["ticker"]] = f["company"]
    lines = []
    for t, fs in sorted(by_t.items()):
        parts = ", ".join(f"{f['form']} FY{f['fiscal_year']} (period ending {f['report_date']})" for f in fs)
        lines.append(f"- {t} ({names[t]}): {parts}")
    return "\n".join(lines) or "(no filings ingested)"


def validate_sub_queries(raw, allowed_tickers: set[str]) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for sq in raw:
        if not isinstance(sq, dict):
            continue
        query = str(sq.get("query") or "").strip()
        if not query:
            continue
        ticker = str(sq.get("ticker") or "").upper() or None
        if ticker not in allowed_tickers:
            ticker = None
        form = sq.get("form") if sq.get("form") in ("10-K", "10-Q") else None
        try:
            fy = int(sq.get("fiscal_year")) if sq.get("fiscal_year") is not None else None
        except (TypeError, ValueError):
            fy = None
        if fy is not None and not 1990 <= fy <= 2100:
            fy = None
        out.append({"query": query, "ticker": ticker, "form": form, "fiscal_year": fy})
    return out[:MAX_SUBQUERIES]


def fallback_plan(question: str, catalog: list[dict]) -> list[dict]:
    """No-LLM plan: one query per company mentioned (by ticker or name), with the year if stated."""
    q = question.lower()
    words = set(re.findall(r"[a-z0-9.&]+", q))
    tickers = []
    for f in catalog:
        name_key = re.split(r"[^a-z]+", f["company"].lower())[0]
        if f["ticker"] not in tickers and (f["ticker"].lower() in words or (name_key and name_key in words)):
            tickers.append(f["ticker"])
    m = re.search(r"\b(?:fy|fiscal(?: year)?)\s*'?(\d{4})\b", q) or re.search(r"\b(20\d{2})\b", q)
    fy = int(m.group(1)) if m else None
    form = "10-Q" if re.search(r"\b(quarter|q[1-4]|10-q)\b", q) else ("10-K" if fy or "annual" in q else None)
    return [{"query": question, "ticker": t, "form": form, "fiscal_year": fy} for t in tickers[:MAX_SUBQUERIES]] or [
        {"query": question, "ticker": None, "form": form, "fiscal_year": fy}
    ]


def with_original_question(question: str, subs: list[dict], max_total: int = 6) -> list[dict]:
    """Multi-query retrieval: also search with the user's own question, under each filter set the plan uses.

    Eval finding: the Planner tends to compress questions into terse keyword queries ("net income"),
    which retrieve far worse with dense embeddings than the full natural-language question. Adding the
    original question can only add candidates, never remove the Planner's.
    """
    out = list(subs)
    seen = {(s["query"].strip().lower(), s["ticker"], s["form"], s["fiscal_year"]) for s in subs}
    filter_sets = []
    for s in subs:
        fs = (s["ticker"], s["form"], s["fiscal_year"])
        if fs not in filter_sets:
            filter_sets.append(fs)
    for ticker, form, fy in filter_sets:
        key = (question.strip().lower(), ticker, form, fy)
        if key not in seen and len(out) < max_total:
            seen.add(key)
            out.append({"query": question, "ticker": ticker, "form": form, "fiscal_year": fy, "original": True})
    return out


def make_planner(ctx):
    def planner(state):
        question = state["question"]
        restrict = state.get("tickers") or None
        catalog = ctx.index.catalog()
        allowed = {f["ticker"] for f in catalog if not restrict or f["ticker"] in restrict}
        note = ""
        try:
            data = ctx.llm.chat_json(
                PLANNER_SYSTEM,
                PLANNER_USER.format(catalog=render_catalog(catalog, restrict), question=question),
                max_tokens=400,
            )
            subs = validate_sub_queries((data or {}).get("sub_queries"), allowed)
        except Exception as e:  # noqa: BLE001 - degrade gracefully to a rule-based plan
            subs, note = [], f" (LLM planner failed, used rule-based plan: {str(e)[:80]})"
        if not subs:
            subs = fallback_plan(question, [f for f in catalog if f["ticker"] in allowed])
        subs = with_original_question(question, subs)
        desc = "; ".join(
            f"'{s['query']}' [{s['ticker'] or '*'} {s['form'] or ''} {s['fiscal_year'] or ''}]".replace("  ", " ")
            for s in subs
        )
        return {
            "sub_queries": subs,
            "pending_queries": subs,
            "evidence": {},
            "round": 0,
            "feedback": "",
            "trace": [f"planner: {len(subs)} sub-queries: {desc}{note}"],
        }

    return planner
