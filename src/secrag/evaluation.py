"""Evaluation harness.

1. make_xbrl_queries  - auto-generates labelled numeric questions from XBRL facts
                        ("What was Apple's net income for fiscal year 2024?"), with
                        the gold value and the strings the gold table chunk must contain.
2. eval_retrieval     - Recall@k (hit rate) and MRR@10 for bm25 / dense / hybrid
                        (/ hybrid+rerank). This is the ablation that justifies RRF.
3. eval_qa            - end-to-end, with vs. without the Verifier:
                          * numeric answer accuracy vs. XBRL gold (objective)
                          * verified-claim accuracy: share of final claims an
                            independent judge model finds supported by their citations
                          * abstention rate, latency
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from .agents.graph import AgentContext, Pipeline
from .agents.verifier import llm_verdicts
from .edgar.xbrl import CONCEPT_ORDER, FactStore
from .index.store import HybridIndex
from .numbers import Num, extract_numbers, is_grounded

QUESTION_LABELS = [
    "Revenue",
    "Net income",
    "Operating income",
    "Gross profit",
    "R&D expense",
    "EPS (diluted)",
    "Operating cash flow",
    "Total assets",
    "Cash and cash equivalents",
]
_PHRASES = {
    "Revenue": "total revenue (net sales)",
    "EPS (diluted)": "diluted earnings per share",
    "R&D expense": "research and development expense",
}


def load_queries(paths: list[str | Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith(("#", "//")):
                    out.append(json.loads(line))
    return out


def number_variants(value: float, unit: str) -> list[str]:
    v = abs(value)
    if "/" in unit:  # per-share
        return [f"{v:.2f}"]
    out = []
    for s in (1e6, 1e3):
        x = v / s
        if x >= 1000:
            out.append(f"{round(x):,}")
    return out


def make_xbrl_queries(index: HybridIndex, store: FactStore) -> list[dict]:
    out: list[dict] = []
    for f in index.catalog():
        if f["form"] != "10-K":
            continue
        facts = [
            x
            for x in store.load(f["ticker"])
            if x.accession == f["accession"] and x.end == f["report_date"] and x.period in ("FY", "instant")
        ]
        for label in QUESTION_LABELS:
            cands = [x for x in facts if x.label == label]
            if not cands:
                continue
            x = min(cands, key=lambda c: CONCEPT_ORDER.index(c.concept))
            variants = number_variants(x.value, x.unit)
            if not variants:
                continue
            phrase = _PHRASES.get(label, label.lower())
            fy = f["fiscal_year"]
            q = (
                f"What were {f['company']}'s {phrase} at the end of fiscal year {fy}?"
                if x.period == "instant"
                else f"What was {f['company']}'s {phrase} for fiscal year {fy}?"
            )
            out.append(
                {
                    "id": f"xbrl-{f['ticker']}-{fy}-{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}",
                    "type": "xbrl",
                    "question": q,
                    "ticker": f["ticker"],
                    "fiscal_year": fy,
                    "form": "10-K",
                    "gold_value": x.value,
                    "unit": x.unit,
                    "concept": x.concept,
                    "evidence_strings": variants,
                    "require_any": ["per share"] if "/" in x.unit else [],
                }
            )
    return out


# ------------------------------------------------------------------ synthetic text queries
TEXT_SECTIONS = ("Item 1 - Business", "Item 1A - Risk Factors", "Item 1C - Cybersecurity",
                 "Item 3 - Legal Proceedings", "Item 7 - MD&A")

QGEN_SYSTEM = """You write evaluation questions for a search system over SEC filings.
Given ONE passage from a company's 10-K, write ONE question that an analyst might ask and that THIS passage
answers specifically (not a generic question that many passages could answer).

Style "paraphrase": use your own words. Do NOT copy distinctive terms or phrases from the passage.
Style "keyword": you MAY use the passage's exact technical terms, names or acronyms, as an expert would.

Always name the company and the fiscal year in the question.
Also return "evidence": an EXACT quote of 4-10 consecutive words copied verbatim from the passage,
containing the key fact that answers the question.

Return JSON only: {"question": "...", "evidence": "..."}"""


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def make_text_queries(index: HybridIndex, llm, per_ticker: int = 5, min_words: int = 120,
                      seed: int = 0, log=print) -> list[dict]:
    """LLM-written questions over narrative 10-K sections, with automatic label checks.

    A generated question is kept only if its evidence quote is verbatim in the source chunk,
    is distinctive (appears in at most 2 chunks of that company/year), and, for paraphrase
    questions, is not leaked into the question text.
    """
    import random

    rng = random.Random(seed)
    latest_10k = {}
    for f in index.catalog():
        if f["form"] == "10-K":
            latest_10k[f["ticker"]] = max(latest_10k.get(f["ticker"], 0), f["fiscal_year"])
    out: list[dict] = []
    for ticker, fy in sorted(latest_10k.items()):
        same = [c for c in index.chunks if c.ticker == ticker and c.fiscal_year == fy]
        pool = [c for c in same if c.form == "10-K" and c.kind == "text"
                and c.section in TEXT_SECTIONS and len(c.text.split()) >= min_words]
        by_sec: dict[str, list] = {}
        for c in pool:
            by_sec.setdefault(c.section, []).append(c)
        for v in by_sec.values():
            rng.shuffle(v)
        order = []  # round-robin across sections so every section gets questions
        while any(by_sec.values()):
            for sec in list(by_sec):
                if by_sec[sec]:
                    order.append(by_sec[sec].pop())
        kept = 0
        for c in order[: per_ticker * 3]:
            if kept >= per_ticker:
                break
            style = "keyword" if kept % 2 else "paraphrase"
            try:
                data = llm.chat_json(QGEN_SYSTEM, f"STYLE: {style}\n\nPASSAGE ({c.header()}):\n{c.text}", max_tokens=200)
                question, evidence = str(data["question"]).strip(), str(data["evidence"]).strip().strip('"')
            except Exception as e:  # noqa: BLE001
                log(f"  skip {c.chunk_id}: {e}")
                continue
            ev = _norm(evidence)
            reason = None
            if len(ev.split()) < 3:
                reason = "evidence too short"
            elif ev not in _norm(c.text):
                reason = "evidence not verbatim in passage"
            elif sum(ev in _norm(x.text) for x in same) > 2:
                reason = "evidence not distinctive"
            elif style == "paraphrase" and ev in _norm(question):
                reason = "evidence leaked into question"
            elif not any(k and k in question.lower()
                         for k in (re.split(r"[^a-z]+", c.company.lower())[0], ticker.lower())):
                reason = "question does not name the company"
            if reason:
                log(f"  reject ({reason}): {question}")
                continue
            kept += 1
            out.append({
                "id": f"syn-{ticker}-{fy}-{kept}",
                "type": f"synthetic-{style}",
                "question": question,
                "ticker": ticker,
                "fiscal_year": fy,
                "form": "10-K",
                "evidence_strings": [evidence],
                "source_chunk": c.chunk_id,
                "section": c.section,
            })
            log(f"  + [{style}] {question}")
    return out


# ------------------------------------------------------------------ retrieval
def is_relevant(chunk, q: dict) -> bool:
    if q.get("ticker") and chunk.ticker != q["ticker"]:
        return False
    low = _norm(chunk.text)
    if q.get("require_any") and not any(_norm(r) in low for r in q["require_any"]):
        return False
    return any(_norm(s) in low for s in q.get("evidence_strings", []))


def _filters(q: dict, mode: str) -> dict | None:
    if mode == "none":
        return None
    f: dict = {"tickers": [q["ticker"]] if q.get("ticker") else None}
    if mode == "ticker_year":
        f["fiscal_year"] = q.get("fiscal_year")
    return f


def eval_retrieval(
    index: HybridIndex,
    queries: list[dict],
    modes: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    k: int = 5,
    filter_mode: str = "ticker",
    rerank: bool = False,
    n_candidates: int = 50,
    bm25_weights: tuple[float, ...] = (),
) -> dict:
    answerable = [q for q in queries if any(is_relevant(c, q) for c in index.chunks)]
    skipped = [q["id"] for q in queries if q not in answerable]
    runs = [(m, m, False, None) for m in modes]
    runs += [(f"hybrid (bm25 w={w:g})", "hybrid", False, w) for w in bm25_weights]
    if rerank:
        runs.append(("hybrid+rerank", "hybrid", True, None))
    results: dict[str, dict] = {}
    per_query: dict[str, list] = {}
    for name, mode, rr, w in runs:
        stats: dict[str, list[tuple[int, float]]] = {}
        rows = []
        for q in answerable:
            hits = index.search(q["question"], k=max(k, 10), mode=mode, filters=_filters(q, filter_mode),
                                n_candidates=n_candidates, rerank=rr, bm25_weight=w)
            ranks = [i for i, h in enumerate(hits) if is_relevant(h.chunk, q)]
            first = ranks[0] if ranks else None
            hit = int(first is not None and first < k)
            mrr = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
            for bucket in ("all", q.get("type", "manual")):
                stats.setdefault(bucket, []).append((hit, mrr))
            rows.append({"id": q["id"], "first_relevant_rank": first})
        results[name] = {
            b: {"recall@k": statistics.mean(h for h, _ in v), "mrr@10": statistics.mean(m for _, m in v), "n": len(v)}
            for b, v in stats.items()
        }
        per_query[name] = rows
    router_acc = None
    if "routed" in modes and answerable:
        from .index.router import classify

        router_acc = statistics.mean(
            (classify(q["question"]) == "numeric") == (q.get("type") == "xbrl") for q in answerable
        )
    return {
        "k": k,
        "router_accuracy": router_acc,
        "filters": filter_mode,
        "n_total": len(queries),
        "n_answerable": len(answerable),
        "skipped_not_in_index": skipped,
        "results": results,
        "per_query": per_query,
    }


# ------------------------------------------------------------------ QA
def answer_matches(answer: str, gold: float) -> bool:
    gold_num = Num(gold, 1.0, False, 0, "gold")
    text = re.sub(r"\[\d+\]", " ", answer)
    return any(is_grounded(n, [gold_num]) for n in extract_numbers(text) if not n.pct)


def eval_qa(
    ctx: AgentContext,
    queries: list[dict],
    configs: tuple[tuple[str, bool, bool], ...] = (
        ("no_verifier", False, False),
        ("verifier_no_xbrl", True, False),  # isolates the verifier from the XBRL answer key
        ("verifier_xbrl", True, True),
    ),
    judge=None,
    log=print,
    workers: int = 1,
) -> tuple[dict, list[dict]]:
    """configs: (name, use_verifier, use_xbrl_crosscheck).

    XBRL questions take their gold answer from XBRL, so a verifier that can see XBRL facts is
    partly looking at the answer key. verifier_no_xbrl measures the verifier without that help.

    workers > 1 runs questions concurrently (useful with vLLM, which batches parallel requests).
    """
    import dataclasses
    from concurrent.futures import ThreadPoolExecutor

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    facts = ctx.facts
    for name, verify, use_xbrl in configs:
        pipe = Pipeline(dataclasses.replace(ctx, facts=facts if use_xbrl else None), verify=verify)
        done = [0]

        def run_one(q: dict) -> dict | None:
            t0 = time.perf_counter()
            try:
                out = pipe.ask(q["question"], tickers=[q["ticker"]] if q.get("ticker") else None)
            except Exception as e:  # noqa: BLE001
                log(f"  [{name}] {q['id']}: ERROR {e}")
                return None
            latency = time.perf_counter() - t0
            kept = out.get("kept_claims") or []
            ok = None
            if q.get("gold_value") is not None:
                # score only the claims actually kept: notes like "(1 claim(s) removed...)" must never count
                ok = bool(kept) and answer_matches(" ".join(c["text"] for c in kept), q["gold_value"])
            evidence = out.get("evidence") or {}
            hit = None  # did the retrieved context contain the gold evidence at all?
            if q.get("evidence_strings"):
                hit = any(is_relevant(SimpleNamespace(ticker=c["ticker"], text=c["text"]), q)
                          for c in evidence.values())
            n_ok = None
            if judge is not None and kept:
                verdicts = llm_verdicts(judge, kept, evidence, "")
                n_ok = sum(verdicts.get(c["id"], {}).get("verdict") == "SUPPORTED" for c in kept)
            done[0] += 1
            log(f"  [{name}] {done[0]}/{len(queries)} {q['id']}: correct={ok} claims={len(kept)} "
                f"gold_in_evidence={hit} judged_ok={n_ok}")
            return {"config": name, "id": q["id"], "question": q["question"], "answer": out.get("answer"),
                    "gold_value": q.get("gold_value"), "numeric_correct": ok, "gold_in_evidence": hit,
                    "claims": len(kept), "judge_supported": n_ok,
                    "rounds": out.get("round", 0), "latency_s": round(latency, 2),
                    "kept_claims": [c.get("text") for c in kept],
                    "dropped_claims": [d.get("text") for d in out.get("dropped") or []],
                    "trace": out.get("trace", [])}

        t_start = time.perf_counter()
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run_one, queries))
        else:
            results = [run_one(q) for q in queries]
        wall = time.perf_counter() - t_start
        cfg_rows = [r for r in results if r is not None]
        rows.extend(cfg_rows)

        n = len(cfg_rows)
        numeric = [r for r in cfg_rows if r["numeric_correct"] is not None]
        answered = [r for r in numeric if r["claims"] > 0]
        correct = sum(r["numeric_correct"] for r in numeric)
        n_claims = sum(r["claims"] for r in cfg_rows)
        judged_ok = sum(r["judge_supported"] or 0 for r in cfg_rows)
        hits = [r["gold_in_evidence"] for r in cfg_rows if r["gold_in_evidence"] is not None]
        latencies = [r["latency_s"] for r in cfg_rows]
        summary[name] = {
            "n": n,
            "numeric_accuracy": correct / len(numeric) if numeric else None,
            # selective-answering view: how often it answers, and how often those answers are right
            "coverage": len(answered) / len(numeric) if numeric else None,
            "precision_when_answering": correct / len(answered) if answered else None,
            "confident_errors": (len(answered) - correct) / len(numeric) if numeric else None,
            "gold_in_evidence": statistics.mean(hits) if hits else None,
            # judge sees claims + citations but NOT the question: measures faithfulness, not correctness
            "claim_faithfulness": (judged_ok / n_claims) if (judge is not None and n_claims) else None,
            "claims_per_answer": n_claims / n if n else 0,
            "abstain_rate": sum(r["claims"] == 0 for r in cfg_rows) / n if n else 0,
            "avg_rounds": statistics.mean(r["rounds"] for r in cfg_rows) if cfg_rows else 0,
            "p50_latency_s": statistics.median(latencies) if latencies else None,
            "workers": workers,
            "wall_time_s": round(wall, 1),
            "questions_per_min": round(60 * n / wall, 2) if wall else None,
        }
    return summary, rows
