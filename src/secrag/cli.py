"""Command line interface:  python -m secrag <command> ...   (or `secrag <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import Settings


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def cmd_ingest(args, s: Settings) -> None:
    from .app import ingest

    stats = ingest(args.tickers, tuple(args.forms), args.limit, s)
    print(json.dumps(stats, indent=2))


def cmd_stats(args, s: Settings) -> None:
    from .app import load_index

    idx = load_index(s)
    print(f"{len(idx.chunks)} chunks ({sum(c.kind == 'table' for c in idx.chunks)} tables)")
    for f in idx.catalog():
        print(f"  {f['ticker']:6} {f['form']:5} FY{f['fiscal_year']}  {f['report_date']}  {f['accession']}")


def cmd_ask(args, s: Settings) -> None:
    from .agents.graph import Pipeline
    from .app import build_context

    if args.no_verify:
        s.max_verify_rounds = 0
    ctx = build_context(s, mode=args.mode, rerank=args.rerank, use_xbrl=not args.no_xbrl)
    out = Pipeline(ctx, verify=not args.no_verify).ask(args.question, tickers=args.tickers)
    if args.json:
        out.pop("evidence", None)
        print(json.dumps(out, indent=2, default=str))
        return
    print("\n" + out["answer"] + "\n")
    for c in out.get("citations", []):
        print(f"  [{c['n']}] {c['company']} {c['form']} (period ending {c['report_date']}) - {c['section']}\n      {c['url']}")
    if args.trace:
        print("\nTrace:")
        for t in out.get("trace", []):
            print("  " + t)
        for v in out.get("verdicts", []) or []:
            print(f"  {v['id']} {v['verdict']}: {v['text']}\n      reason: {v['reason']} | numeric: {v.get('numeric_check')}")


def cmd_search(args, s: Settings) -> None:
    from .app import load_index

    idx = load_index(s)
    filters = {"tickers": args.tickers, "form": args.form, "fiscal_year": args.year}
    for i, h in enumerate(idx.search(args.query, k=args.k, mode=args.mode, filters=filters), 1):
        c = h.chunk
        print(f"\n#{i} [{c.kind}] {c.header()}\n    id={c.chunk_id}")
        print("    " + c.text[: args.chars].replace("\n", "\n    "))


def cmd_make_queries(args, s: Settings) -> None:
    from .app import load_index
    from .edgar.xbrl import FactStore
    from .evaluation import make_xbrl_queries

    qs = make_xbrl_queries(load_index(s), FactStore(s.xbrl_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for q in qs:
            fh.write(json.dumps(q) + "\n")
    print(f"wrote {len(qs)} queries to {args.out}")


def cmd_make_text_queries(args, s: Settings) -> None:
    from .app import load_index, make_llm
    from .evaluation import make_text_queries

    qs = make_text_queries(load_index(s), make_llm(s), per_ticker=args.per_ticker, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for q in qs:
            fh.write(json.dumps(q) + "\n")
    print(f"\nwrote {len(qs)} queries to {args.out}  (spot-check ~10 of them before trusting the eval)")


def _save(name: str, payload) -> Path:
    out = Path("eval/results") / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def cmd_eval_retrieval(args, s: Settings) -> None:
    from .app import load_index
    from .evaluation import eval_retrieval, load_queries

    idx = load_index(s, rerank=args.rerank)
    res = eval_retrieval(idx, load_queries(args.queries), tuple(args.modes), k=args.k,
                         filter_mode=args.filters, rerank=args.rerank, n_candidates=s.n_candidates,
                         bm25_weights=tuple(args.bm25_weights or ()))
    print(f"\n{res['n_answerable']}/{res['n_total']} queries have gold evidence in the index "
          f"(filters={res['filters']}, k={res['k']})\n")
    buckets = sorted({b for r in res["results"].values() for b in r})
    print("| retriever | " + " | ".join(f"Recall@{args.k} ({b})" for b in buckets) + " | MRR@10 (all) |")
    print("|---" * (len(buckets) + 2) + "|")
    for name, r in res["results"].items():
        cells = " | ".join(f"{_pct(r[b]['recall@k'])} (n={r[b]['n']})" if b in r else "-" for b in buckets)
        print(f"| {name} | {cells} | {r['all']['mrr@10']:.3f} |")
    if res.get("router_accuracy") is not None:
        print(f"\nrouter: {_pct(res['router_accuracy'])} of questions sent to the intended retriever "
              "(numeric -> dense, narrative -> hybrid)")
    print(f"\nsaved {_save('retrieval', res)}")


def cmd_eval_qa(args, s: Settings) -> None:
    from .app import build_context, make_llm
    from .evaluation import eval_qa, load_queries

    queries = load_queries(args.queries)
    if args.sample:  # random sample across all companies (reproducible via --seed)
        import random

        queries = random.Random(args.seed).sample(queries, min(args.sample, len(queries)))
    if args.limit:
        queries = queries[: args.limit]
    ctx = build_context(s, mode=args.mode, rerank=args.rerank)
    judge = None if args.no_judge else make_llm(s, judge=True)
    all_cfgs = {"no_verifier": False, "verifier_no_xbrl": True, "verifier_xbrl": True}
    cfgs = tuple((n, all_cfgs[n], n == "verifier_xbrl") for n in args.configs)
    summary, rows = eval_qa(ctx, queries, configs=cfgs, judge=judge, workers=args.workers)
    print("\n| config | accuracy | coverage | precision when answering | confident errors | gold in context "
          "| claim faithfulness | avg rounds | p50 latency |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in summary.items():
        print(f"| {name} | {_pct(r['numeric_accuracy'])} | {_pct(r['coverage'])} | "
              f"{_pct(r['precision_when_answering'])} | {_pct(r['confident_errors'])} | {_pct(r['gold_in_evidence'])} | "
              f"{_pct(r['claim_faithfulness'])} | {r['avg_rounds']:.2f} | {r['p50_latency_s'] or 0:.1f}s |")
    print()
    for name, r in summary.items():
        print(f"{name}: {r['n']} questions in {r['wall_time_s'] / 60:.1f} min with {r['workers']} worker(s) "
              f"= {r['questions_per_min']} questions/min")
    print(f"\nsaved {_save('qa', {'summary': summary, 'rows': rows})}")


def cmd_qa_failures(args, s: Settings) -> None:
    import glob

    path = args.file or sorted(glob.glob("eval/results/qa-*.json"))[-1]
    rows = json.load(open(path))["rows"]
    rows = [r for r in rows if r.get("numeric_correct") is False and (not args.config or r["config"] == args.config)]
    buckets = {"abstained": 0, "retrieval miss (gold not in context)": 0, "wrong answer despite gold in context": 0}
    for r in rows:
        if r["claims"] == 0:
            kind = "abstained"
        elif r.get("gold_in_evidence") is False:
            kind = "retrieval miss (gold not in context)"
        else:
            kind = "wrong answer despite gold in context"
        buckets[kind] += 1
        if args.verbose or kind != "abstained":
            gold = r.get("gold_value")
            print(f"[{r['config']}] {r['id']}  ({kind})  gold={gold:,.0f}" if gold is not None else f"[{r['config']}] {r['id']}")
            print("   " + (r["answer"] or "")[:300].replace("\n", " "))
            for d in r.get("dropped_claims") or []:
                print(f"   dropped claim: {d[:200]}")
            print()
    print(f"\n{path}")
    for k, v in buckets.items():
        print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="secrag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ingest", help="download + parse + index filings from EDGAR")
    a.add_argument("--tickers", nargs="+", required=True)
    a.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    a.add_argument("--limit", type=int, default=2, help="filings per form per company")
    a.set_defaults(fn=cmd_ingest)

    a = sub.add_parser("stats", help="show what is in the index")
    a.set_defaults(fn=cmd_stats)

    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--tickers", nargs="+")
    a.add_argument("--mode", choices=["bm25", "dense", "hybrid", "routed"], default="routed")
    a.add_argument("--rerank", action="store_true")
    a.add_argument("--no-verify", action="store_true")
    a.add_argument("--no-xbrl", action="store_true", help="verifier without the XBRL cross-check")
    a.add_argument("--trace", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    a = sub.add_parser("search", help="show raw retrieval results (debugging / writing eval questions)")
    a.add_argument("query")
    a.add_argument("--tickers", nargs="+")
    a.add_argument("--form", choices=["10-K", "10-Q"])
    a.add_argument("--year", type=int)
    a.add_argument("--mode", choices=["bm25", "dense", "hybrid", "routed"], default="dense")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--chars", type=int, default=400)
    a.set_defaults(fn=cmd_search)

    a = sub.add_parser("make-queries", help="auto-generate labelled numeric eval queries from XBRL")
    a.add_argument("--out", default="eval/queries_xbrl.jsonl")
    a.set_defaults(fn=cmd_make_queries)

    a = sub.add_parser("make-text-queries", help="LLM-generated, auto-checked questions over 10-K narrative sections")
    a.add_argument("--out", default="eval/queries_synthetic.jsonl")
    a.add_argument("--per-ticker", type=int, default=5)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_make_text_queries)

    a = sub.add_parser("eval-retrieval", help="Recall@k / MRR ablation: bm25 vs dense vs hybrid")
    a.add_argument("--queries", nargs="+", required=True)
    a.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid", "routed"])
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--filters", choices=["none", "ticker", "ticker_year"], default="ticker")
    a.add_argument("--rerank", action="store_true", help="also evaluate hybrid + cross-encoder rerank")
    a.add_argument("--bm25-weights", nargs="+", type=float,
                   help="extra hybrid runs with these BM25 weights in weighted RRF (dense weight = 1)")
    a.set_defaults(fn=cmd_eval_retrieval)

    a = sub.add_parser("eval-qa", help="end-to-end QA with vs without the Verifier")
    a.add_argument("--queries", nargs="+", required=True)
    a.add_argument("--limit", type=int)
    a.add_argument("--sample", type=int, help="random sample of N questions (across all companies)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--mode", choices=["bm25", "dense", "hybrid", "routed"], default="routed")
    a.add_argument("--rerank", action="store_true")
    a.add_argument("--no-judge", action="store_true")
    a.add_argument("--workers", type=int, default=1, help="questions answered concurrently (try 8-16 with vLLM)")
    a.add_argument("--configs", nargs="+", choices=["no_verifier", "verifier_no_xbrl", "verifier_xbrl"],
                   default=["no_verifier", "verifier_no_xbrl", "verifier_xbrl"])
    a.set_defaults(fn=cmd_eval_qa)

    a = sub.add_parser("qa-failures", help="classify wrong answers from the latest eval-qa run")
    a.add_argument("--file")
    a.add_argument("--config", choices=["no_verifier", "verifier_no_xbrl", "verifier_xbrl"])
    a.add_argument("--verbose", action="store_true", help="also list abstentions")
    a.set_defaults(fn=cmd_qa_failures)

    args = p.parse_args(argv)
    args.fn(args, Settings())


if __name__ == "__main__":
    main(sys.argv[1:])
