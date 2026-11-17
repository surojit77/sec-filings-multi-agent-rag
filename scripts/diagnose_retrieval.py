"""Why did the pipeline miss the gold chunk? Re-run retrieval offline (no LLM) for each miss in the
latest eval-qa results, using (a) the Planner's sub-queries + filters from the trace and
(b) the original question with the eval's ticker+year filters.

Usage:  python scripts/diagnose_retrieval.py [eval/results/qa-XXXX.json]
"""
import glob
import json
import re
import sys

from secrag.app import load_index
from secrag.config import Settings
from secrag.evaluation import is_relevant

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("eval/results/qa-*.json"))[-1]
rows = json.load(open(path))["rows"]
queries = {}
for f in glob.glob("eval/queries_*.jsonl"):
    for line in open(f):
        if line.strip() and not line.startswith("#"):
            q = json.loads(line)
            queries[q["id"]] = q
idx = load_index(Settings())


def parse_plan(trace_line):
    subs = []
    for query, flt in re.findall(r"'(.+?)' \[([^\]]*)\]", trace_line):
        toks = flt.split()
        ticker = toks[0] if toks and toks[0] != "*" else None
        form = next((t for t in toks if t.startswith("10-")), None)
        fy = next((int(t) for t in toks if t.isdigit()), None)
        subs.append((query, ticker, form, fy))
    return subs


def hit(query, q, tickers, form, fy):
    hits = idx.search(query, k=5, mode="dense", filters={"tickers": tickers, "form": form, "fiscal_year": fy})
    return any(is_relevant(h.chunk, q) for h in hits)


n = planner_hits = question_hits = 0
for r in rows:
    if r.get("gold_in_evidence") is not False or r["id"] not in queries:
        continue
    q = queries[r["id"]]
    n += 1
    plan = parse_plan(r["trace"][0]) if r.get("trace") else []
    p_hit = any(hit(qq, q, [t] if t else [q["ticker"]], form, fy) for qq, t, form, fy in plan)
    q_hit = hit(q["question"], q, [q["ticker"]], None, q.get("fiscal_year"))
    planner_hits += p_hit
    question_hits += q_hit
    plan_txt = "; ".join(f"'{qq}' [{t or '*'} {form or '-'} {fy or '-'}]" for qq, t, form, fy in plan)
    print(f"{r['id']}\n   planner: {plan_txt}  -> gold in top5: {p_hit}\n   original question  -> gold in top5: {q_hit}"
          f"   (expected FY {q.get('fiscal_year')})\n")

print(f"{path}\nmisses: {n} | gold found by planner queries: {planner_hits} | by original question: {question_hits}")
