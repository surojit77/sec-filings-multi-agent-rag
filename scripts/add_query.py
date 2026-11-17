import json, sys
qid, ticker, year, question, *phrases = sys.argv[1:]
ticker = ticker.upper()
hits = {p: 0 for p in phrases}
for line in open("data/index/chunks.jsonl"):
    c = json.loads(line)
    if c["ticker"] == ticker and c["report_date"][:4] == year:
        for p in phrases:
            hits[p] += p.lower() in c["text"].lower()
missing = [p for p, n in hits.items() if n == 0]
if missing:
    sys.exit(f"NOT FOUND in {ticker} {year}: {missing}")
print("chunks containing each phrase:", hits, "(more than ~3 means too generic)")
q = {"id": qid, "type": "manual", "question": question, "ticker": ticker,
     "fiscal_year": int(year), "evidence_strings": phrases}
open("eval/queries_manual.jsonl", "a").write(json.dumps(q) + "\n")
print("added", qid)
