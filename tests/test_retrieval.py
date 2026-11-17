from secrag.edgar.xbrl import Fact, FactStore
from secrag.evaluation import eval_retrieval, make_xbrl_queries
from secrag.index.embedder import HashEmbedder
from secrag.index.store import HybridIndex, rrf


def test_rrf_prefers_consensus():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [i for i, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd
    assert set(order) == {1, 2, 3, 4}


def test_search_modes_and_filters(index):
    for mode in ("bm25", "dense", "hybrid"):
        hits = index.search("Apple net sales consolidated statements of operations", k=3, mode=mode)
        assert hits and hits[0].chunk.ticker == "AAPL"
    msft = index.search("revenue", k=5, filters={"tickers": ["MSFT"]})
    assert msft and all(h.chunk.ticker == "MSFT" for h in msft)
    assert index.search("revenue", filters={"tickers": ["AAPL"], "fiscal_year": 1999}) == []


def test_persistence_roundtrip(index):
    index.save()
    again = HybridIndex.load(index.index_dir, HashEmbedder())
    assert len(again.chunks) == len(index.chunks)
    assert again.has_accession("0000320193-24-000123")
    assert {f["ticker"] for f in again.catalog()} == {"AAPL", "MSFT"}


def test_xbrl_queries_and_retrieval_eval(index, tmp_path):
    store = FactStore(tmp_path / "xbrl")
    store.save("AAPL", [
        Fact("AAPL", "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue", "USD", 391035e6,
             "2023-10-01", "2024-09-28", "FY", "10-K", "0000320193-24-000123", "2024-11-01"),
        Fact("AAPL", "EarningsPerShareDiluted", "EPS (diluted)", "USD/shares", 6.08,
             "2023-10-01", "2024-09-28", "FY", "10-K", "0000320193-24-000123", "2024-11-01"),
    ])
    qs = make_xbrl_queries(index, store)
    assert {q["concept"] for q in qs} == {"RevenueFromContractWithCustomerExcludingAssessedTax", "EarningsPerShareDiluted"}
    rev = next(q for q in qs if q["concept"].startswith("Revenue"))
    assert "391,035" in rev["evidence_strings"] and "fiscal year 2024" in rev["question"]
    res = eval_retrieval(index, qs, k=5)
    assert res["n_answerable"] == 2
    assert set(res["results"]) == {"bm25", "dense", "hybrid"}
    assert 0.0 <= res["results"]["hybrid"]["all"]["recall@k"] <= 1.0


def test_weighted_rrf():
    # doc 9 is 1st for ranker A only, doc 7 is 1st for ranker B only
    assert rrf([[9, 7], [7, 9]], weights=[1.0, 1.0])[0][1] == rrf([[9, 7], [7, 9]])[0][1]
    assert rrf([[9, 7], [7, 9]], weights=[0.3, 1.0])[0][0] == 7
    assert rrf([[9, 7], [7, 9]], weights=[1.0, 0.3])[0][0] == 9


def test_make_text_queries_filters_bad_labels(index):
    from secrag.evaluation import make_text_queries

    class QGen:
        def __init__(self):
            self.n = 0

        def chat_json(self, system, user, max_tokens=None):
            self.n += 1
            passage = user.split("):\n", 1)[1]
            if self.n == 1:  # hallucinated quote -> rejected
                return {"question": "What did Apple say?", "evidence": "a sentence that is not in the passage"}
            words = passage.split()
            return {"question": f"Apple fiscal 2024 question {self.n}?", "evidence": " ".join(words[-8:])}

    qs = make_text_queries(index, QGen(), per_ticker=2, min_words=20, log=lambda *_: None)
    assert qs, "should keep at least one valid question"
    for q in qs:
        src = next(c for c in index.chunks if c.chunk_id == q["source_chunk"])
        assert q["evidence_strings"][0] in " ".join(src.text.split())
        assert q["type"] in ("synthetic-paraphrase", "synthetic-keyword")


def test_router():
    from secrag.index.router import classify

    numeric = [
        "What was Apple Inc.'s total revenue (net sales) for fiscal year 2025?",
        "What were NVIDIA CORP's total assets at the end of fiscal year 2026?",
        "How much did Microsoft spend on research and development expense in fiscal 2026?",
        "What was Tesla's operating cash flow for fiscal year 2025?",
    ]
    narrative = [
        "How does Tesla test and prepare its response to cybersecurity incidents?",
        "What risks does NVIDIA face from export controls?",
        "What drove the increase in Apple's Services net sales?",  # metric named, but asks 'what drove'
        "Who oversees information security matters at NVIDIA?",
    ]
    assert all(classify(q) == "numeric" for q in numeric)
    assert all(classify(q) == "narrative" for q in narrative)


def test_one_row_layout_table_heading_is_detected():
    from secrag.edgar.parser import parse_filing_html

    html = ("<html><body><table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>"
            "<p>Our business is subject to many risks described below in considerable detail.</p></body></html>")
    blocks = parse_filing_html(html, "10-K")
    assert blocks[-1].section == "Item 1A - Risk Factors"


def test_routed_search(index):
    hits = index.search("What was Apple's net income for fiscal year 2024?", k=3, mode="routed")
    assert hits
