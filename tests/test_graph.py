"""End-to-end LangGraph runs with a scripted LLM: shows self-correction working."""
from conftest import FakeLLM, evidence_label_containing

from secrag.agents.graph import AgentContext, Pipeline
from secrag.edgar.xbrl import FactStore
from secrag.evaluation import answer_matches

PLAN = lambda _u: {"sub_queries": [  # noqa: E731
    {"query": "Apple net sales consolidated statements of operations", "ticker": "AAPL", "form": "10-K",
     "fiscal_year": 2024},
    {"query": "made-up", "ticker": "NOPE", "form": "8-K", "fiscal_year": "x"},  # sanitised by validator
]}
ALWAYS_SUPPORTED = lambda u: {"results": [  # noqa: E731  (a lazy verifier LLM that approves everything)
    {"id": f"C{i}", "verdict": "SUPPORTED", "reason": "ok", "followup_query": None} for i in range(1, 6)
]}


def make_ctx(index, settings, llm):
    return AgentContext(llm=llm, index=index, facts=FactStore(settings.xbrl_dir), settings=settings)


def test_numeric_check_triggers_self_correction(index, settings):
    attempts = []

    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        assert lab, "table chunk should have been retrieved"
        attempts.append(user)
        if len(attempts) == 1:  # first draft hallucinates a number
            return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $400.5 billion.", "citations": [lab]}]}
        assert "numbers not found" in user  # verifier feedback reached the analyst
        return {"claims": [
            {"text": "Apple's net sales for fiscal 2024 were $391,035 million.", "citations": [lab]},
            {"text": "Apple's net sales grew 2.0% in fiscal 2024.", "citations": [lab],
             "calculation": "(391035 - 383285) / 383285 * 100"},
        ]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    out = Pipeline(make_ctx(index, settings, llm)).ask("What were Apple's net sales in fiscal 2024?")
    assert out["round"] == 1
    assert [v["verdict"] for v in out["verdicts"]] == ["SUPPORTED", "SUPPORTED"]
    assert "$391,035 million" in out["answer"] and "400.5" not in out["answer"]
    assert out["citations"][0]["ticker"] == "AAPL"
    assert answer_matches(out["answer"], 391035e6)
    assert llm.calls == ["Planner", "Analyst", "Verifier", "Analyst", "Verifier"]


def test_unfixable_claim_is_dropped_after_budget(index, settings):
    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $500 billion.", "citations": [lab]}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    out = Pipeline(make_ctx(index, settings, llm)).ask("What were Apple's net sales in fiscal 2024?")
    assert out["round"] == settings.max_verify_rounds
    assert out["kept_claims"] == [] and len(out["dropped"]) == 1
    assert "couldn't find verifiable support" in out["answer"]


def test_no_verifier_baseline_keeps_hallucination(index, settings):
    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $400.5 billion.", "citations": [lab]}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    out = Pipeline(make_ctx(index, settings, llm), verify=False).ask("Apple net sales 2024?")
    assert "400.5" in out["answer"] and llm.calls == ["Planner", "Analyst"]
    assert not answer_matches(out["answer"], 391035e6)


def test_llm_contradiction_is_respected(index, settings):
    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2023 were $391,035 million.", "citations": [lab]}]}

    def verifier(user):  # numbers match, but the period is wrong -> LLM catches it
        return {"results": [{"id": "C1", "verdict": "CONTRADICTED", "reason": "391,035 is fiscal 2024",
                             "followup_query": "Apple net sales fiscal 2023"}]}

    settings.max_verify_rounds = 1
    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=verifier)
    out = Pipeline(make_ctx(index, settings, llm)).ask("Apple net sales fiscal 2023?")
    assert out["dropped"][0]["verdict"] == "CONTRADICTED"
    assert any("Apple net sales fiscal 2023" in t for t in out["trace"]) or out["round"] == 1


def test_fallback_plan_extracts_company_and_year(index):
    from secrag.agents.planner import fallback_plan

    plan = fallback_plan("What was Apple's net income for fiscal 2024?", index.catalog())
    assert plan == [{"query": "What was Apple's net income for fiscal 2024?", "ticker": "AAPL", "form": "10-K",
                     "fiscal_year": 2024}]
    both = fallback_plan("Compare MSFT and Apple revenue", index.catalog())
    assert {p["ticker"] for p in both} == {"AAPL", "MSFT"}


def test_planner_failure_uses_rule_based_plan(index, settings):
    def broken_planner(_u):
        raise RuntimeError("Error code: 500 - token repeat limit reached")

    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $391,035 million.", "citations": [lab]}]}

    llm = FakeLLM(planner=broken_planner, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    out = Pipeline(make_ctx(index, settings, llm)).ask("What were Apple's net sales in fiscal 2024?")
    assert out["sub_queries"][0]["ticker"] == "AAPL" and "rule-based" in out["trace"][0]
    assert out["round"] == 0


def test_off_topic_answer_triggers_retry(index, settings):
    """A TRUE claim about the wrong line item (net sales) must not pass for a net-income question."""
    attempts = []

    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        attempts.append(user)
        if len(attempts) == 1:
            return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $391,035 million.", "citations": [lab]}]}
        assert "do NOT answer the question" in user and "net income" in user
        return {"claims": [{"text": "Apple's net income for fiscal 2024 was $93,736 million.", "citations": [lab]}]}

    def verifier(user):
        on_topic = "net income" in user.split("CLAIMS:")[1]
        return {"answers_question": on_topic, "gap": None if on_topic else "net income for fiscal 2024",
                "results": [{"id": "C1", "verdict": "SUPPORTED", "reason": "stated in P1", "followup_query": None}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=verifier)
    out = Pipeline(make_ctx(index, settings, llm)).ask("What was Apple's net income for fiscal 2024?")
    assert out["round"] == 1
    assert "$93,736 million" in out["answer"] and "391,035" not in out["answer"]
    assert "does not answer" in " ".join(out["trace"])
    assert "Note:" not in out["answer"]


def test_eval_qa_configs_toggle_xbrl(index, settings):
    from secrag.evaluation import eval_qa

    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $391,035 million.", "citations": [lab]}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    q = [{"id": "x", "question": "What were Apple's net sales for fiscal 2024?", "ticker": "AAPL", "gold_value": 391035e6}]
    summary, rows = eval_qa(make_ctx(index, settings, llm), q, log=lambda *_: None)
    assert set(summary) == {"no_verifier", "verifier_no_xbrl", "verifier_xbrl"}
    assert all(r["numeric_correct"] for r in rows)


def test_original_question_added_to_plan():
    from secrag.agents.planner import with_original_question

    q = "What was Apple's net income for fiscal 2024?"
    subs = [{"query": "net income", "ticker": "AAPL", "form": "10-K", "fiscal_year": 2024}]
    out = with_original_question(q, subs)
    assert out[1] == {"query": q, "ticker": "AAPL", "form": "10-K", "fiscal_year": 2024, "original": True}
    assert with_original_question(q, out) == out  # idempotent, no duplicates


def test_rag_fusion_puts_original_question_first(index, settings):
    from secrag.agents.retriever import make_retriever

    ctx = make_ctx(index, settings, llm=None)
    # Disjoint result lists (different companies) -> every rank-r pair ties under RRF;
    # the user's original question must win the tie even though it is listed second.
    pending = [
        {"query": "cloud revenue", "ticker": "MSFT", "form": None, "fiscal_year": None},
        {"query": "Apple net sales consolidated statements of operations", "ticker": "AAPL", "form": None,
         "fiscal_year": None, "original": True},
    ]
    out = make_retriever(ctx)({"question": "What were Apple's net sales?", "pending_queries": pending})
    first = next(iter(out["evidence"].values()))
    assert first["ticker"] == "AAPL" and "Net sales |" in first["text"]


def test_rag_fusion_promotes_consensus():
    from secrag.index.store import rrf

    # chunk "b" is 2nd for both queries; "a" and "c" are 1st for only one query each
    assert rrf([["a", "b"], ["c", "b"]])[0][0] == "b"


def test_abstention_is_never_scored_correct(index, settings):
    from secrag.evaluation import eval_qa

    def analyst(user):  # unfixable claim -> dropped -> abstention note contains "1 claim(s) removed"
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's diluted EPS was $9.99.", "citations": [lab]}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    q = [{"id": "eps", "question": "What was Apple's diluted EPS for fiscal 2024?", "ticker": "AAPL", "gold_value": 1.0}]
    summary, rows = eval_qa(make_ctx(index, settings, llm), q, configs=(("verifier_xbrl", True, True),),
                            log=lambda *_: None)
    assert rows[0]["claims"] == 0 and "1 claim(s) removed" in rows[0]["answer"]
    assert rows[0]["numeric_correct"] is False


def test_eval_qa_concurrent_workers(index, settings):
    from secrag.evaluation import eval_qa

    def analyst(user):
        lab = evidence_label_containing(user, "Net sales |")
        return {"claims": [{"text": "Apple's net sales for fiscal 2024 were $391,035 million.", "citations": [lab]}]}

    llm = FakeLLM(planner=PLAN, analyst=analyst, verifier=ALWAYS_SUPPORTED)
    qs = [{"id": f"q{i}", "question": "What were Apple's net sales for fiscal 2024?", "ticker": "AAPL",
           "gold_value": 391035e6} for i in range(8)]
    summary, rows = eval_qa(make_ctx(index, settings, llm), qs, configs=(("verifier_xbrl", True, True),),
                            log=lambda *_: None, workers=4)
    s = summary["verifier_xbrl"]
    assert s["n"] == 8 and s["numeric_accuracy"] == 1.0 and s["workers"] == 4 and s["questions_per_min"] > 0
    assert sorted(r["id"] for r in rows) == sorted(q["id"] for q in qs)
