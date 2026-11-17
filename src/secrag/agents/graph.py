"""LangGraph wiring.

    planner -> retriever -> analyst -> verifier --(all supported / budget spent)--> finalize
                  ^                        |
                  +------(retry: follow-up queries + feedback)

With verify=False the verifier is skipped (analyst -> finalize): that is the
ablation baseline used in eval-qa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from .analyst import make_analyst
from .planner import make_planner
from .retriever import make_retriever
from .state import RAGState
from .verifier import make_verifier


@dataclass
class AgentContext:
    llm: Any
    index: Any
    facts: Any
    settings: Settings
    retrieval_mode: str = "hybrid"
    rerank: bool = False

    @property
    def max_evidence(self) -> int:
        return self.settings.max_evidence


def make_finalize(ctx: AgentContext, verify: bool):
    def finalize(state):
        evidence = state.get("evidence") or {}
        if verify:
            verdicts = state.get("verdicts") or []
            kept = [v for v in verdicts if v["verdict"] == "SUPPORTED"]
            dropped = [v for v in verdicts if v["verdict"] != "SUPPORTED"]
        else:
            kept = [dict(c, verdict="UNVERIFIED") for c in state.get("claims") or [] if c["citations"]]
            dropped = [dict(c, verdict="NO_EVIDENCE") for c in state.get("claims") or [] if not c["citations"]]

        order: list[str] = []
        for c in kept:
            for cid in c["citations"]:
                if cid in evidence and cid not in order:
                    order.append(cid)
        num = {cid: i + 1 for i, cid in enumerate(order)}
        lines = []
        for c in kept:
            marks = "".join(f"[{num[cid]}]" for cid in c["citations"] if cid in num)
            lines.append(f"{c['text']} {marks}".strip())
        answer = " ".join(lines) if lines else "I couldn't find verifiable support for an answer in the ingested filings."
        if verify and kept and state.get("answer_gap"):
            answer += f"\n\n(Note: this may not fully answer the question; still missing: {state['answer_gap']}.)"
        if verify and dropped:
            answer += f"\n\n({len(dropped)} claim(s) removed because they could not be verified against the filings.)"
        citations = [
            {
                "n": num[cid],
                "chunk_id": cid,
                "company": evidence[cid]["company"],
                "ticker": evidence[cid]["ticker"],
                "form": evidence[cid]["form"],
                "report_date": evidence[cid]["report_date"],
                "section": evidence[cid]["section"],
                "url": evidence[cid]["url"],
            }
            for cid in order
        ]
        return {
            "answer": answer,
            "citations": citations,
            "kept_claims": kept,
            "dropped": dropped,
            "trace": [f"finalize: kept {len(kept)}, dropped {len(dropped)}"],
        }

    return finalize


def build_graph(ctx: AgentContext, verify: bool = True):
    g = StateGraph(RAGState)
    g.add_node("planner", make_planner(ctx))
    g.add_node("retriever", make_retriever(ctx))
    g.add_node("analyst", make_analyst(ctx))
    g.add_node("finalize", make_finalize(ctx, verify))
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "analyst")
    if verify:
        g.add_node("verifier", make_verifier(ctx))
        g.add_edge("analyst", "verifier")
        g.add_conditional_edges(
            "verifier", lambda s: s.get("next", "done"), {"retry": "retriever", "done": "finalize"}
        )
    else:
        g.add_edge("analyst", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


class Pipeline:
    def __init__(self, ctx: AgentContext, verify: bool = True):
        self.ctx = ctx
        self.verify = verify
        self.graph = build_graph(ctx, verify)

    def ask(self, question: str, tickers: list[str] | None = None) -> dict:
        init: RAGState = {
            "question": question,
            "tickers": [t.upper() for t in tickers] if tickers else [],
            "trace": [],
        }
        return self.graph.invoke(init, config={"recursion_limit": 50})
