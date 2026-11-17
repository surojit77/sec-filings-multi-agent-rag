from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    tickers: list[str]  # optional user restriction
    sub_queries: list[dict]  # planner output
    pending_queries: list[dict]  # what the retriever should run next
    evidence: dict[str, dict]  # chunk_id -> chunk dict (insertion-ordered)
    claims: list[dict]  # analyst output: {id, text, citations[chunk_id], calculation}
    verdicts: list[dict]  # verifier output, one per claim
    feedback: str  # verifier -> analyst on retry
    answer_gap: str | None  # verifier: what the answer still fails to address
    round: int  # verification rounds used
    next: str  # "retry" | "done"
    answer: str
    citations: list[dict]
    kept_claims: list[dict]
    dropped: list[dict]
    trace: Annotated[list[str], operator.add]
