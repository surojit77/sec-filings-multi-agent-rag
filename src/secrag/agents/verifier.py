"""Verifier agent: hybrid claim verification.

For every claim:
  1. structural check  - does it cite anything?
  2. numeric check     - every number must be grounded in the cited passages or
                         XBRL facts (unit-scale and rounding aware); derived numbers
                         need a `calculation` that re-evaluates correctly.
  3. LLM entailment    - one batched call judges each claim against its passages
                         (and the XBRL facts table, when enabled).
A claim is SUPPORTED only if (2) and (3) both pass.

The same LLM call also checks ANSWER RELEVANCE: do the claims, taken together,
answer the question that was asked (right metric, company, period)? A true
statement about operating *expenses* does not answer a question about operating
*income*. Unsupported claims or an irrelevant answer produce follow-up queries +
feedback, and the graph loops back to Retriever -> Analyst (self-correction)
until everything passes or the round budget is spent.
"""
from __future__ import annotations

from ..edgar.xbrl import render_facts
from ..numbers import Num, check_claim_numbers, extract_numbers
from .analyst import header_of
from .prompts import VERIFIER_SYSTEM, VERIFIER_USER

VERDICTS = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}
MAX_FOLLOWUPS = 3


def relevant_facts(ctx, claims: list[dict], evidence: dict) -> list:
    if ctx.facts is None:
        return []
    cited = [evidence[cid] for c in claims for cid in c["citations"] if cid in evidence]
    if not cited:
        return []
    tickers = sorted({c["ticker"] for c in cited})
    years = {int(c["report_date"][:4]) for c in cited}
    years |= {y - 1 for y in years}
    return ctx.facts.query(tickers, years=years, periods={"FY", "Q", "instant"})


def llm_check(llm, claims: list[dict], evidence: dict, facts_text: str,
              question: str | None = None) -> tuple[dict[str, dict], dict]:
    """One batched call. Returns (claim_id -> {verdict, reason, followup_query}, {answers_question, gap})."""
    meta = {"answers_question": True, "gap": None}
    to_check = [c for c in claims if c["citations"]]
    if not to_check:
        return {}, meta
    cited_ids: list[str] = []
    for c in to_check:
        for cid in c["citations"]:
            if cid in evidence and cid not in cited_ids:
                cited_ids.append(cid)
    label = {cid: f"P{i + 1}" for i, cid in enumerate(cited_ids)}
    passages = "\n\n".join(f"[{label[cid]}] {header_of(evidence[cid])}\n{evidence[cid]['text']}" for cid in cited_ids)
    claim_lines = []
    for c in to_check:
        refs = ", ".join(label[cid] for cid in c["citations"] if cid in label)
        line = f"{c['id']}: {c['text']}\n    cites: {refs}"
        if c.get("calculation"):
            line += f"\n    calculation: {c['calculation']}"
        claim_lines.append(line)
    try:
        data = llm.chat_json(
            VERIFIER_SYSTEM,
            VERIFIER_USER.format(
                question=question or "(not provided: skip the answers_question check and return true)",
                passages=passages,
                facts=facts_text or "(none)",
                claims="\n".join(claim_lines),
            ),
            max_tokens=900,
        )
    except Exception:  # noqa: BLE001
        return {}, meta
    data = data if isinstance(data, dict) else {}
    out: dict[str, dict] = {}
    for r in data.get("results") or []:
        if not isinstance(r, dict) or "id" not in r:
            continue
        verdict = str(r.get("verdict", "")).upper().strip()
        out[str(r["id"])] = {
            "verdict": verdict if verdict in VERDICTS else "UNSUPPORTED",
            "reason": str(r.get("reason") or ""),
            "followup_query": r.get("followup_query") or None,
        }
    if question and data.get("answers_question") is False:
        meta = {"answers_question": False, "gap": str(data.get("gap") or "").strip() or None}
    return out, meta


def llm_verdicts(llm, claims: list[dict], evidence: dict, facts_text: str) -> dict[str, dict]:
    """Per-claim entailment only (used by the eval judge)."""
    return llm_check(llm, claims, evidence, facts_text)[0]


def verify_claims(ctx, claims: list[dict], evidence: dict, question: str | None = None) -> tuple[list[dict], dict]:
    facts = relevant_facts(ctx, claims, evidence)
    fact_nums = [Num(f.value, 1.0, False, 0, "xbrl") for f in facts]
    lv_all, meta = llm_check(ctx.llm, claims, evidence, render_facts(facts), question)
    results = []
    for c in claims:
        r = dict(c)
        if not c["citations"]:
            r.update(verdict="NO_EVIDENCE", reason="claim cites no evidence", numeric_check="skipped",
                     followup_query=c["text"])
            results.append(r)
            continue
        cited_text = "\n".join(evidence[cid]["text"] for cid in c["citations"] if cid in evidence)
        ev_nums = extract_numbers(cited_text, skip_years=False) + fact_nums
        num_ok, num_reason = check_claim_numbers(c["text"], c.get("calculation"), ev_nums)
        lv = lv_all.get(c["id"], {"verdict": "UNSUPPORTED", "reason": "verifier returned no verdict",
                                  "followup_query": None})
        verdict, reason = lv["verdict"], lv["reason"]
        if verdict == "SUPPORTED" and not num_ok:
            verdict, reason = "UNSUPPORTED", num_reason
        r.update(verdict=verdict, reason=reason, numeric_check=num_reason, followup_query=lv.get("followup_query"))
        results.append(r)
    return results, meta


def _feedback(results: list[dict], meta: dict) -> str:
    lines = []
    if not meta.get("answers_question", True):
        gap = meta.get("gap") or "the specific item the question asks for"
        lines.append(f"Your claims do NOT answer the question. Missing: {gap}. "
                     "Answer exactly what was asked (same metric, company and period); do not substitute a related item.")
    lines.append("Each claim was checked against its cited passages. Keep SUPPORTED claims that answer the question; "
                 "fix or remove the others.")
    for r in results:
        lines.append(f"- {r['verdict']}: \"{r['text']}\"" + ("" if r["verdict"] == "SUPPORTED" else f" -- {r['reason']}"))
    return "\n".join(lines)


def make_verifier(ctx):
    def verifier(state):
        claims = state.get("claims") or []
        evidence = state.get("evidence") or {}
        rnd = state.get("round", 0)
        results, meta = verify_claims(ctx, claims, evidence, state.get("question"))
        failed = [r for r in results if r["verdict"] != "SUPPORTED"]
        off_topic = not meta["answers_question"]
        summary = ", ".join(f"{r['id']}={r['verdict']}" for r in results) or "no claims"
        if off_topic:
            summary += f" | does not answer the question (missing: {meta['gap'] or '?'})"
        update = {"verdicts": results, "answer_gap": meta["gap"] if off_topic else None,
                  "trace": [f"verifier (round {rnd}): {summary}"]}

        needs_retry = bool(failed) or not claims or off_topic
        if needs_retry and rnd < ctx.settings.max_verify_rounds:
            followups: list[dict] = []
            seen: set[str] = set()
            sources = list(failed)
            if off_topic or not claims:
                # re-search for the missing item with the Planner's original filters (company/form/year)
                plan = (state.get("sub_queries") or [{}])[0]
                gap_q = f"{state['question']} {meta['gap'] or ''}".strip()
                sources.insert(0, {"text": gap_q, "citations": [], "followup_query": None, "_filters": plan})
            for r in sources:
                q = (r.get("followup_query") or r["text"]).strip()
                if not q or q.lower() in seen:
                    continue
                seen.add(q.lower())
                cited = [evidence[cid] for cid in r.get("citations", []) if cid in evidence]
                if "_filters" in r:
                    f = r["_filters"]
                    followups.append({"query": q, "ticker": f.get("ticker"), "form": f.get("form"),
                                      "fiscal_year": f.get("fiscal_year")})
                else:
                    followups.append({"query": q, "ticker": cited[0]["ticker"] if cited else None,
                                      "form": None, "fiscal_year": None})
            update.update(
                next="retry",
                round=rnd + 1,
                pending_queries=followups[:MAX_FOLLOWUPS],
                feedback=_feedback(results, meta) if claims else "No claims were produced. Answer from the evidence.",
            )
        else:
            update["next"] = "done"
        return update

    return verifier
