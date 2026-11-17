"""Analyst agent: writes the answer as atomic, cited claims."""
from __future__ import annotations

import re

from .prompts import ANALYST_SYSTEM, ANALYST_USER


def header_of(c: dict) -> str:
    return (
        f"{c['company']} ({c['ticker']}) {c['form']}, period ending {c['report_date']} "
        f"(fiscal {c['report_date'][:4]}) - {c['section']}"
    )


def render_evidence(chunks: list[dict], labels: list[str]) -> str:
    return "\n\n".join(f"[{lab}] {header_of(c)}\n{c['text']}" for lab, c in zip(labels, chunks))


def _norm_cite(x) -> str | None:
    if isinstance(x, int):
        return f"E{x}"
    m = re.search(r"E?\s*(\d+)", str(x), re.IGNORECASE)
    return f"E{m.group(1)}" if m else None


def make_analyst(ctx):
    def analyst(state):
        chunks = list((state.get("evidence") or {}).values())
        labels = [f"E{i + 1}" for i in range(len(chunks))]
        label_to_id = {lab: c["chunk_id"] for lab, c in zip(labels, chunks)}
        feedback = state.get("feedback") or ""
        user = ANALYST_USER.format(
            question=state["question"],
            evidence=render_evidence(chunks, labels) or "(no evidence retrieved)",
            feedback=f"\nFEEDBACK ON YOUR PREVIOUS ATTEMPT:\n{feedback}\n" if feedback else "",
        )
        note = ""
        try:
            data = ctx.llm.chat_json(ANALYST_SYSTEM, user, max_tokens=1200)
        except Exception as e:  # noqa: BLE001
            data, note = {}, f" (analyst failed: {e})"
        raw_claims = data.get("claims") if isinstance(data, dict) else None
        claims = []
        for c in raw_claims or []:
            if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                continue
            cites = []
            for x in c.get("citations") or []:
                lab = _norm_cite(x)
                if lab in label_to_id and label_to_id[lab] not in cites:
                    cites.append(label_to_id[lab])
            calc = c.get("calculation")
            claims.append(
                {
                    "id": f"C{len(claims) + 1}",
                    "text": str(c["text"]).strip(),
                    "citations": cites,
                    "calculation": str(calc).strip() if calc else None,
                }
            )
        return {"claims": claims, "trace": [f"analyst: {len(claims)} claims{note}"]}

    return analyst
