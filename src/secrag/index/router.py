"""Query-type routing.

Eval finding: dense retrieval wins on numeric table lookups ("What was X's net income for
FY2025?"), where BM25 drags in prose that merely repeats the metric name, while BM25-backed
hybrid wins on narrative questions (risk factors, legal proceedings, exact terms).
So: numeric metric questions -> dense, everything else -> hybrid.

The vocabulary below is generic financial-statement language, written up front rather than
fitted to the eval questions.
"""
from __future__ import annotations

import re

_METRIC = re.compile(
    r"\b(revenues?|net sales|net income|net (?:loss|earnings)|operating income|operating loss|"
    r"gross (?:profit|margin)|earnings per share|eps|total assets|total liabilities|"
    r"cash and cash equivalents|(?:operating |free )?cash flows?|capital expenditures|"
    r"research and development (?:expense|costs?)|r&d (?:expense|spending)|operating expenses|"
    r"(?:stockholders'?|shareholders'?) equity|long-term debt|dividends? per share)\b",
    re.IGNORECASE,
)
_ASKS_AMOUNT = re.compile(r"\b(how much|what (?:was|were|is|are)|amount|total|report(?:ed)?)\b", re.IGNORECASE)


def classify(query: str) -> str:
    """'numeric' for questions asking for a financial-statement figure, else 'narrative'."""
    return "numeric" if _METRIC.search(query) and _ASKS_AMOUNT.search(query) else "narrative"


def route(query: str) -> str:
    return "dense" if classify(query) == "numeric" else "hybrid"
