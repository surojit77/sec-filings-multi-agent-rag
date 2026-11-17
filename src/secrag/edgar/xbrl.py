"""Structured XBRL facts from SEC's companyfacts API.

Used twice:
  1. by the Verifier, as an authoritative cross-check for numeric claims;
  2. by the eval harness, to auto-generate labelled numeric questions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

# us-gaap concept -> human label. Several concepts can share a label (companies
# tag revenue differently); order = preference when more than one is present.
KEY_CONCEPTS: dict[str, str] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "Revenues": "Revenue",
    "SalesRevenueNet": "Revenue",
    "CostOfGoodsAndServicesSold": "Cost of revenue",
    "CostOfRevenue": "Cost of revenue",
    "GrossProfit": "Gross profit",
    "ResearchAndDevelopmentExpense": "R&D expense",
    "SellingGeneralAndAdministrativeExpense": "SG&A expense",
    "OperatingExpenses": "Operating expenses",
    "OperatingIncomeLoss": "Operating income",
    "NetIncomeLoss": "Net income",
    "EarningsPerShareBasic": "EPS (basic)",
    "EarningsPerShareDiluted": "EPS (diluted)",
    "Assets": "Total assets",
    "Liabilities": "Total liabilities",
    "StockholdersEquity": "Stockholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
    "NetCashProvidedByUsedInOperatingActivities": "Operating cash flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "Capital expenditures",
    "LongTermDebtNoncurrent": "Long-term debt",
}
CONCEPT_ORDER = list(KEY_CONCEPTS)


@dataclass
class Fact:
    ticker: str
    concept: str
    label: str
    unit: str
    value: float
    start: str | None
    end: str
    period: str  # "FY", "Q", "YTD6M", "instant", ...
    form: str
    accession: str
    filed: str

    @property
    def fiscal_year(self) -> int:
        return int(self.end[:4])


def _period(start: str | None, end: str) -> str:
    if not start:
        return "instant"
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    if 350 <= days <= 380:
        return "FY"
    if 80 <= days <= 100:
        return "Q"
    return f"YTD{round(days / 30.4)}M"


def extract_facts(ticker: str, companyfacts: dict) -> list[Fact]:
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    out: list[Fact] = []
    seen: set[tuple] = set()
    for concept, label in KEY_CONCEPTS.items():
        node = gaap.get(concept)
        if not node:
            continue
        for unit, items in node.get("units", {}).items():
            for it in items:
                form = it.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue
                key = (concept, unit, it.get("start"), it["end"], it.get("accn"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Fact(
                        ticker=ticker.upper(),
                        concept=concept,
                        label=label,
                        unit=unit,
                        value=float(it["val"]),
                        start=it.get("start"),
                        end=it["end"],
                        period=_period(it.get("start"), it["end"]),
                        form=form,
                        accession=it.get("accn", ""),
                        filed=it.get("filed", ""),
                    )
                )
    return out


def fmt_value(f: Fact) -> str:
    if f.unit == "USD" and abs(f.value) >= 1e6:
        return f"{f.value:,.0f} USD (= {f.value / 1e6:,.0f} million)"
    if "/" in f.unit:
        return f"{f.value:.2f} {f.unit}"
    return f"{f.value:,.0f} {f.unit}"


def render_facts(facts: list[Fact], limit: int = 40) -> str:
    lines = [
        f"{f.ticker} | {f.label} | {f.period} ending {f.end} | {fmt_value(f)}" for f in facts[:limit]
    ]
    return "\n".join(lines)


class FactStore:
    """One JSONL file of facts per ticker under data/xbrl/."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._cache: dict[str, list[Fact]] = {}

    def save(self, ticker: str, facts: list[Fact]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{ticker.upper()}.jsonl", "w", encoding="utf-8") as fh:
            for f in facts:
                fh.write(json.dumps(asdict(f)) + "\n")
        self._cache[ticker.upper()] = facts

    def load(self, ticker: str) -> list[Fact]:
        t = ticker.upper()
        if t not in self._cache:
            path = self.dir / f"{t}.jsonl"
            if not path.exists():
                self._cache[t] = []
            else:
                with open(path, encoding="utf-8") as fh:
                    self._cache[t] = [Fact(**json.loads(line)) for line in fh if line.strip()]
        return self._cache[t]

    def query(
        self, tickers: list[str], years: set[int] | None = None, periods: set[str] | None = None
    ) -> list[Fact]:
        """Facts for tickers/years, de-duplicated (latest filing wins) and sorted."""
        best: dict[tuple, Fact] = {}
        for t in tickers:
            for f in self.load(t):
                if years and f.fiscal_year not in years:
                    continue
                if periods and f.period not in periods:
                    continue
                key = (f.ticker, f.label, f.unit, f.start, f.end)
                cur = best.get(key)
                better = cur is None or (
                    CONCEPT_ORDER.index(f.concept),
                    -int(f.filed.replace("-", "") or 0),
                ) < (CONCEPT_ORDER.index(cur.concept), -int(cur.filed.replace("-", "") or 0))
                if better:
                    best[key] = f
        return sorted(best.values(), key=lambda f: (f.ticker, -f.fiscal_year, f.period, f.label))
