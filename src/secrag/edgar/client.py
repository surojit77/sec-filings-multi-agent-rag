"""Minimal, polite SEC EDGAR client: ticker lookup, filing lists, documents, XBRL facts.

SEC fair-access rules: send a User-Agent with contact info and stay under
10 requests/second. Every response is cached on disk, so re-running ingestion
never downloads the same thing twice.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass
class FilingMeta:
    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    filing_date: str
    report_date: str  # period of report, YYYY-MM-DD
    primary_doc: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_doc)

    @property
    def fiscal_year(self) -> int:
        # Convention used across the project: fiscal year = calendar year of the
        # period end. Matches how most companies label their FY (AAPL FY2024 ends
        # Sep 2024, NVDA FY2025 ends Jan 2025, MSFT FY2024 ends Jun 2024).
        return int(self.report_date[:4])

    def to_dict(self) -> dict:
        return asdict(self)


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_interval: float = 0.15, timeout: float = 30.0):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC EDGAR requires a User-Agent with contact info. "
                "Set SEC_USER_AGENT='Your Name you@example.com' in .env"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = 0.0
        self._tickers: dict[str, tuple[int, str]] | None = None

    # ------------------------------------------------------------------ http
    def _get(self, url: str) -> requests.Response:
        resp = None
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp
        assert resp is not None
        resp.raise_for_status()
        return resp

    def _cached(self, url: str, rel_path: str, as_json: bool, max_age: float | None = None):
        path = self.cache_dir / rel_path
        fresh = path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age)
        if fresh:
            text = path.read_text(encoding="utf-8")
        else:
            text = self._get(url).text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return json.loads(text) if as_json else text

    # ------------------------------------------------------------------ api
    def lookup(self, ticker: str) -> tuple[int, str]:
        """Return (CIK, company title) for a ticker."""
        if self._tickers is None:
            data = self._cached(TICKERS_URL, "company_tickers.json", True, max_age=7 * 86400)
            self._tickers = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in data.values()}
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise ValueError(f"Unknown ticker: {ticker}") from None

    def list_filings(
        self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit_per_form: int = 2
    ) -> list[FilingMeta]:
        """Most recent filings of the given forms (amendments like 10-K/A are excluded)."""
        cik, title = self.lookup(ticker)
        sub = self._cached(
            SUBMISSIONS_URL.format(cik=cik), f"submissions/CIK{cik:010d}.json", True, max_age=86400
        )
        recent = sub["filings"]["recent"]
        company = sub.get("name") or title
        counts = {f: 0 for f in forms}
        out: list[FilingMeta] = []
        for i, form in enumerate(recent["form"]):
            if form not in counts or counts[form] >= limit_per_form:
                continue
            if not recent["primaryDocument"][i] or not recent["reportDate"][i]:
                continue
            out.append(
                FilingMeta(
                    ticker=ticker.upper(),
                    cik=cik,
                    company=company,
                    form=form,
                    accession=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_doc=recent["primaryDocument"][i],
                )
            )
            counts[form] += 1
        return out

    def fetch_document(self, filing: FilingMeta) -> str:
        return self._cached(filing.url, f"filings/{filing.ticker}/{filing.accession}.html", False)

    def company_facts(self, cik: int) -> dict:
        return self._cached(
            FACTS_URL.format(cik=cik), f"companyfacts/CIK{cik:010d}.json", True, max_age=86400
        )
