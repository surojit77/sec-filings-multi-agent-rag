"""Shared fixtures. Everything here runs offline: no EDGAR, no GPU, no model downloads."""
from __future__ import annotations

import re

import pytest

from secrag.config import Settings
from secrag.edgar.client import FilingMeta
from secrag.edgar.parser import parse_filing_html
from secrag.index.chunker import chunk_blocks
from secrag.index.embedder import HashEmbedder
from secrag.index.store import HybridIndex

# Synthetic, EDGAR-shaped 10-K (structure mimics real inline-XBRL filings).
FILLER = " ".join(
    "The Company depends on component suppliers and outsourcing partners located primarily in Asia "
    "and any disruption could materially adversely affect results.".split()
    * 3
)
SAMPLE_10K = f"""
<html><head><title>x</title><style>.a{{}}</style></head><body>
<div style="display: none"><ix:header><ix:hidden>hidden 999,999</ix:hidden></ix:header></div>
<p>TABLE OF CONTENTS</p>
<table>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
  <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>20</td></tr>
</table>
<p><b>PART I</b></p>
<p>Item 1A. Risk Factors</p>
<p>{FILLER}</p>
<p>12</p>
<p><b>PART II</b></p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
<p>Total net sales increased 2% or $7.8 billion during 2024 compared to 2023, driven by higher
<span>Services</span> net sales. {FILLER}</p>
<p>CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<p>(In millions, except number of shares)</p>
<table>
  <tr><td></td><td>2024</td><td></td><td>2023</td></tr>
  <tr><td>Net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
  <tr><td>Net income</td><td>$</td><td>93,736</td><td>$</td><td>96,995</td></tr>
  <tr><td>Other income/(expense), net</td><td>(</td><td>269</td><td>)</td><td>(</td><td>565</td><td>)</td></tr>
  <tr><td>Diluted earnings per share</td><td>$</td><td>6.08</td><td>$</td><td>6.13</td></tr>
  <tr><td>Gross margin percentage</td><td>46.2</td><td>%</td><td>44.1</td><td>%</td></tr>
</table>
<p>Item 7 of this report should be read together with the financial statements and the notes. {FILLER}</p>
</body></html>
"""

META = FilingMeta(
    ticker="AAPL",
    cik=320193,
    company="Apple Inc.",
    form="10-K",
    accession="0000320193-24-000123",
    filing_date="2024-11-01",
    report_date="2024-09-28",
    primary_doc="aapl-20240928.htm",
)
META_MSFT = FilingMeta(
    ticker="MSFT",
    cik=789019,
    company="Microsoft Corp",
    form="10-K",
    accession="0000950170-24-087843",
    filing_date="2024-07-30",
    report_date="2024-06-30",
    primary_doc="msft-10k_20240630.htm",
)
SAMPLE_MSFT = """
<html><body><p>Item 7. Management's Discussion and Analysis</p>
<p>Microsoft Cloud revenue increased 23% driven by Azure and other cloud services. Server products and
cloud services revenue grew as customers continued to migrate workloads to the cloud during the fiscal year,
and we continued to invest in AI infrastructure and datacenter capacity.</p>
<p>INCOME STATEMENTS</p><p>(In millions)</p>
<table><tr><td>Year Ended June 30,</td><td>2024</td><td>2023</td></tr>
<tr><td>Total revenue</td><td>$</td><td>245,122</td><td>$</td><td>211,915</td></tr></table>
</body></html>
"""


@pytest.fixture
def blocks():
    return parse_filing_html(SAMPLE_10K, "10-K")


@pytest.fixture
def index(tmp_path):
    idx = HybridIndex(tmp_path / "index", HashEmbedder())
    idx.add(chunk_blocks(parse_filing_html(SAMPLE_10K, "10-K"), META, size=60, overlap=10))
    idx.add(chunk_blocks(parse_filing_html(SAMPLE_MSFT, "10-K"), META_MSFT, size=60, overlap=10))
    return idx


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.max_verify_rounds = 2
    return s


def evidence_label_containing(user_prompt: str, needle: str) -> str | None:
    """Find the [E#] label of the evidence block that contains `needle`."""
    for m in re.finditer(r"\[(E\d+)\][^\n]*\n(.*?)(?=\n\n\[E\d+\]|\Z)", user_prompt, re.S):
        if needle in m.group(2):
            return m.group(1)
    return None


class FakeLLM:
    """Scripted LLM: routes by agent (system prompt) to handler functions."""

    def __init__(self, planner=None, analyst=None, verifier=None):
        self.planner, self.analyst, self.verifier = planner, analyst, verifier
        self.calls: list[str] = []

    def chat_json(self, system, user, max_tokens=None):
        for name in ("Planner", "Analyst", "Verifier"):
            if f"the {name} agent" in system:
                self.calls.append(name)
                return getattr(self, name.lower())(user)
        raise AssertionError("unknown agent prompt")
