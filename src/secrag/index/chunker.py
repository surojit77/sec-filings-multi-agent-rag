"""Section-aware, table-aware chunking.

- Text: paragraphs are packed into ~`size`-word chunks with `overlap` words of
  carry-over, never crossing a section (Item) boundary.
- Tables: kept whole when they fit (up to 2x size); otherwise split by rows,
  repeating the caption/units/header row in every piece so each chunk is
  self-describing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..edgar.client import FilingMeta
from ..edgar.parser import Block

MIN_TEXT_WORDS = 15  # drops table-of-contents fragments like "Item 7. MD&A 23"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    company: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    section: str
    kind: str
    url: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    def header(self) -> str:
        return (
            f"{self.company} ({self.ticker}) {self.form}, period ending {self.report_date} "
            f"(fiscal {self.fiscal_year}) - {self.section}"
        )

    def indexed_text(self) -> str:
        # Contextual header is indexed with the body so "Apple 2024 10-K" style
        # queries can match chunks whose body never mentions the company/year.
        return f"{self.header()}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def _pack(paras: list[str], size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for p in paras:
        units.extend(_split_words(p, size, overlap))
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for u in units:
        w = len(u.split())
        if cur and n + w > size:
            chunks.append("\n".join(cur))
            tail = " ".join(chunks[-1].split()[-overlap:]) if overlap else ""
            cur, n = ([tail], len(tail.split())) if tail else ([], 0)
        cur.append(u)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _split_table(text: str, max_words: int) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    lines = text.split("\n")
    meta = [line for line in lines if line.startswith("[")]
    rows = [line for line in lines if not line.startswith("[")]
    header = meta + rows[:1]
    base = sum(len(line.split()) for line in header)
    out: list[str] = []
    cur: list[str] = []
    n = base
    for r in rows[1:]:
        w = len(r.split())
        if cur and n + w > max_words:
            out.append("\n".join(header + cur))
            cur, n = [], base
        cur.append(r)
        n += w
    if cur:
        out.append("\n".join(header + cur))
    return out


def chunk_blocks(blocks: list[Block], meta: FilingMeta, size: int = 300, overlap: int = 50) -> list[Chunk]:
    pieces: list[tuple[str, str, str]] = []  # (section, kind, text)
    cur_sec: str | None = None
    paras: list[str] = []

    def flush_text() -> None:
        if paras:
            pieces.extend((cur_sec or "", "text", t) for t in _pack(paras, size, overlap))
            paras.clear()

    for b in blocks:
        if b.kind == "table":
            flush_text()
            cur_sec = b.section
            pieces.extend((b.section, "table", t) for t in _split_table(b.text, size * 2))
            continue
        if b.section != cur_sec:
            flush_text()
            cur_sec = b.section
        paras.extend(p for p in b.text.split("\n") if p.strip())
    flush_text()

    chunks: list[Chunk] = []
    for section, kind, text in pieces:
        if kind == "text" and len(text.split()) < MIN_TEXT_WORDS:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{meta.accession}-{len(chunks):04d}",
                text=text,
                ticker=meta.ticker,
                company=meta.company,
                form=meta.form,
                filing_date=meta.filing_date,
                report_date=meta.report_date,
                accession=meta.accession,
                section=section,
                kind=kind,
                url=meta.url,
            )
        )
    return chunks
