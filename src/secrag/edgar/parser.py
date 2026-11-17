"""Turn EDGAR (inline-XBRL) HTML into an ordered list of section-labelled text/table blocks.

Why this matters: most numeric answers in a 10-K live in tables. Naive
`get_text()` flattens a table into a soup of numbers with no row labels,
which neither BM25 nor an embedding model can use. Here every table is
converted to `label | value | value` rows, tagged with its units
("in millions") and the caption line above it, and kept as its own block.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# EDGAR inline-XBRL filings are XHTML that start with an <?xml ...?> declaration.
# Parsing them as HTML is intended (lenient, keeps ix:* tags), so silence the warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class Block:
    section: str  # e.g. "Item 7 - MD&A"
    kind: str  # "text" | "table"
    text: str


TENK_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
    "9": "Changes in Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors and Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Relationships and Transactions",
    "14": "Accountant Fees",
    "15": "Exhibits",
}
TENQ_NAMES = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "MD&A",
    ("I", "3"): "Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity",
    ("II", "3"): "Defaults",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}

_ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
_PART_RE = re.compile(r"^part\s+(iv|iii|ii|i)\b", re.IGNORECASE)
_UNITS_RE = re.compile(r"in\s+(millions|thousands|billions)", re.IGNORECASE)
_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section"]


def _is_hidden(tag: Tag) -> bool:
    if not isinstance(tag, Tag) or not tag.name:
        return False
    if tag.name.lower() == "ix:header":
        return True
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _cell_text(cell: Tag) -> str:
    t = " ".join(cell.get_text(" ", strip=True).split())
    return re.sub(r"\s+([)%])", r"\1", re.sub(r"([($])\s+", r"\1", t))  # "( 101,321 )" -> "(101,321)"


def table_rows(table: Tag) -> list[str]:
    """Rows as 'cell | cell | ...', merging EDGAR's split '$', '(', ')' and '%' cells."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        out: list[str] = []
        prefix = ""
        for td in tr.find_all(["td", "th"]):
            t = _cell_text(td)
            if not t or t == "$":
                continue
            if t in {"(", "$("}:
                prefix = "("
                continue
            if t in {")", "%", ")%", "%)"} and out:
                out[-1] += t
                continue
            out.append(prefix + t.lstrip("$ ").strip() if t.startswith("$") else prefix + t)
            prefix = ""
        if out:
            row = " | ".join(out)
            if not rows or rows[-1] != row:
                rows.append(row)
    return rows


def _section_label(form: str, part: str | None, item: str, rest: str) -> str:
    if form.upper().startswith("10-Q"):
        name = TENQ_NAMES.get((part or "I", item))
    else:
        name = TENK_NAMES.get(item)
    if not name:
        name = rest[:60].strip(" .:-")
    return f"Item {item}" + (f" - {name}" if name else "")


def _looks_like_header(rest: str) -> bool:
    # "Item 7. MD&A" / "ITEM 1A" are headers; "Item 7 of this report ..." is not.
    return rest == "" or rest[0].isupper() or rest[0] in "\"'“"


def _is_noise(line: str) -> bool:
    low = line.lower()
    return bool(re.fullmatch(r"\d{1,3}", line)) or low in {"table of contents", "index"}


def parse_filing_html(html: str, form: str = "10-K") -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(_is_hidden):
        tag.decompose()

    # 1) tables -> placeholders (so they survive text extraction intact)
    tables: dict[str, str] = {}
    for i, table in enumerate(soup.find_all("table")):
        if table.find_parent("table") is not None:
            continue  # nested table: handled as part of its parent
        rows = table_rows(table)
        if len(rows) >= 2:
            key = f"@@TABLE{i}@@"
            tables[key] = "\n".join(rows)
            table.replace_with(NavigableString(f"\n{key}\n"))
        else:  # layout table, e.g. a heading split across cells: "Item 1A." | "Risk Factors"
            table.replace_with(NavigableString("\n" + " ".join(r.replace(" | ", " ") for r in rows) + "\n"))

    # 2) block elements -> line breaks, then split into clean lines
    for el in soup.find_all(_BLOCK_TAGS):
        if el.parent is not None:
            el.insert_after(NavigableString("\n"))
    lines = [" ".join(line.split()) for line in soup.get_text().split("\n")]
    lines = [line for line in lines if line and not _is_noise(line)]

    # 3) walk lines, tracking PART / ITEM headers
    blocks: list[Block] = []
    section, part = "Cover", None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append(Block(section, "text", "\n".join(buf)))
            buf.clear()

    for line in lines:
        if line in tables:
            context = buf[-2:]
            flush()
            prefix = []
            caption = next((c for c in reversed(context) if len(c) < 150 and not _UNITS_RE.search(c)), None)
            if caption:
                prefix.append(f"[Context: {caption}]")
            units = _UNITS_RE.search(" ".join(context) + " " + tables[line][:300])
            if units:
                prefix.append(f"[Amounts in {units.group(1).lower()}]")
            blocks.append(Block(section, "table", "\n".join(prefix + [tables[line]])))
            continue
        if len(line) < 80 and _PART_RE.match(line):
            part = _PART_RE.match(line).group(1).upper()
            continue
        m = _ITEM_RE.match(line) if len(line) < 200 else None
        if m and _looks_like_header(m.group(2)):
            flush()
            section = _section_label(form, part, m.group(1).upper(), m.group(2))
        buf.append(line)
    flush()
    return blocks
