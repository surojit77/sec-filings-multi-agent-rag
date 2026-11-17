from conftest import META

from secrag.edgar.xbrl import FactStore, extract_facts
from secrag.index.chunker import chunk_blocks


def test_sections_and_hidden_content(blocks):
    sections = [b.section for b in blocks]
    assert "Item 1A - Risk Factors" in sections
    assert "Item 7 - MD&A" in sections
    full = "\n".join(b.text for b in blocks)
    assert "999,999" not in full  # hidden ix:header removed
    # "Item 7 of this report ..." is prose, not a new header -> stays in Item 7
    prose = next(b for b in blocks if "should be read together" in b.text)
    assert prose.section == "Item 7 - MD&A"


def test_table_conversion(blocks):
    tables = [b for b in blocks if b.kind == "table" and "Net sales" in b.text]
    assert len(tables) == 1
    t = tables[0]
    assert t.section == "Item 7 - MD&A"
    assert "[Amounts in millions]" in t.text
    assert "[Context: CONSOLIDATED STATEMENTS OF OPERATIONS]" in t.text
    assert "Net sales | 391,035 | 383,285" in t.text
    assert "Other income/(expense), net | (269) | (565)" in t.text
    assert "Gross margin percentage | 46.2% | 44.1%" in t.text


def test_chunking(blocks):
    chunks = chunk_blocks(blocks, META, size=60, overlap=10)
    assert all(c.ticker == "AAPL" and c.fiscal_year == 2024 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    text_chunks = [c for c in chunks if c.kind == "text"]
    assert all(len(c.text.split()) <= 60 + 10 + 40 for c in text_chunks)
    # TOC fragments are too short to be chunks
    assert not any(c.text.strip().startswith("Item 1A. | Risk Factors") and c.kind == "text" for c in chunks)
    table = next(c for c in chunks if c.kind == "table" and "Net sales" in c.text)
    assert "Apple Inc. (AAPL) 10-K" in table.indexed_text()


def test_xbrl_extraction(tmp_path):
    cf = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2023-10-01", "end": "2024-09-28", "val": 93736000000, "accn": "A1", "fy": 2024,
             "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"start": "2024-06-30", "end": "2024-09-28", "val": 14736000000, "accn": "A1", "fy": 2024,
             "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"start": "2023-10-01", "end": "2024-09-28", "val": 1, "accn": "X", "form": "8-K", "filed": "2024-11-01"},
        ]}},
        "Assets": {"units": {"USD": [
            {"end": "2024-09-28", "val": 364980000000, "accn": "A1", "form": "10-K", "filed": "2024-11-01"}]}},
    }}}
    facts = extract_facts("aapl", cf)
    assert {f.period for f in facts} == {"FY", "Q", "instant"}
    assert all(f.form != "8-K" for f in facts)
    store = FactStore(tmp_path)
    store.save("AAPL", facts)
    store2 = FactStore(tmp_path)
    fy = store2.query(["AAPL"], years={2024}, periods={"FY"})
    assert len(fy) == 1 and fy[0].value == 93736000000
