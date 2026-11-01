"""# GLUE (not from course): tests for app/chunking.py (header-aware chunking)."""
from pathlib import Path

from app.chunking import chunk_document
from app.loaders import ParsedDoc, Segment, load_document
from tests.conftest import ROOT

HR = ROOT / "data" / "hr_docs"


def doc(*segments, title="Test Doc", **kw):
    return ParsedDoc(path=Path("t.md"), source="t.md", title=title, fmt="md", segments=[Segment(t, p) for t, p in segments], **kw)


def test_sections_become_chunks_with_breadcrumb_and_metadata():
    d = doc(("# Test Doc\n\n## 1. Intro\nHello there.\n\n## 2. Rules\n\n### 2.1 Leave\nTake it.", None),
            doc_type="policy", category="leave", effective_date="2027-01-01", version="4.0")
    chunks = chunk_document(d)
    leave = next(c for c in chunks if c.metadata["heading"] == "2.1 Leave")
    assert leave.page_content.startswith("Test Doc > 2. Rules > 2.1 Leave\n")
    m = leave.metadata
    assert (m["doc_title"], m["doc_type"], m["category"], m["effective_date"], m["version"], m["section"]) == (
        "Test Doc", "policy", "leave", "2027-01-01", "4.0", "2. Rules > 2.1 Leave")
    assert "page" not in m and m["chunk_id"]


def test_heading_is_carried_across_pdf_pages():
    d = doc(("## 1. Rules\nPage one text.", 1), ("Still the rules, continued on page two.", 2))
    chunks = chunk_document(d)
    p2 = next(c for c in chunks if c.metadata["page"] == 2)
    assert p2.metadata["section"] == "1. Rules" and p2.metadata["page"] == 2


def test_long_section_is_split_and_every_piece_keeps_the_breadcrumb():
    body = " ".join(f"Sentence number {i} about leave." for i in range(200))
    chunks = chunk_document(doc((f"## 1. Long\n{body}", None)), chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(c.page_content.startswith("Test Doc > 1. Long\n") for c in chunks)
    assert all(len(c.page_content) < 500 + 60 for c in chunks)


def test_long_table_repeats_header_row():
    rows = "\n".join(f"| Band {i} | {i * 100} |" for i in range(80))
    table = f"| Band | Amount |\n|---|---|\n{rows}"
    chunks = chunk_document(doc((f"## 1. T\n{table}", None)), chunk_size=400, chunk_overlap=0)
    assert len(chunks) > 2
    assert all("| Band | Amount |" in c.page_content for c in chunks)


def test_chunk_ids_are_stable_and_unique_per_document():
    d = load_document(HR / "Leave_Policy.pdf")
    a, b = chunk_document(d), chunk_document(d)
    ids = [c.metadata["chunk_id"] for c in a]
    assert ids == [c.metadata["chunk_id"] for c in b] and len(set(ids)) == len(ids)


def test_identical_chunks_in_one_file_are_deduplicated():
    d = doc(("## 1. A\nSame text.\n\n## 1. A\nSame text.", None))
    assert len(chunk_document(d)) == 1


def test_real_documents_chunk_cleanly():
    for f in HR.iterdir():
        if f.name.startswith("_"):
            continue
        chunks = chunk_document(load_document(f))
        assert chunks, f.name
        assert all(c.metadata["doc_title"] and c.page_content.strip() for c in chunks), f.name
        assert max(len(c.page_content) for c in chunks) < 1300, f.name
