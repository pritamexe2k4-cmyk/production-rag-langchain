"""# GLUE (not from course): tests for app/loaders.py (multi-format loading, tables, headers)."""
from pathlib import Path

import pytest

from app.loaders import (
    extract_metadata, load_document, mark_headings, parse_date, strip_repeated_lines, table_to_markdown,
)
from tests.conftest import ROOT

HR = ROOT / "data" / "hr_docs"


@pytest.mark.parametrize("name,fmt", [
    ("Leave_Policy.pdf", "pdf"), ("POSH_Policy.docx", "docx"), ("HR_FAQ.md", "md"), ("Holiday_List_2027.html", "html"),
])
def test_each_format_loads_with_metadata(name, fmt):
    d = load_document(HR / name)
    assert d.fmt == fmt and d.title and d.doc_type in {"policy", "faq", "calendar"}
    assert d.effective_date.startswith("20") and d.version and d.category


def test_pdf_table_becomes_markdown_not_one_cell_per_line():
    text = load_document(HR / "Leave_Policy.pdf").full_text()
    assert "| Band C | 8 | 8 | 22 | 36 |" in text


def test_pdf_running_header_and_footer_removed():
    text = load_document(HR / "Leave_Policy.pdf").full_text()
    assert "FICTIONAL SAMPLE" not in text and "Internal use only - Page" not in text


def test_pdf_headings_detected_and_pages_numbered():
    d = load_document(HR / "Leave_Policy.pdf")
    assert [s.page for s in d.segments] == [1, 2, 3]
    assert "## 2. Types of leave" in d.full_text() and "### 2.3 Privilege Leave (PL)" in d.full_text()
    assert d.title == "Leave Policy"


def test_docx_headings_and_tables():
    text = load_document(HR / "Employee_Benefits_and_Insurance_Guide.docx").full_text()
    assert "## 2. Group Medical Insurance" in text and "| Band C | Rs 7,50,000 |" in text


def test_html_drops_nav_script_footer_keeps_table():
    text = load_document(HR / "Holiday_List_2027.html").full_text()
    assert "ignored by the loader" not in text and "Home" not in text and "Copyright" not in text
    assert "| 11 | 29 October 2027 | Friday | Diwali (Lakshmi Puja) |" in text


def test_markdown_front_matter_and_plain_text(tmp_path):
    (tmp_path / "a.md").write_text("---\nDocument type: guideline\nEffective date: 15 March 2026\n---\n# My Doc\n\nBody text.", encoding="utf-8")
    d = load_document(tmp_path / "a.md")
    assert d.title == "My Doc" and d.doc_type == "guideline" and d.effective_date == "2026-03-15"
    (tmp_path / "b.txt").write_text("just text", encoding="utf-8")
    assert load_document(tmp_path / "b.txt").title == "b"


def test_unsupported_and_empty_files_raise(tmp_path):
    (tmp_path / "x.xyz").write_text("hi")
    (tmp_path / "e.md").write_text("   ")
    with pytest.raises(ValueError):
        load_document(tmp_path / "x.xyz")
    with pytest.raises(ValueError):
        load_document(tmp_path / "e.md")


def test_strip_repeated_lines_and_headings():
    pages = [["HEADER", "text a", "Page 1"], ["HEADER", "text b", "Page 2"], ["HEADER", "text c", "Page 3"]]
    assert strip_repeated_lines(pages) == [["text a"], ["text b"], ["text c"]]
    assert strip_repeated_lines(pages[:2]) == pages[:2]  # too few pages to be sure
    assert mark_headings(["1. Purpose", "2.1 Casual Leave", "1.1 This is a sentence.", "# comment"]) == [
        "## 1. Purpose", "### 2.1 Casual Leave", "1.1 This is a sentence.", "\\# comment"]


def test_metadata_helpers():
    assert parse_date("01 April 2026") == "2026-04-01" and parse_date("garbage") == ""
    m = extract_metadata("# T\nDocument type: Policy\nCategory: IT Security\nEffective date: 2026-01-15\nVersion: 5.1")
    assert m == {"doc_type": "policy", "category": "it_security", "effective_date": "2026-01-15", "version": "5.1"}
    assert table_to_markdown([["a", "b"], ["1", "2\n3"]]) == "| a | b |\n|---|---|\n| 1 | 2 3 |"
