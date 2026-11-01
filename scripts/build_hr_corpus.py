"""
# GLUE (not from course): builds the sample HR corpus in data/hr_docs/ from the Markdown
# sources in scripts/hr_corpus_src/, rendering each document to a different format
# (PDF, DOCX, Markdown, HTML) so the ingestion pipeline is tested on mixed formats.

All content is FICTIONAL ("Acme Analytics Pvt Ltd" does not exist).
Only needed if you want to regenerate the files; the generated files are committed.

    uv run python scripts/build_hr_corpus.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "hr_corpus_src"
OUT = ROOT / "data" / "hr_docs"

# source file -> (output name, format)
PLAN = {
    "01_leave_policy.md": ("Leave_Policy.pdf", "pdf"),
    "02_hybrid_wfh_policy.md": ("Hybrid_Work_and_WFH_Policy.docx", "docx"),
    "03_code_of_conduct.md": ("Code_of_Conduct.pdf", "pdf"),
    "04_posh_policy.md": ("POSH_Policy.docx", "docx"),
    "05_expense_travel_policy.md": ("Expense_and_Travel_Reimbursement_Policy.md", "md"),
    "06_onboarding_guide.md": ("New_Joiner_Onboarding_Guide.html", "html"),
    "07_performance_review_policy.md": ("Performance_Review_Policy.md", "md"),
    "08_notice_period_exit_policy.md": ("Notice_Period_and_Exit_Policy.pdf", "pdf"),
    "09_benefits_insurance_guide.md": ("Employee_Benefits_and_Insurance_Guide.docx", "docx"),
    "10_it_security_policy.md": ("IT_Security_and_Acceptable_Use_Policy.pdf", "pdf"),
    "11_holiday_list_2027.md": ("Holiday_List_2027.html", "html"),
    "12_employee_handbook.md": ("Employee_Handbook.docx", "docx"),
    "13_compensation_payroll_policy.md": ("Compensation_and_Payroll_Policy.md", "md"),
    "14_hr_faq.md": ("HR_FAQ.md", "md"),
}


def parse(md: str):
    """Tiny markdown parser -> list of blocks: (kind, payload)."""
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            blocks.append(("h", (level, ln.lstrip("#").strip())))
            i += 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            blocks.append(("ul", items))
        else:
            blocks.append(("p", ln.strip()))
            i += 1
    return blocks


def title_of(blocks):
    return next(p[1] for k, p in blocks if k == "h" and p[0] == 1)


def to_html(blocks, title):
    out = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
           "<style>body{font-family:Arial;max-width:900px;margin:auto}table,td,th{border:1px solid #999;border-collapse:collapse;padding:4px}</style>",
           "<script>var analytics = 'ignored by the loader';</script></head><body>",
           "<nav><a href='/'>Home</a> | <a href='/policies'>Policies</a> | <a href='/benefits'>Benefits</a></nav><main>"]
    for k, p in blocks:
        if k == "h":
            out.append(f"<h{p[0]}>{html.escape(p[1])}</h{p[0]}>")
        elif k == "p":
            out.append(f"<p>{html.escape(p)}</p>")
        elif k == "ul":
            out.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in p) + "</ul>")
        else:
            head, *body = p
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr></thead><tbody>")
            out += ["<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body]
            out.append("</tbody></table>")
    out.append("</main><footer>Acme Analytics Pvt Ltd (fictional) - Internal use only - Copyright 2026</footer></body></html>")
    return "\n".join(out)


def to_docx(blocks, path):
    from docx import Document

    d = Document()
    for k, p in blocks:
        if k == "h":
            d.add_heading(p[1], level=min(p[0], 3))
        elif k == "p":
            d.add_paragraph(p)
        elif k == "ul":
            for x in p:
                d.add_paragraph(x, style="List Bullet")
        else:
            t = d.add_table(rows=len(p), cols=len(p[0]), style="Table Grid")
            for r, row in enumerate(p):
                for c, cell in enumerate(row):
                    t.cell(r, c).text = cell
                    if r == 0:
                        for run in t.cell(r, c).paragraphs[0].runs:
                            run.bold = True
            d.add_paragraph("")
    d.save(path)


def to_pdf(blocks, title, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    h = {1: ss["Title"], 2: ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, spaceBefore=10),
         3: ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11.5, spaceBefore=6)}
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=13.5)
    cell = ParagraphStyle("c", parent=body, fontSize=8.5, leading=10.5)

    def esc(t):
        return html.escape(t)

    story = []
    for k, p in blocks:
        if k == "h":
            story.append(Paragraph(esc(p[1]), h[min(p[0], 3)]))
        elif k == "p":
            story.append(Paragraph(esc(p), body))
        elif k == "ul":
            story.append(ListFlowable([ListItem(Paragraph(esc(x), body)) for x in p], bulletType="bullet", start="-", leftIndent=14))
        else:
            data = [[Paragraph(esc(c), cell) for c in row] for row in p]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 4 * mm)]

    def deco(canvas, doc):  # repeated header/footer on every page (ingestion must strip these)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 287 * mm, f"Acme Analytics Pvt Ltd - {title} - FICTIONAL SAMPLE")
        canvas.drawString(20 * mm, 10 * mm, f"Confidential - Internal use only - Page {doc.page}")

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm,
                      bottomMargin=20 * mm, title=title, author="Acme Analytics Pvt Ltd (fictional)").build(story, onFirstPage=deco, onLaterPages=deco)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for src, (name, fmt) in PLAN.items():
        md = (SRC / src).read_text(encoding="utf-8")
        blocks = parse(md)
        title = title_of(blocks)
        dest = OUT / name
        if fmt == "md":
            dest.write_text(md, encoding="utf-8")
        elif fmt == "html":
            dest.write_text(to_html(blocks, title), encoding="utf-8")
        elif fmt == "docx":
            to_docx(blocks, dest)
        elif fmt == "pdf":
            to_pdf(blocks, title, dest)
        print(f"{src:40s} -> {dest.relative_to(ROOT)}")
    (OUT / "_ABOUT_THIS_CORPUS.md").write_text(
        "# About this corpus\n\nEvery document in this folder is FICTIONAL sample data for testing a RAG system. "
        "Acme Analytics Pvt Ltd does not exist; names, numbers and policies are invented. Do not use them as HR or legal advice.\n\n"
        "Files starting with an underscore are skipped by the ingestion pipeline.\n\n"
        "Regenerate with `uv run python scripts/build_hr_corpus.py` (sources in scripts/hr_corpus_src/).\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
