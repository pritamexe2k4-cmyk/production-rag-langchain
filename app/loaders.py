"""
# GLUE (not from course) - PHASE 1 ADDITION: multi-format document loading.

The course loads one PDF with PyPDFLoader (part1/document_loaders.py). Real HR folders
mix PDF, Word, Markdown and HTML, and policies depend on headings and tables, so this
module turns EVERY format into the same thing: Markdown text (headings as `#`, tables as
`| a | b |`) plus document metadata. The chunker (app/chunking.py) then works on one format.

  .pdf   pdfplumber: text + tables in reading order, repeated page headers/footers removed,
         numbered headings ("2.1 Casual Leave") detected. (PyPDFLoader flattens tables into
         one cell per line, e.g. "Band A\\n8\\n8\\n18", which breaks table questions.)
  .docx  python-docx: Heading styles -> `#`, tables -> Markdown tables
  .html  BeautifulSoup (bs4 is already a course dependency): h1-h4, p, ul, table; nav/script dropped
  .md    as is        .txt  as is

Metadata is read from a small block near the top of each document:
    Document type: policy | Category: leave | Effective date: 2027-01-01 | Version: 4.0 | Owner: ...
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}


@dataclass
class Segment:
    """A piece of a document as Markdown. For PDFs one segment = one page (1-based)."""

    text: str
    page: int | None = None


@dataclass
class ParsedDoc:
    path: Path
    source: str                 # file name, shown in citations
    title: str
    doc_type: str = "general"   # policy | guideline | handbook | calendar | faq | ...
    category: str = "general"   # leave | wfh | expenses | ...
    effective_date: str = ""    # ISO date, "" if unknown
    version: str = ""
    owner: str = ""
    fmt: str = ""
    file_hash: str = ""
    segments: list[Segment] = field(default_factory=list)

    @property
    def pages(self) -> int:
        return len(self.segments)

    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ metadata
_META_KEYS = {
    "document type": "doc_type", "category": "category", "effective date": "effective_date",
    "version": "version", "owner": "owner",
}
_META_RE = re.compile(r"^\s*(Document type|Category|Effective date|Version|Owner)\s*:\s*(.+?)\s*$", re.I | re.M)
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y")


def parse_date(value: str) -> str:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def extract_metadata(text: str) -> dict:
    """Read the 'Document type: ...' style block from the first ~3000 characters."""
    found = {}
    for key, value in _META_RE.findall(text[:3000]):
        found.setdefault(_META_KEYS[key.lower()], value)
    if "effective_date" in found:
        found["effective_date"] = parse_date(found["effective_date"])
    for k in ("doc_type", "category"):
        if k in found:
            found[k] = found[k].strip().lower().replace(" ", "_")
    return found


# ---------------------------------------------------------------------- tables
def table_to_markdown(rows: list[list]) -> str:
    rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in r] for r in rows if r and any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


# ------------------------------------------------------------------------- PDF
_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^.:;,!?]{2,80})$")


def strip_repeated_lines(pages: list[list[str]], min_pages: int = 3, share: float = 0.6) -> list[list[str]]:
    """Remove running headers/footers: lines in the first/last 2 lines of >= 60% of pages
    (digits normalised, so 'Page 1' and 'Page 2' count as the same line)."""
    if len(pages) < min_pages:
        return pages
    norm = lambda s: re.sub(r"\d+", "#", s.strip().lower())  # noqa: E731
    counts: Counter = Counter()
    for lines in pages:
        edge = {norm(x) for x in lines[:2] + lines[-2:] if x.strip()}
        counts.update(edge)
    repeated = {k for k, v in counts.items() if v >= max(2, share * len(pages))}
    return [[x for x in lines if norm(x) not in repeated or not x.strip()] for lines in pages]


def mark_headings(lines: list[str]) -> list[str]:
    """Turn 'N. Title' / 'N.M Title' lines into Markdown headings (## / ###)."""
    out = []
    for ln in lines:
        m = _HEADING_RE.match(ln.strip())
        if m:
            depth = m.group(1).count(".") + 1
            out.append("#" * (depth + 1) + " " + ln.strip())
        elif ln.lstrip().startswith("#"):
            out.append("\\" + ln)  # e.g. a code comment "# Initialize" must not become a heading
        else:
            out.append(ln)
    return out


def _pdf_page_lines(page) -> list[str]:
    """Page text in reading order, with each ruled table replaced by a Markdown table."""
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    x0, top, x1, bottom = page.bbox
    lines: list[str] = []
    cursor = top

    def text_between(a: float, b: float):
        if b - a > 1:
            t = page.crop((x0, a, x1, b)).extract_text() or ""
            lines.extend(t.splitlines())

    for t in tables:
        text_between(cursor, t.bbox[1])
        md = table_to_markdown(t.extract())
        if md:
            lines.extend(["", *md.splitlines(), ""])
        cursor = t.bbox[3]
    text_between(cursor, bottom)
    return lines


def load_pdf(path: Path) -> tuple[list[Segment], str | None]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        pages = [_pdf_page_lines(p) for p in pdf.pages]
    pages = strip_repeated_lines(pages)
    segments = [Segment("\n".join(mark_headings(lines)).strip(), page=i + 1) for i, lines in enumerate(pages)]
    first = next((ln.strip() for ln in (pages[0] if pages else []) if ln.strip()), None)
    return [s for s in segments if s.text], first


# ------------------------------------------------------------------------ DOCX
def load_docx(path: Path) -> tuple[list[Segment], str | None]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    out: list[str] = []
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, doc)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "") if p.style is not None else ""
            m = re.match(r"Heading (\d)", style)
            if m:
                out.append("#" * min(int(m.group(1)), 4) + " " + text)
            elif style == "Title":
                out.append("# " + text)
            elif "List" in style:
                out.append("- " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            out += ["", table_to_markdown([[c.text for c in r.cells] for r in Table(el, doc).rows]), ""]
    title = doc.core_properties.title or None
    return [Segment("\n".join(out).strip())], title


# ------------------------------------------------------------------------ HTML
def load_html(path: Path) -> tuple[list[Segment], str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    root = soup.body or soup
    out: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "pre"]):
        if el.name in ("p", "pre") and el.find_parent(["table", "ul", "ol"]):
            continue
        if el.name in ("ul", "ol") and el.find_parent(["ul", "ol", "table"]):
            continue
        if el.name == "table":
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            out += ["", table_to_markdown(rows), ""]
        elif el.name in ("ul", "ol"):
            out += [f"- {li.get_text(' ', strip=True)}" for li in el.find_all("li")]
        elif el.name[0] == "h":
            out.append("#" * int(el.name[1]) + " " + el.get_text(" ", strip=True))
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                out.append(txt)
    return [Segment("\n".join(out).strip())], title


# --------------------------------------------------------------------- MD / TXT
def load_text(path: Path) -> tuple[list[Segment], str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):  # optional YAML front matter -> plain 'key: value' lines
        _, _, rest = text[3:].partition("\n---")
        front, text = text[3:].partition("\n---")[0], rest.lstrip("\n")
        text = front.strip() + "\n\n" + text
    return [Segment(text.strip())], None


_LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".html": load_html, ".htm": load_html,
            ".md": load_text, ".txt": load_text}


def load_document(path: Path) -> ParsedDoc:
    """Parse one file into a ParsedDoc. Raises on unreadable files (caller reports it)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"unsupported file type: {suffix}")
    segments, title_hint = _LOADERS[suffix](path)
    if not segments or not any(s.text.strip() for s in segments):
        raise ValueError("no text could be extracted (scanned PDF? OCR is not part of this pipeline)")
    full = "\n\n".join(s.text for s in segments)
    meta = extract_metadata(full)
    h1 = re.search(r"^# (.+)$", full, re.M)
    if suffix == ".pdf":  # PDF text has no real H1; the first line of page 1 is the title
        title = title_hint or path.stem.replace("_", " ")
    else:
        title = (h1.group(1).strip() if h1 else None) or title_hint or path.stem.replace("_", " ")
    return ParsedDoc(
        path=path, source=path.name, title=title, fmt=suffix.lstrip("."), file_hash=file_sha256(path),
        segments=segments,
        **{k: v for k, v in meta.items() if k in ("doc_type", "category", "effective_date", "version", "owner")},
    )
