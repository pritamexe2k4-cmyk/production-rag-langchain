"""
# GLUE (not from course) - PHASE 1 ADDITION: header-aware chunking.

Course pattern, extended. part1/text_splitters.py already shows MarkdownHeaderTextSplitter
and RecursiveCharacterTextSplitter; here they are chained:

  1. MarkdownHeaderTextSplitter  -> one piece per section ("# Title > ## 2. Types > ### 2.3 Privilege Leave")
  2. RecursiveCharacterTextSplitter (chunk_size / chunk_overlap) -> only for sections that are too long
  3. every chunk starts with a breadcrumb line "Title > Section" (a static version of the
     "contextual chunk header" idea of part6_advanced/02_contextual_retrieval.py), so the
     embedding and the BM25 index both know which document/section the text belongs to
  4. a long table that has to be split keeps its header row at the top of every piece

PDF pages are chunked page by page, and the heading that was open at the end of a page is
carried to the next page, so a chunk at the top of page 2 still knows its section.
"""
from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.loaders import ParsedDoc

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
_LEVELS = [h[1] for h in HEADERS]


def _carry_prefix(path: dict[str, str]) -> str:
    return "\n".join("#" * (i + 1) + " " + path[k] for i, k in enumerate(_LEVELS) if k in path)


def _repeat_table_header(pieces: list[str]) -> list[str]:
    header = None
    out = []
    for piece in pieces:
        lines = piece.splitlines()
        for i, ln in enumerate(lines[:-1]):  # remember the header row of the latest table
            if ln.startswith("|") and re.match(r"^\|[-| ]+\|$", lines[i + 1]):
                header = [ln, lines[i + 1]]
        if header and lines and lines[0].startswith("|") and lines[0] != header[0] and not re.match(r"^\|[-| ]+\|$", lines[0]):
            piece = "\n".join(header + lines)
        out.append(piece)
    return out


def chunk_id(meta: dict, text: str) -> str:
    """Stable id: same file + same metadata + same text -> same id (this is the content-hash dedup)."""
    key = "|".join(str(meta.get(k, "")) for k in ("source", "doc_type", "effective_date", "version", "page", "section"))
    return hashlib.sha1(f"{key}|{text}".encode()).hexdigest()


def chunk_document(doc: ParsedDoc, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[Document] = []
    seen: set[str] = set()
    carry: dict[str, str] = {"h1": doc.title}

    for seg in doc.segments:
        text = (_carry_prefix(carry) + "\n" + seg.text) if carry else seg.text
        sections = header_splitter.split_text(text)
        for sec in sections:
            body = re.sub(r" +\n", "\n", sec.page_content).strip()
            if not body:
                continue
            path = {k: v for k, v in sec.metadata.items() if k in _LEVELS}
            names = [path[k] for k in _LEVELS if k in path]
            section = " > ".join(names[1:]) if names and names[0] == doc.title else " > ".join(names)
            crumb = " > ".join([doc.title] + ([section] if section else []))
            pieces = [body] if len(body) <= chunk_size else _repeat_table_header(size_splitter.split_text(body))
            for piece in pieces:
                meta = {
                    "source": doc.source, "doc_title": doc.title, "doc_type": doc.doc_type, "category": doc.category,
                    "section": section, "heading": names[-1] if names else doc.title, "fmt": doc.fmt,
                }
                if doc.effective_date:
                    meta["effective_date"] = doc.effective_date
                if doc.version:
                    meta["version"] = doc.version
                if seg.page is not None:
                    meta["page"] = seg.page  # 1-based
                content = f"{crumb}\n{piece}"
                cid = chunk_id(meta, content)
                if cid in seen:  # identical chunk twice in one file -> keep one
                    continue
                seen.add(cid)
                meta["chunk_id"] = cid
                chunks.append(Document(page_content=content, metadata=meta))
        if sections:  # carry the last open heading path to the next page
            last = {k: v for k, v in sections[-1].metadata.items() if k in _LEVELS}
            carry = last or carry
    return chunks
