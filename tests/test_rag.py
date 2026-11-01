"""# GLUE (not from course): tests for app/rag.py (ingest + retrieve) with fake embeddings."""
import pytest

from app.config import get_settings
from app.rag import RAGService, REFUSAL


def make():
    return RAGService(get_settings())


def test_ingest_demo_pdf_gives_course_chunk_count():
    stats = make().ingest("langchain_demo.pdf")
    # demo PDF: 3 pages, split at 500/50 (course default)
    assert stats["pages"] == 3
    assert stats["chunks_added"] == stats["total_chunks_in_store"] > 0


def test_ingest_is_idempotent():
    rag = make()
    first = rag.ingest("langchain_demo.pdf")
    second = rag.ingest("langchain_demo.pdf")
    assert second["chunks_added"] == 0
    assert second["total_chunks_in_store"] == first["total_chunks_in_store"]


def test_retrieve_on_empty_store_returns_nothing():
    assert make().retrieve("anything") == []


def test_retrieve_returns_sources_with_page_and_score():
    rag = make()
    rag.ingest("langchain_demo.pdf")
    hits = rag.retrieve("PyPDFLoader", k=3)
    assert len(hits) == 3
    assert all(h.document.metadata["source"] == "langchain_demo.pdf" for h in hits)
    assert all(0 < h.score <= 1 for h in hits)


def test_hybrid_mode_keyword_hit_is_returned():
    rag = make()
    rag.settings.hybrid_search = True
    rag.ingest("langchain_demo.pdf")
    hits = rag.retrieve("PyPDFLoader", k=5)
    assert any("PyPDFLoader" in h.document.page_content for h in hits)


def test_min_relevance_score_filters_everything():
    rag = make()
    rag.ingest("langchain_demo.pdf")
    rag.settings.min_relevance_score = 0.99999
    assert rag.retrieve("PyPDFLoader") == []


@pytest.mark.parametrize("bad", ["../pyproject.toml", "../../etc/passwd", "/etc/passwd"])
def test_path_traversal_blocked(bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        make().ingest(bad)
