"""# GLUE (not from course): tests for app/rag.py with fake embeddings (no OpenAI)."""
import pytest

from app.config import get_settings
from app.rag import REFUSAL, RAGService
from tests.conftest import DEMO_FILE, DEMO_PATH, EMBED_CALLS


def make(**kw):
    return RAGService(get_settings(), **kw)


def test_ingest_demo_pdf():
    stats = make().ingest(DEMO_PATH)
    assert stats["pages"] == 3
    assert stats["chunks_added"] == stats["total_chunks_in_store"] > 0
    assert stats["added"] == [DEMO_FILE] and stats["errors"] == {}


def test_ingest_is_idempotent():
    rag = make()
    first = rag.ingest(DEMO_PATH)
    second = rag.ingest(DEMO_PATH)
    assert second["chunks_added"] == 0 and second["unchanged"] == [DEMO_FILE]
    assert second["total_chunks_in_store"] == first["total_chunks_in_store"]


def test_search_on_empty_store_refuses():
    res = make().search("anything")
    assert res.results == [] and res.refused and res.reason == "empty index"


def test_vector_search_returns_scores_and_metadata():
    rag = make()
    rag.ingest(DEMO_PATH)
    res = rag.search("PyPDFLoader", k=3)
    assert len(res.results) == 3
    assert all(h.document.metadata["source"] == DEMO_FILE for h in res.results)
    assert all(0 < h.vector_score <= 1 for h in res.results)
    assert all(h.document.metadata["doc_title"] for h in res.results)


def test_hybrid_mode_keyword_hit_is_returned():
    rag = make()
    rag.settings.hybrid_search = True
    rag.ingest(DEMO_PATH)
    res = rag.search("PyPDFLoader", k=5)
    assert any("PyPDFLoader" in h.document.page_content for h in res.results)
    assert any(h.bm25_rank for h in res.results)


def test_min_relevance_score_gate_refuses_but_keeps_top_score():
    rag = make()
    rag.ingest(DEMO_PATH)
    rag.settings.min_relevance_score = 0.99999
    res = rag.search("PyPDFLoader")
    assert res.results == [] and res.refused and res.top_score is not None and res.ranked


@pytest.mark.parametrize("bad", ["../pyproject.toml", "../../etc/passwd", "/etc/passwd"])
def test_path_traversal_blocked(bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        make().ingest(bad)


def test_documents_listing():
    rag = make()
    rag.ingest(DEMO_PATH)
    docs = rag.list_documents()
    assert docs[0]["path"] == DEMO_PATH and docs[0]["chunks"] > 0


def test_embeddings_are_sent_in_batches():
    rag = make()
    rag.settings.embed_batch_size = 4
    rag.ingest(DEMO_PATH)
    assert len(EMBED_CALLS) >= 2 and max(EMBED_CALLS) <= 4
    assert sum(EMBED_CALLS) == rag.count()
