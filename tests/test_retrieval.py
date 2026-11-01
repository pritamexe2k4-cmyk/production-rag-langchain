"""# GLUE (not from course): tests for hybrid search, metadata filters, reranking and the refusal gate."""
import json

import httpx
import pytest

from app.config import get_settings
from app.rag import RAGService
from app.reranker import CohereReranker, CrossEncoderReranker, build_reranker
from tests.conftest import FakeReranker, HR_PATH


@pytest.fixture
def rag():
    r = RAGService(get_settings(), reranker=None)
    r.ingest(HR_PATH)
    return r


def test_hybrid_adds_keyword_matches_that_the_vector_search_misses(rag):
    """The test embeddings are random, so vector search is noise; BM25 must still surface the exact clause.
    (k is large so the fused list is not cut before we look at it.)"""
    rag.settings.hybrid_search = True
    hit = next(r for r in rag.search("sandwich rule", k=40).ranked if "sandwich rule" in r.document.page_content.lower())
    assert hit.bm25_rank == 1 and hit.document.metadata["source"] == "Leave_Policy.pdf"
    rag.settings.hybrid_search = False
    assert all(r.bm25_rank is None for r in rag.search("sandwich rule", k=40).ranked)


def test_every_result_carries_citation_metadata(rag):
    for r in rag.search("notice period band C", k=5).results:
        m = r.document.metadata
        assert m["doc_title"] and m["doc_type"] and m["source"] and m.get("effective_date")


def test_doc_type_filter(rag):
    rag.settings.hybrid_search = True
    res = rag.search("leave", k=8, doc_types=["faq"])
    assert res.results and {r.document.metadata["doc_type"] for r in res.results} == {"faq"}


def test_category_and_type_filters_combine(rag):
    rag.settings.hybrid_search = True
    res = rag.search("leave days", k=8, doc_types=["policy"], categories=["leave", "wfh"])
    assert res.results
    assert {r.document.metadata["category"] for r in res.results} <= {"leave", "wfh"}
    assert {r.document.metadata["doc_type"] for r in res.results} == {"policy"}


def test_filter_with_no_matches_refuses(rag):
    res = rag.search("leave", doc_types=["nonexistent"])
    assert res.results == [] and res.refused


def test_reranker_reorders_candidates_and_sets_scores(rag):
    rag.settings.hybrid_search = True
    q = "privilege leave band c entitlement days"
    before = [r.document.metadata["chunk_id"] for r in rag.search(q, k=5).ranked]
    rag.reranker = FakeReranker()
    res = rag.search(q, k=5)
    after = [r.document.metadata["chunk_id"] for r in res.ranked]
    assert all(r.rerank_score is not None for r in res.ranked)
    assert [r.rerank_score for r in res.ranked] == sorted((r.rerank_score for r in res.ranked), reverse=True)
    assert before != after


def test_rerank_threshold_refuses_when_nothing_relevant(rag):
    rag.reranker = FakeReranker()
    rag.settings.rerank_min_score = 0.5
    res = rag.search("xylophone quantum burrito", k=5)
    assert res.refused and res.results == [] and res.ranked and res.top_score is not None
    assert "RERANK_MIN_SCORE" in res.reason


def test_rerank_threshold_keeps_only_chunks_above_it(rag):
    rag.reranker = FakeReranker()
    rag.settings.rerank_min_score = 0.6
    res = rag.search("privilege leave band c entitlement days", k=5)
    assert res.results and all(r.rerank_score >= 0.6 for r in res.results)
    assert len(res.results) <= len(res.ranked)


def test_cohere_reranker_request_and_response_shape():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]})

    rr = CohereReranker("key123", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert rr.score("q", ["a", "b"]) == [0.1, 0.9]
    assert seen["auth"] == "Bearer key123" and seen["body"]["documents"] == ["a", "b"] and seen["body"]["query"] == "q"


def test_build_reranker_options(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "reranker", "none")
    assert build_reranker(s) is None
    monkeypatch.setattr(s, "reranker", "bogus")
    with pytest.raises(ValueError):
        build_reranker(s)
    monkeypatch.setattr(s, "reranker", "cohere")
    monkeypatch.setattr(s, "cohere_api_key", "")
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        build_reranker(s)


def test_cross_encoder_without_the_extra_gives_a_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # simulates "not installed"
    with pytest.raises(RuntimeError, match="uv sync --extra rerank"):
        CrossEncoderReranker("any/model")
