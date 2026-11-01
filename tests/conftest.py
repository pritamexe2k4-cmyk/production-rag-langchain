"""
# GLUE (not from course): shared test setup.

Tests never call OpenAI. Following the mock approach in the course's
testing_patterns.py (Mock LLM -> deterministic answer), we swap the two network
classes (OpenAIEmbeddings, ChatOpenAI) for fakes. No real key is needed.
"""
import os

# must be set BEFORE app.* is imported (settings are read at import time)
os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["RERANKER"] = "none"
os.environ.pop("LANGCHAIN_API_KEY", None)

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


CALLS: list = []  # every prompt the fake LLM received (reset per test)
EMBED_CALLS: list = []  # size of every embed_documents() batch (reset per test)
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
DEMO_PATH = "docs/langchain_demo.pdf"   # relative to data/
DEMO_FILE = "langchain_demo.pdf"
HR_PATH = "hr_docs"


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings that remember how many texts were embedded per call."""

    def embed_documents(self, texts):
        EMBED_CALLS.append(len(texts))
        return super().embed_documents(texts)


class FakeReranker:
    """Deterministic stand-in for the cross-encoder: score = share of query words found in the chunk."""

    name = "fake-reranker"

    def score(self, query, texts):
        q = set(__import__("re").findall(r"[a-z0-9]+", query.lower()))
        return [len(q & set(__import__("re").findall(r"[a-z0-9]+", t.lower()))) / max(len(q), 1) for t in texts]


class FakeChat(BaseChatModel):
    """Answers '1.0' to grading prompts, a rewritten query to rewrite prompts,
    otherwise echoes the first source line of the RAG prompt it was given."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = str(messages[-1].content)
        CALLS.append(text)
        if "Relevance score (0-1)" in text:
            out = "1.0" if "PyPDFLoader" in text else "0.0"
        elif "Rewritten query:" in text:
            out = "PyPDFLoader PDF loader"
        elif "Context:" in text:
            ctx = text.split("Context:")[1].split("Question:")[0]
            out = "From the docs: " + ctx.strip().splitlines()[0] if ctx.strip() else "I don't have information about that in my knowledge base."
        else:
            out = "plain chat reply"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])


@pytest.fixture(autouse=True)
def _fakes(monkeypatch, tmp_path):
    import app.agent as agent_mod
    import app.rag as rag_mod
    from app.config import get_settings

    CALLS.clear()
    EMBED_CALLS.clear()
    monkeypatch.setattr(rag_mod, "OpenAIEmbeddings", lambda **kw: CountingEmbeddings(size=64))
    monkeypatch.setattr(agent_mod, "ChatOpenAI", lambda **kw: FakeChat())

    s = get_settings()
    monkeypatch.setattr(s, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(s, "data_dir", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data"))
    monkeypatch.setattr(s, "hybrid_search", False)
    monkeypatch.setattr(s, "reranker", "none")  # phase 1: no model download in tests; reranker tests inject a fake
    monkeypatch.setattr(s, "rerank_min_score", 0.02)
    monkeypatch.setattr(s, "retrieval_k", 5)
    monkeypatch.setattr(s, "candidates", 20)
    monkeypatch.setattr(s, "chunk_size", 1000)
    monkeypatch.setattr(s, "chunk_overlap", 100)
    monkeypatch.setattr(s, "agentic_retrieval", False)
    monkeypatch.setattr(s, "ingest_api_key", "")
    monkeypatch.setattr(s, "min_relevance_score", 0.0)
    yield
