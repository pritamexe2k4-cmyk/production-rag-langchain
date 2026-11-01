"""# GLUE (not from course): tests for the retrieval nodes added to the course's LangGraph agent."""
from app.agent import ProductionAgent
from app.config import get_settings
from app.rag import RAGService, REFUSAL
from tests.conftest import CALLS, FakeChat


def rag_with_pdf():
    rag = RAGService(get_settings())
    rag.ingest("langchain_demo.pdf")
    return rag


def test_plain_chat_skips_retrieval():
    agent = ProductionAgent(rag=rag_with_pdf())
    out = agent.invoke("hello")
    assert out["response"] == "plain chat reply" and out["sources"] == []


def test_rag_answer_has_sources():
    agent = ProductionAgent(rag=rag_with_pdf())
    out = agent.invoke("Which loader loads PDFs?", use_retrieval=True)
    assert out["response"].startswith("From the docs:")
    assert out["sources"] and out["sources"][0]["source"] == "langchain_demo.pdf"
    assert out["sources"][0]["page"] in (1, 2, 3)


def test_empty_store_refuses_without_calling_llm():
    agent = ProductionAgent(rag=RAGService(get_settings()))
    out = agent.invoke("anything", use_retrieval=True)
    assert out["response"] == REFUSAL and out["model_used"] == "no_context"
    assert CALLS == []


def test_agentic_loop_grades_and_keeps_relevant_docs():
    s = get_settings()
    s.agentic_retrieval = True
    agent = ProductionAgent(rag=rag_with_pdf())
    out = agent.invoke("pdf loader please", use_retrieval=True)
    # fake grader only scores chunks containing PyPDFLoader as relevant
    assert any("Relevance score" in c for c in CALLS)
    assert out["sources"]


def test_agentic_rewrites_then_refuses_when_nothing_relevant(monkeypatch):
    s = get_settings()
    s.agentic_retrieval = True
    s.max_rewrites = 1
    monkeypatch.setattr(s, "agentic_retrieval", True)
    rag = RAGService(get_settings())
    rag.ingest("langchain_demo.pdf")
    # make the fake grader reject everything
    import app.agent as agent_mod
    class Strict(FakeChat):
        def _generate(self, messages, *a, **k):
            if "Relevance score (0-1)" in str(messages[-1].content):
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatGeneration, ChatResult
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="0.0"))])
            return super()._generate(messages, *a, **k)
    monkeypatch.setattr(agent_mod, "ChatOpenAI", lambda **kw: Strict())
    out = ProductionAgent(rag=rag).invoke("pizza recipe", use_retrieval=True)
    assert out["response"] == REFUSAL
    assert out["rewritten_query"]  # it tried a rewrite first
