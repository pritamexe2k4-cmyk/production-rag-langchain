"""# GLUE (not from course): tests for the retrieval nodes added to the course's LangGraph agent."""
from app.agent import ProductionAgent
from app.config import get_settings
from app.rag import REFUSAL, RAGService
from tests.conftest import CALLS, DEMO_FILE, DEMO_PATH, HR_PATH, FakeChat, FakeReranker


def rag_with(path=DEMO_PATH, **kw):
    rag = RAGService(get_settings(), **kw)
    rag.ingest(path)
    return rag


def test_plain_chat_skips_retrieval():
    out = ProductionAgent(rag=rag_with()).invoke("hello")
    assert out["response"] == "plain chat reply" and out["sources"] == []


def test_rag_answer_has_cited_sources_with_title_and_section():
    agent = ProductionAgent(rag=rag_with(HR_PATH))
    out = agent.invoke("What is the sandwich rule?", use_retrieval=True)
    assert out["response"].startswith("From the docs:")
    s = out["sources"][0]
    assert s["n"] == 1 and s["doc_title"] and s["source"] and s["effective_date"]
    assert any("[1]" in c for c in CALLS)                 # prompt told the model how to cite
    assert out["refused"] is False and out["retrieved"] and "text" in out["retrieved"][0]
    assert out["context"]


def test_empty_store_refuses_without_calling_llm():
    out = ProductionAgent(rag=RAGService(get_settings())).invoke("anything", use_retrieval=True)
    assert out["response"] == REFUSAL and out["model_used"] == "no_context" and out["refused"] is True
    assert CALLS == []


def test_rerank_gate_refuses_without_calling_llm_but_reports_scores():
    s = get_settings()
    s.rerank_min_score = 0.9
    agent = ProductionAgent(rag=rag_with(HR_PATH, reranker=FakeReranker()))
    out = agent.invoke("xylophone quantum burrito", use_retrieval=True)
    assert out["response"] == REFUSAL and out["sources"] == [] and CALLS == []
    assert out["retrieved"] and out["top_score"] is not None   # eval can still see what was retrieved


def test_filters_and_top_k_are_passed_to_retrieval():
    s = get_settings()
    s.hybrid_search = True
    agent = ProductionAgent(rag=rag_with(HR_PATH))
    out = agent.invoke("leave", use_retrieval=True, doc_types=["faq"], top_k=2)
    assert len(out["sources"]) == 2 and {x["doc_type"] for x in out["sources"]} == {"faq"}


def test_agentic_loop_grades_and_keeps_relevant_docs():
    get_settings().agentic_retrieval = True
    out = ProductionAgent(rag=rag_with()).invoke("pdf loader please", use_retrieval=True)
    assert any("Relevance score" in c for c in CALLS)
    assert out["sources"]


def test_agentic_rewrites_then_refuses_when_nothing_relevant(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "agentic_retrieval", True)
    s.max_rewrites = 1
    rag = rag_with()
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
    assert out["rewritten_query"]
