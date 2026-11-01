"""
LangGraph Agent with Production Error Handling
Retry logic, model fallback, and structured state management.

SOURCE: lang-production-api  app/agent.py.
The original nodes (process / fallback / error) and their routing are kept as-is.
Everything marked `# GLUE (not from course)` was added by us, because the
course's production agent has NO retrieval. The glue adds:

    START -> retrieve -> [grade <-> rewrite] -> prepare -> process -> ...
                         (nothing found) -> no_context -> END

  retrieve   : app/rag.py (course chunk/embed/Chroma/hybrid code)
  grade      : prompt copied from part6_advanced/04_agentic_rag.py (grade_documents)
  rewrite    : prompt copied from part6_advanced/04_agentic_rag.py (rewrite_query)
  router     : same logic as 04_agentic_rag.py (should_retry_or_generate)
  prepare    : builds the course RAG prompt (part1/rag_pipeline.py) for `process`
"""

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from app.config import get_settings
from app.rag import RAG_PROMPT, REFUSAL, RAGService, Retrieved, format_docs_with_sources


# === Agent State ===

class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    # --- GLUE (not from course): retrieval state ---
    use_retrieval: bool
    question: str
    rewritten_query: str
    rewrites: int
    documents: list          # list[app.rag.Retrieved]
    relevance_score: float
    llm_input: Optional[list[BaseMessage]]  # what `process`/`fallback` send to the LLM
    # phase 1: filters + diagnostics
    doc_types: Optional[list[str]]
    categories: Optional[list[str]]
    top_k: Optional[int]
    ranked: list             # top-k before the refusal gate (for the eval's recall@k)
    top_score: Optional[float]


def source_item(r: Retrieved, n: int, answer: str = "", full: bool = False) -> dict:
    """GLUE (phase 1): one citation. `cited` = the model wrote [n] in its answer."""
    m = r.document.metadata
    return {
        "n": n,
        "source": m.get("source", "unknown"),
        "doc_title": m.get("doc_title", ""),
        "section": m.get("section", ""),
        "doc_type": m.get("doc_type", ""),
        "effective_date": m.get("effective_date", ""),
        "page": m.get("page"),
        "score": round(r.score, 4) if r.score is not None else None,
        "vector_score": round(r.vector_score, 4) if r.vector_score is not None else None,
        "rerank_score": round(r.rerank_score, 4) if r.rerank_score is not None else None,
        "cited": f"[{n}]" in answer,
        "snippet": r.document.page_content[:200],
        **({"text": r.document.page_content} if full else {}),  # full chunk text (evaluation only)
    }


# GLUE (not from course): prompts copied from part6_advanced/04_agentic_rag.py
GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Given a user query and a document,
determine if the document contains information relevant to answering the query.

Output ONLY a number between 0 and 1:
- 1.0 = Highly relevant, directly answers the query
- 0.7 = Somewhat relevant, contains related information
- 0.3 = Marginally relevant, tangentially related
- 0.0 = Not relevant at all

Output ONLY the number, nothing else."""),
    ("human", """Query: {query}

Document: {document}

Relevance score (0-1):"""),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a RAG system.
The original query didn't retrieve relevant documents.

Rewrite the query to be more specific and likely to match relevant documents.
Consider:
- Adding synonyms or related terms
- Being more specific about what information is needed
- Rephrasing to match how documentation is typically written

Output ONLY the rewritten query, nothing else."""),
    ("human", """Original query: {query}

Rewritten query:"""),
])

    
# === Agent Builder ===

class ProductionAgent:
    """
    Production LangGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    - (GLUE) retrieval over the Chroma knowledge base
    """

    def __init__(self, rag: Optional[RAGService] = None):
        settings = get_settings()
        self.settings = settings
        self.rag = rag  # GLUE (not from course)

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # We handle retries ourselves
            api_key=settings.openai_api_key,
        )
        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine."""

        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model."""
            try:
                # GLUE: use the RAG prompt when `prepare` built one
                response = self.primary_llm.invoke(state.get("llm_input") or state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                response = self.fallback_llm.invoke(state.get("llm_input") or state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request "
                        "right now. Please try again in a moment."
                    ))
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # ------------------------------------------------------------------
        # GLUE (not from course): retrieval nodes
        # ------------------------------------------------------------------
        def route_start(state: AgentState) -> str:
            return "retrieve" if state.get("use_retrieval") and self.rag else "process"

        def retrieve(state: AgentState) -> dict:
            query = state.get("rewritten_query") or state["question"]
            res = self.rag.search(
                query, k=state.get("top_k"), doc_types=state.get("doc_types"), categories=state.get("categories")
            )
            return {"documents": res.results, "ranked": res.ranked, "top_score": res.top_score}

        def grade(state: AgentState) -> dict:
            """04_agentic_rag.py::grade_documents - LLM scores each chunk, keep >= 0.5."""
            docs = state["documents"]
            scores, kept = [], []
            for r in docs:
                result = (GRADING_PROMPT | self.primary_llm).invoke(
                    {"query": state["question"], "document": r.document.page_content}
                )
                try:
                    score = float(str(result.content).strip())
                except ValueError:
                    score = 0.5  # course default if parsing fails
                scores.append(score)
                if score >= 0.5:
                    kept.append(r)
            avg = sum(scores) / len(scores) if scores else 0.0
            return {"documents": kept, "relevance_score": avg}

        def rewrite(state: AgentState) -> dict:
            """04_agentic_rag.py::rewrite_query."""
            result = (REWRITE_PROMPT | self.primary_llm).invoke({"query": state["question"]})
            return {
                "rewritten_query": str(result.content).strip(),
                "rewrites": state.get("rewrites", 0) + 1,
            }

        def route_after_retrieve(state: AgentState) -> str:
            if self.settings.agentic_retrieval and state["documents"]:
                return "grade"
            return "prepare" if state["documents"] else "no_context"

        def route_after_grade(state: AgentState) -> str:
            """04_agentic_rag.py::should_retry_or_generate"""
            docs = state["documents"]
            if state.get("relevance_score", 0) >= 0.5 and docs:
                return "prepare"
            if state.get("rewrites", 0) < self.settings.max_rewrites:
                return "rewrite"
            return "prepare" if docs else "no_context"

        def prepare(state: AgentState) -> dict:
            """Build the course RAG prompt (rag_pipeline.py) from the retrieved chunks."""
            context = format_docs_with_sources([r.document for r in state["documents"]])
            prompt = RAG_PROMPT.format(
                refusal=REFUSAL, context=context, question=state["question"]
            )
            return {"llm_input": [HumanMessage(content=prompt)]}

        def no_context(state: AgentState) -> dict:
            """Nothing relevant retrieved: refuse without calling the LLM (no hallucination)."""
            return {
                "messages": [AIMessage(content=REFUSAL)],
                "documents": [],
                "model_used": "no_context",
            }

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)
        # GLUE nodes
        graph.add_node("retrieve", retrieve)
        graph.add_node("grade", grade)
        graph.add_node("rewrite", rewrite)
        graph.add_node("prepare", prepare)
        graph.add_node("no_context", no_context)

        # (original: graph.add_edge(START, "process"))
        graph.add_conditional_edges(
            START, route_start, {"retrieve": "retrieve", "process": "process"}
        )
        graph.add_conditional_edges(
            "retrieve", route_after_retrieve,
            {"grade": "grade", "prepare": "prepare", "no_context": "no_context"},
        )
        graph.add_conditional_edges(
            "grade", route_after_grade,
            {"prepare": "prepare", "rewrite": "rewrite", "no_context": "no_context"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("prepare", "process")
        graph.add_edge("no_context", END)
        # END GLUE

        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str, use_retrieval: bool = False, doc_types: Optional[list[str]] = None,
               categories: Optional[list[str]] = None, top_k: Optional[int] = None) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None, ...}

        GLUE: `use_retrieval=True` runs the RAG path (used by /query);
        the default keeps the course's original plain-chat behaviour (used by /chat).
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
            # GLUE
            "use_retrieval": use_retrieval,
            "question": message,
            "rewritten_query": "",
            "rewrites": 0,
            "documents": [],
            "relevance_score": 0.0,
            "llm_input": None,
            "doc_types": doc_types,
            "categories": categories,
            "top_k": top_k,
            "ranked": [],
            "top_score": None,
        })

        response = result["messages"][-1].content
        # GLUE: don't show sources when the model itself said it doesn't know
        refused = REFUSAL in str(response)
        docs = [] if refused else result.get("documents", [])
        return {
            "response": response,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            # GLUE
            "sources": [source_item(r, i + 1, str(response)) for i, r in enumerate(docs)],
            "rewritten_query": result.get("rewritten_query") or None,
            "refused": refused,
            "top_score": result.get("top_score"),
            # top-k before the refusal gate, for evaluation (recall@k) and debugging
            "retrieved": [source_item(r, i + 1, full=True) for i, r in enumerate(result.get("ranked", []))],
            "context": [r.document.page_content for r in docs],  # what the LLM saw (faithfulness judge)
        }
