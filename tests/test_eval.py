"""# GLUE (not from course): checks the evaluation harness plumbing offline (fake LLM)."""
import json

from app.agent import ProductionAgent
from app.config import get_settings
from app.rag import RAGService
from evaluation.eval_rag import QUESTIONS, contains_answer_score, keyword_pass, run_eval


def test_scoring_helpers():
    assert keyword_pass("Use PyPDFLoader.", ["PyPDFLoader"])
    assert not keyword_pass("no idea", ["PyPDFLoader"])
    assert contains_answer_score("lazy_load helps", ["lazy_load"]) == 1.0


def test_run_eval_pipeline_end_to_end_with_fakes():
    rag = RAGService(get_settings())
    rag.ingest("langchain_demo.pdf")
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    report = run_eval(ProductionAgent(rag=rag), qs)
    assert report["total"] == len(qs) and len(report["rows"]) == len(qs)
    assert all("sources" in r for r in report["rows"])
