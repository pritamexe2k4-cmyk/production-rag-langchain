"""# GLUE (not from course): tests for the evaluation harness and for the HR question set itself."""
import json

import pytest

from app.agent import ProductionAgent
from app.config import get_settings
from app.loaders import load_document
from app.rag import RAGService
from evaluation.eval_rag import (
    HR_QUESTIONS, calibrate, keyword_pass, load_questions, print_report, retrieval_metrics, run_eval, same_word,
    term_in, tokens, write_results,
)
from tests.conftest import HR_PATH, ROOT


# ------------------------------------------------------------ matching
def test_preprocessing_regression_normalizing_matches_normalize():
    """The phase-0 eval failed 'What does the document say about preprocessing?' although the right page was
    retrieved: the answer said 'normalizing' and the check was `'normalize' in answer`."""
    assert "normalize" not in "Clean and normalizing the text".lower()          # why the old check failed
    assert keyword_pass("Clean and normalizing text after loading improves embedding quality.", ["normalize"])
    assert keyword_pass("It recommends normalization of the text.", ["normalize"])
    assert not keyword_pass("It says to split the text into chunks.", ["normalize"])


def test_word_and_number_matching():
    assert same_word("day", "days") and same_word("approve", "approval") and not same_word("leave", "lease")
    assert not same_word("8", "18") and not same_word("12", "120")
    assert keyword_pass("Rs 7,50,000 as a family floater", ["7,50,000|7.5 lakh"])
    assert keyword_pass("about 7.5 lakh rupees", ["7,50,000|7.5 lakh"])
    assert keyword_pass("Core hours are 11:00 AM to 4:00 PM IST", ["11:00", "4:00"])
    assert keyword_pass("Call 1800-000-1234 any time.", ["1800-000-1234"])
    assert not keyword_pass("You get 18 days", ["8"])                       # 18 is not 8
    assert keyword_pass("Write to payroll@acme-analytics.example.", ["payroll@acme-analytics.example"])
    assert term_in("working days", tokens("within 10 working days of leaving"))


# --------------------------------------------------------- question set
def test_hr_question_set_shape():
    qs = load_questions("hr")
    cats = {c: sum(q["category"] == c for q in qs) for c in {q["category"] for q in qs}}
    assert 40 <= len(qs) <= 60 and len({q["id"] for q in qs}) == len(qs)
    assert cats["out_of_scope"] >= 8 and cats["table"] >= 8 and cats["multi_doc"] >= 8 and cats["factual"] >= 15


def test_hr_questions_are_grounded_in_the_corpus():
    """Every expected document exists, and every required fact really appears in at least one expected document."""
    texts = {f.name: load_document(f).full_text() for f in (ROOT / "data" / "hr_docs").iterdir() if not f.name.startswith("_")}
    for q in load_questions("hr"):
        if q["category"] == "out_of_scope":
            assert q["expected_docs"] == [] and q["must_include"] == []
            continue
        assert q["expected_docs"] and all(d in texts for d in q["expected_docs"]), q["id"]
        toks = [t for d in q["expected_docs"] for t in tokens(texts[d])]
        for group in q["must_include"]:
            assert any(term_in(alt, toks) for alt in group.split("|")), (q["id"], group)
        assert keyword_pass(q["reference"], q["must_include"]), (q["id"], "reference answer must pass its own check")


def test_demo_suite_is_converted_to_the_new_format():
    qs = load_questions("demo")
    assert len(qs) == 8 and any(q["category"] == "out_of_scope" for q in qs)
    assert all({"must_include", "expected_docs", "reference"} <= set(q) for q in qs)


# ----------------------------------------------------------------- metrics
def row(cat, docs, retrieved, mode="any", top=None):
    return {"q": {"category": cat, "expected_docs": docs, "doc_mode": mode}, "retrieved": [{"source": s} for s in retrieved], "top_score": top}


def test_retrieval_metrics_hit_recall_mrr():
    rows = [
        row("factual", ["a"], ["x", "a", "y"]),                      # found at rank 2
        row("factual", ["b"], ["x", "y", "z", "w", "b"]),            # found at rank 5
        row("multi_doc", ["a", "b"], ["a", "x", "b"], mode="all"),   # a@1, b@3
        row("out_of_scope", [], ["a"]),                              # ignored
    ]
    m = retrieval_metrics(rows)
    assert m["n"] == 3
    assert (m["hit@1"], m["hit@3"], m["hit@5"]) == (0.333, 0.667, 1.0)
    assert m["recall@1"] == round((0 + 0 + 0.5) / 3, 3) and m["recall@3"] == round((1 + 0 + 1) / 3, 3) and m["recall@5"] == 1.0
    assert m["mrr"] == round((1 / 2 + 1 / 5 + 1) / 3, 3)


def test_calibration_picks_a_separating_threshold():
    rows = [row("factual", ["a"], [], top=0.9), row("factual", ["a"], [], top=0.6), row("out_of_scope", [], [], top=0.1), row("out_of_scope", [], [], top=0.3)]
    c = calibrate(rows)
    assert c["separable"] and 0.3 < c["suggested_threshold"] <= 0.6 and c["accuracy_at_suggested_threshold"] == 1.0
    assert calibrate(rows[:2]) == {}


# --------------------------------------------------------------- end to end
def test_run_eval_end_to_end_with_fakes_and_results_file(tmp_path, capsys):
    rag = RAGService(get_settings(), reranker=None)
    rag.ingest(HR_PATH)
    qs = load_questions("hr")[:6] + [q for q in load_questions("hr") if q["category"] == "out_of_scope"][:2]

    class FakeJudge:
        def invoke(self, prompt):
            from langchain_core.messages import AIMessage
            return AIMessage(content="Y")

    rep = run_eval(ProductionAgent(rag=rag), qs, FakeJudge())
    m = rep["metrics"]
    assert m["questions"] == 8 and m["out_of_scope"] == 2 and m["retrieval"]["n"] == 6
    assert m["answers"]["judge_correctness"] is not None and m["answers"]["faithfulness"] is not None
    assert all("retrieved" in r and "answer" in r for r in rep["rows"])
    print_report(rep)
    assert "Overall pass rate" in capsys.readouterr().out
    out = tmp_path / "results.json"
    write_results(rep, out, get_settings(), "hr")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["suite"] == "hr" and len(data["questions"]) == 8 and "config" in data and "calibration" in data["metrics"]
    assert "answer" in data["questions"][0] and "text" not in json.dumps(data["questions"][0]["retrieved"])
