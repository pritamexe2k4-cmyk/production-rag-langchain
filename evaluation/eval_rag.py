"""
# GLUE (not from course): small evaluation harness for the RAG pipeline.

The checking ideas come from production-course-main-code/testing_patterns.py
(part5_production/testing_patterns.py):
  * keyword check          <- IntegrationTestSuite.test_basic_qa ("expected_contains", any-of)
  * contains_answer score  <- contains_answer() evaluator (share of key words > 3 chars found)
  * LLM-as-judge (--judge) <- correctness() evaluator prompt (Y / N)
  * refusal check          <- demo_rag_with_fallback() idea: out-of-scope questions must be refused

The questions (evaluation/questions.json) are about the course demo PDF
(data/docs/langchain_demo.pdf). It calls the agent directly, so the security/cache layers
are not involved. Every run is LangSmith-traced if tracing is on.

    uv run --env-file .env python -m evaluation.eval_rag --ingest
    uv run --env-file .env python -m evaluation.eval_rag --judge --hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from app.rag import REFUSAL

QUESTIONS = Path(__file__).with_name("questions.json")


def contains_answer_score(answer: str, expected: list[str]) -> float:
    """Port of testing_patterns.contains_answer: share of >3-char words from the reference found in the answer."""
    reference = " ".join(expected).lower()
    key_words = [w for w in reference.split() if len(w) > 3]
    if not key_words:
        return 1.0
    pred = answer.lower()
    return sum(1 for w in key_words if w in pred) / len(key_words)


def keyword_pass(answer: str, expected: list[str]) -> bool:
    """Port of IntegrationTestSuite: pass if ANY expected string appears."""
    return any(e.lower() in answer.lower() for e in expected)


def judge_correctness(llm, question: str, submission: str, reference: str) -> float:
    """Prompt copied from testing_patterns.correctness (LLM-as-judge)."""
    from langchain_core.prompts import ChatPromptTemplate

    grade_prompt = ChatPromptTemplate.from_template(
        "You are a grader. Given a question, a submission, and a reference answer, "
        "determine if the submission is correct, accurate, and factual compared to "
        "the reference answer.\n\n"
        "Question: {question}\n"
        "Submission: {submission}\n"
        "Reference: {reference}\n\n"
        "Respond with ONLY 'Y' if correct or 'N' if incorrect."
    )
    result = llm.invoke(grade_prompt.format(question=question, submission=submission, reference=reference))
    return 1.0 if str(result.content).strip().upper() == "Y" else 0.0


def run_eval(agent, questions: list[dict], judge_llm=None) -> dict:
    rows = []
    for q in questions:
        out = agent.invoke(q["question"], use_retrieval=True)
        answer = str(out["response"])
        if q["answerable"]:
            ok = keyword_pass(answer, q["expected_contains"]) and REFUSAL not in answer
            row = {"score": contains_answer_score(answer, q["expected_contains"]), "passed": ok}
            if judge_llm is not None:
                row["judge"] = judge_correctness(judge_llm, q["question"], answer, ", ".join(q["expected_contains"]))
            row["cited"] = bool(out["sources"])
        else:
            row = {"score": None, "passed": REFUSAL in answer or not out["sources"], "cited": False}
        rows.append({"id": q["id"], "question": q["question"], "answer": answer[:200], **row,
                     "sources": [f"{s['source']} p{s['page']}" for s in out["sources"]]})
    passed = sum(r["passed"] for r in rows)
    return {"total": len(rows), "passed": passed, "pass_rate": passed / len(rows) if rows else 0.0, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", action="store_true", help="ingest data/docs/langchain_demo.pdf first")
    ap.add_argument("--judge", action="store_true", help="also grade with an LLM judge (extra OpenAI calls)")
    ap.add_argument("--hybrid", action="store_true", help="HYBRID_SEARCH=true")
    ap.add_argument("--agentic", action="store_true", help="AGENTIC_RETRIEVAL=true")
    ap.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 if pass rate is lower")
    args = ap.parse_args(argv)
    if args.hybrid:
        os.environ["HYBRID_SEARCH"] = "true"
    if args.agentic:
        os.environ["AGENTIC_RETRIEVAL"] = "true"

    from langchain_openai import ChatOpenAI

    from app.agent import ProductionAgent
    from app.config import get_settings
    from app.rag import RAGService

    settings = get_settings()
    rag = RAGService(settings)
    if args.ingest or rag.count() == 0:
        print("Ingesting:", rag.ingest("langchain_demo.pdf"))
    agent = ProductionAgent(rag=rag)
    judge = ChatOpenAI(model=settings.primary_model, temperature=0, api_key=settings.openai_api_key) if args.judge else None

    report = run_eval(agent, json.loads(QUESTIONS.read_text(encoding="utf-8")), judge)
    for r in report["rows"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<16} score={r['score']} judge={r.get('judge')} sources={r['sources']}")
    print(f"\n{report['passed']}/{report['total']} passed ({report['pass_rate']:.0%})")
    return 0 if report["pass_rate"] >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
