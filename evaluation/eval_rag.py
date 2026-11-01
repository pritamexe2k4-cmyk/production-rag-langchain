"""
# GLUE (not from course) - PHASE 1: evaluation harness for the RAG pipeline.

Ideas taken from production-course-main-code/testing_patterns.py (part5_production/):
  * keyword check             <- IntegrationTestSuite ("expected_contains")
  * LLM-as-judge correctness  <- correctness() evaluator prompt (Y / N against a reference)
  * refusal check             <- demo_rag_with_fallback() idea: out-of-scope questions must be refused

Suites
  hr    54 questions over the fictional HR corpus (data/hr_docs): factual, table lookups, multi-doc, out-of-scope
  demo  8 questions over the course demo PDF (the original phase-0 eval)

What is measured
  Retrieval   hit@k, recall@k (k = 1, 3, 5) and MRR at DOCUMENT level (did the right file come back?)
  Answers     keyword check (stem-aware), LLM-judge correctness vs. a reference answer,
              LLM-judge faithfulness (is every claim supported by the retrieved context?)
  Refusal     refusal accuracy on out-of-scope questions, false-refusal rate on answerable ones
  Calibration the best top-score for every question, so you can pick RERANK_MIN_SCORE from data

    uv run --env-file .env python -m evaluation.eval_rag --ingest            # HR suite, writes evaluation/results.json
    uv run --env-file .env python -m evaluation.eval_rag --no-judge          # cheaper: no judge calls
    uv run --env-file .env python -m evaluation.eval_rag --no-rerank --limit 10
    uv run --env-file .env python -m evaluation.eval_rag --suite demo --ingest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.rag import REFUSAL

HERE = Path(__file__).parent
HR_QUESTIONS = HERE / "questions_hr.json"
DEMO_QUESTIONS = HERE / "questions_demo.json"
KS = (1, 3, 5)


# ----------------------------------------------------------- keyword matching
_TOKEN_RE = re.compile(
    r"[a-z0-9._-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"   # e-mail address
    r"|\d+(?:-\d+)+"                              # 1800-000-1234
    r"|\d+:\d+"                                   # 11:00
    r"|\d+\.\d+"                                  # 7.5
    r"|[a-z0-9]+"                                  # words and numbers ('90-day' -> '90', 'day')
)


def tokens(text: str) -> list[str]:
    """Lowercase tokens; '7,50,000' -> '750000'; keeps '11:00', '1800-000-1234', '7.5' and e-mail addresses whole."""
    return _TOKEN_RE.findall(re.sub(r"(?<=\d),(?=\d)", "", text.lower()))


def same_word(a: str, b: str) -> bool:
    """Tolerant word match: 'normalize' == 'normalizing' == 'normalization', 'day' == 'days'.
    (The phase-0 check used a plain substring test, so 'normalize' did NOT match 'normalizing'.)"""
    if a == b:
        return True
    if any(ch.isdigit() for ch in a + b):
        return False
    short, long_ = sorted((a, b), key=len)
    if len(short) < 6:
        return long_ in (short + "s", short + "es")
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= len(short) - 2


def term_in(term: str, toks: list[str]) -> bool:
    t = tokens(term)
    if not t:
        return True
    return any(all(same_word(toks[i + j], t[j]) for j in range(len(t))) for i in range(len(toks) - len(t) + 1))


def keyword_pass(answer: str, must_include: list[str]) -> bool:
    """Every entry must appear; an entry 'a|b' is satisfied by either alternative."""
    toks = tokens(answer)
    return all(any(term_in(alt, toks) for alt in group.split("|")) for group in must_include)


# ------------------------------------------------------------------ judges
GRADE_PROMPT = (  # copied from testing_patterns.correctness
    "You are a grader. Given a question, a submission, and a reference answer, "
    "determine if the submission is correct, accurate, and factual compared to "
    "the reference answer.\n\n"
    "Question: {question}\nSubmission: {submission}\nReference: {reference}\n\n"
    "Respond with ONLY 'Y' if correct or 'N' if incorrect."
)
FAITHFUL_PROMPT = (  # same style, new criterion
    "You are a grader. Given a context and an answer, determine if EVERY factual claim in the answer "
    "is supported by the context. Ignore citation markers like [1].\n\n"
    "Context:\n{context}\n\nAnswer: {answer}\n\n"
    "Respond with ONLY 'Y' if fully supported or 'N' if any claim is not supported."
)


def _yn(llm, prompt: str) -> float:
    return 1.0 if str(llm.invoke(prompt).content).strip().upper().startswith("Y") else 0.0


def judge_correctness(llm, question: str, submission: str, reference: str) -> float:
    return _yn(llm, GRADE_PROMPT.format(question=question, submission=submission, reference=reference))


def judge_faithfulness(llm, context: list[str], answer: str) -> float:
    return _yn(llm, FAITHFUL_PROMPT.format(context="\n\n".join(context), answer=answer))


# --------------------------------------------------------------- retrieval metrics
def _doc_rank(retrieved: list[dict], expected: list[str]) -> list[int | None]:
    """For each expected doc, the 1-based rank of its first chunk in the retrieved list (None = missing)."""
    order = [r["source"] for r in retrieved]
    return [order.index(d) + 1 if d in order else None for d in expected]


def retrieval_metrics(rows: list[dict]) -> dict:
    """rows: [{'q': question dict, 'retrieved': [{'source': ...}, ...]}]. Answerable questions only.
    doc_mode 'any': one expected doc is enough (hit=1 recall=1). 'all': recall = share of expected docs found."""
    rows = [r for r in rows if r["q"]["category"] != "out_of_scope" and r["q"]["expected_docs"]]
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows)}
    for k in KS:
        hits, recalls = [], []
        for r in rows:
            q = r["q"]
            ranks = _doc_rank(r["retrieved"][:k], q["expected_docs"])
            found = [x for x in ranks if x is not None]
            hits.append(1.0 if found else 0.0)
            recalls.append((1.0 if found else 0.0) if q.get("doc_mode", "any") == "any" else len(found) / len(q["expected_docs"]))
        out[f"hit@{k}"] = round(sum(hits) / len(hits), 3)
        out[f"recall@{k}"] = round(sum(recalls) / len(recalls), 3)
    rr = []
    for r in rows:
        first = [x for x in _doc_rank(r["retrieved"], r["q"]["expected_docs"]) if x is not None]
        rr.append(1 / min(first) if first else 0.0)
    out["mrr"] = round(sum(rr) / len(rr), 3)
    return out


def calibrate(rows: list[dict]) -> dict:
    """Pick the score threshold that best separates answerable from out-of-scope questions."""
    ans = [r["top_score"] for r in rows if r["q"]["category"] != "out_of_scope" and r["top_score"] is not None]
    oos = [r["top_score"] for r in rows if r["q"]["category"] == "out_of_scope" and r["top_score"] is not None]
    if not ans or not oos:
        return {}
    best_t, best_acc = None, -1.0
    for t in sorted(set(ans + oos)):
        acc = (sum(s >= t for s in ans) + sum(s < t for s in oos)) / (len(ans) + len(oos))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return {
        "answerable_top_score_min": round(min(ans), 4),
        "answerable_top_score_p10": round(sorted(ans)[max(0, int(0.1 * len(ans)) - 1)], 4),
        "out_of_scope_top_score_max": round(max(oos), 4),
        "separable": min(ans) > max(oos),
        "suggested_threshold": round(best_t, 4),
        "accuracy_at_suggested_threshold": round(best_acc, 3),
    }


# ----------------------------------------------------------------------- run
def load_questions(suite: str) -> list[dict]:
    if suite == "hr":
        return json.loads(HR_QUESTIONS.read_text(encoding="utf-8"))
    old = json.loads(DEMO_QUESTIONS.read_text(encoding="utf-8"))  # phase-0 format
    return [{
        "id": q["id"], "category": "factual" if q["answerable"] else "out_of_scope", "question": q["question"],
        "expected_docs": ["langchain_demo.pdf"] if q["answerable"] else [], "doc_mode": "any",
        "must_include": ["|".join(q["expected_contains"])] if q["expected_contains"] else [],
        "reference": ", ".join(q["expected_contains"]) or "Not in the document; the assistant must refuse.",
    } for q in old]


def run_eval(agent, questions: list[dict], judge_llm=None, k: int = 5) -> dict:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        out = agent.invoke(q["question"], use_retrieval=True, top_k=k)
        latency = (time.perf_counter() - t0) * 1000
        answer = str(out["response"])
        refused = bool(out.get("refused", REFUSAL in answer))
        retrieved = out.get("retrieved") or out.get("sources", [])
        row = {"q": q, "answer": answer, "refused": refused, "latency_ms": round(latency, 1),
               "top_score": out.get("top_score"), "retrieved": retrieved, "sources": out.get("sources", [])}
        if q["category"] == "out_of_scope":
            row["passed"] = refused
        else:
            row["keyword_pass"] = (not refused) and keyword_pass(answer, q["must_include"])
            row["passed"] = row["keyword_pass"]
            row["cited"] = any(s.get("cited") for s in out.get("sources", []))
            if judge_llm is not None:
                row["judge_correct"] = 0.0 if refused else judge_correctness(judge_llm, q["question"], answer, q["reference"])
                row["faithful"] = None if refused else judge_faithfulness(judge_llm, out.get("context") or [r.get("text", r["snippet"]) for r in out.get("sources", [])], answer)
        rows.append(row)

    ans = [r for r in rows if r["q"]["category"] != "out_of_scope"]
    oos = [r for r in rows if r["q"]["category"] == "out_of_scope"]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    judged = [r for r in ans if "judge_correct" in r]
    faith = [r["faithful"] for r in judged if r["faithful"] is not None]
    by_cat = {}
    for cat in sorted({r["q"]["category"] for r in rows}):
        rc = [r for r in rows if r["q"]["category"] == cat]
        by_cat[cat] = {"n": len(rc), "passed": mean([float(r["passed"]) for r in rc])}
        if cat != "out_of_scope":
            by_cat[cat]["hit@5"] = retrieval_metrics(rc).get("hit@5")
            if judged:
                by_cat[cat]["judge_correct"] = mean([r["judge_correct"] for r in rc if "judge_correct" in r])
    metrics = {
        "questions": len(rows), "answerable": len(ans), "out_of_scope": len(oos),
        "pass_rate": mean([float(r["passed"]) for r in rows]),
        "retrieval": retrieval_metrics(rows),
        "answers": {
            "keyword_accuracy": mean([float(r["keyword_pass"]) for r in ans]),
            "judge_correctness": mean([r["judge_correct"] for r in judged]) if judged else None,
            "faithfulness": mean(faith) if faith else None,
            "citation_rate": mean([float(r["cited"]) for r in ans if not r["refused"]]),
            "false_refusal_rate": mean([float(r["refused"]) for r in ans]),
        },
        "refusal": {"accuracy": mean([float(r["refused"]) for r in oos]), "n": len(oos)},
        "by_category": by_cat,
        "latency_ms": {"mean": round(statistics.mean(r["latency_ms"] for r in rows), 1) if rows else None,
                       "max": max((r["latency_ms"] for r in rows), default=None)},
        "calibration": calibrate(rows),
    }
    return {"metrics": metrics, "rows": rows}


def print_report(rep: dict) -> None:
    m = rep["metrics"]
    line = "-" * 78
    print(f"\n{line}\nRESULTS: {m['questions']} questions ({m['answerable']} answerable, {m['out_of_scope']} out-of-scope)\n{line}")
    for r in rep["rows"]:
        q = r["q"]
        extra = f" judge={r['judge_correct']:.0f}" if "judge_correct" in r else ""
        extra += f" top={r['top_score']:.3f}" if r["top_score"] is not None else ""
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {q['id']:<5} {q['category']:<12}{extra}  {q['question'][:58]}")
    print(f"\nOverall pass rate            {m['pass_rate']:.0%}")
    rt = m["retrieval"]
    if rt.get("n"):
        print(f"Retrieval  (n={rt['n']})          " + "  ".join(f"hit@{k}={rt[f'hit@{k}']:.2f} recall@{k}={rt[f'recall@{k}']:.2f}" for k in KS) + f"  MRR={rt['mrr']:.2f}")
    a = m["answers"]
    fmt = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"Answers    keyword={fmt(a['keyword_accuracy'])}  judge-correct={fmt(a['judge_correctness'])}  faithful={fmt(a['faithfulness'])}  cited={fmt(a['citation_rate'])}  false-refusal={fmt(a['false_refusal_rate'])}")
    print(f"Refusal    out-of-scope refused: {fmt(m['refusal']['accuracy'])} ({m['refusal']['n']} questions)")
    for cat, c in m["by_category"].items():
        print(f"  {cat:<13} n={c['n']:<3} passed={fmt(c['passed'])}" + (f"  hit@5={c['hit@5']}" if c.get("hit@5") is not None else "") + (f"  judge={fmt(c['judge_correct'])}" if c.get("judge_correct") is not None else ""))
    c = m["calibration"]
    if c:
        print(f"Threshold  answerable top-score min={c['answerable_top_score_min']} (p10 {c['answerable_top_score_p10']}), "
              f"out-of-scope max={c['out_of_scope_top_score_max']}, separable={c['separable']}\n"
              f"           best single threshold ~ {c['suggested_threshold']} ({c['accuracy_at_suggested_threshold']:.0%} of questions on the right side)")
    print(f"Latency    mean {m['latency_ms']['mean']} ms, max {m['latency_ms']['max']} ms\n{line}")
    failed = [r for r in rep["rows"] if not r["passed"]]
    if failed:
        print("FAILED questions (full answers are in the results file):")
        for r in failed:
            print(f"  {r['q']['id']}: {r['q']['question']}\n     -> {r['answer'][:160]!r}\n     retrieved: {[s['source'] for s in r['retrieved'][:5]]}")


def write_results(rep: dict, path: Path, settings, suite: str) -> None:
    keys = ("embedding_model", "primary_model", "chunk_size", "chunk_overlap", "retrieval_k", "hybrid_search", "reranker",
            "rerank_model", "rerank_min_score", "min_relevance_score", "agentic_retrieval", "candidates")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "suite": suite,
        "config": {k: getattr(settings, k) for k in keys}, "metrics": rep["metrics"],
        "questions": [{
            "id": r["q"]["id"], "category": r["q"]["category"], "question": r["q"]["question"], "reference": r["q"]["reference"],
            "expected_docs": r["q"]["expected_docs"], "passed": r["passed"], "answer": r["answer"], "refused": r["refused"],
            "keyword_pass": r.get("keyword_pass"), "judge_correct": r.get("judge_correct"), "faithful": r.get("faithful"),
            "top_score": r["top_score"], "latency_ms": r["latency_ms"],
            "retrieved": [{k: v for k, v in s.items() if k not in ("text", "snippet")} for s in r["retrieved"]],
        } for r in rep["rows"]],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["hr", "demo"], default="hr")
    ap.add_argument("--ingest", action="store_true", help="ingest the suite's documents first (incremental)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judges (fewer OpenAI calls)")
    ap.add_argument("--no-rerank", action="store_true", help="RERANKER=none")
    ap.add_argument("--no-hybrid", action="store_true", help="HYBRID_SEARCH=false (vector only)")
    ap.add_argument("--agentic", action="store_true", help="AGENTIC_RETRIEVAL=true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 if the pass rate is lower")
    args = ap.parse_args(argv)
    if args.no_rerank:
        os.environ["RERANKER"] = "none"
    if args.no_hybrid:
        os.environ["HYBRID_SEARCH"] = "false"
    if args.agentic:
        os.environ["AGENTIC_RETRIEVAL"] = "true"
    if args.suite == "demo":
        os.environ.setdefault("COLLECTION_NAME", "demo_langchain")

    from langchain_openai import ChatOpenAI

    from app.agent import ProductionAgent
    from app.config import get_settings
    from app.rag import RAGService

    settings = get_settings()
    rag = RAGService(settings)
    path = "hr_docs" if args.suite == "hr" else "docs/langchain_demo.pdf"
    if args.ingest or rag.count() == 0:
        rep = rag.ingest(path)
        print("Ingest:", {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items() if k != "files"})
    agent = ProductionAgent(rag=rag)
    judge = None if args.no_judge else ChatOpenAI(model=settings.primary_model, temperature=0, api_key=settings.openai_api_key)

    questions = load_questions(args.suite)
    if args.limit:
        questions = questions[: args.limit]
    rep = run_eval(agent, questions, judge, k=5)
    print_report(rep)
    write_results(rep, Path(args.out), settings, args.suite)
    print(f"\nWrote {args.out}")
    return 0 if rep["metrics"]["pass_rate"] >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
