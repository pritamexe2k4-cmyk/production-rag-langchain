"""
# GLUE (not from course): developer check that needs NO OpenAI key.

Runs the HR retrieval pipeline end to end with a LOCAL embedding model (sentence-transformers
all-MiniLM-L6-v2) instead of OpenAI, plus the real cross-encoder reranker, and prints
hit@k / recall@k and the refusal-threshold calibration for the HR eval questions.
Numbers differ from the OpenAI run (different embeddings) but show whether the corpus,
chunking, hybrid search and reranker work together. Needs `uv sync --extra rerank`.

    uv run python -m scripts.check_retrieval_offline [--no-rerank] [--no-hybrid]
"""
import argparse
import json
import os
import tempfile

os.environ.setdefault("OPENAI_API_KEY", "sk-offline-not-used")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.embeddings import Embeddings  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag import RAGService  # noqa: E402
from evaluation.eval_rag import HR_QUESTIONS, retrieval_metrics  # noqa: E402


class LocalEmbeddings(Embeddings):
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(name)

    def embed_documents(self, texts):
        return self.m.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-hybrid", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    s.chroma_dir = tempfile.mkdtemp()
    s.hybrid_search = not a.no_hybrid
    if a.no_rerank:
        s.reranker = "none"
    rag = RAGService(s, embeddings=LocalEmbeddings())
    print("ingest:", {k: v for k, v in rag.ingest("hr_docs").items() if k in ("added", "errors", "chunks_added")})
    qs = json.loads(HR_QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    for q in qs:
        res = rag.search(q["question"], k=5)
        rows.append({"q": q, "retrieved": [{"source": r.document.metadata["source"]} for r in res.ranked], "top_score": res.top_score})
    m = retrieval_metrics(rows)
    print(json.dumps(m, indent=2))
    ans = [r["top_score"] for r in rows if r["q"]["category"] != "out_of_scope" and r["top_score"] is not None]
    oos = [r["top_score"] for r in rows if r["q"]["category"] == "out_of_scope" and r["top_score"] is not None]
    print(f"top score, answerable: min={min(ans):.3f}  out-of-scope: max={max(oos):.3f}")
    print("lowest answerable:", sorted((round(r["top_score"], 3), r["q"]["id"]) for r in rows if r["q"]["category"] != "out_of_scope")[:5])
    print("out-of-scope scores:", sorted(((round(r["top_score"], 3), r["q"]["id"]) for r in rows if r["q"]["category"] == "out_of_scope"), reverse=True))
    misses = [r["q"]["id"] for r in rows if r["q"]["category"] != "out_of_scope" and not any(d["source"] in r["q"]["expected_docs"] for d in r["retrieved"])]
    print("missed (no expected doc in top-5):", misses)


if __name__ == "__main__":
    main()
