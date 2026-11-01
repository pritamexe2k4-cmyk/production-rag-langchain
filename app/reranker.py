"""
# GLUE (not from course) - PHASE 1 ADDITION: reranking.

The course README lists "Retrieval Noise: 10 docs retrieved, only 2 are relevant -> reranking"
as failure mode #3 and its Part 6 requirements mention sentence-transformers, but no script
implements a reranker. This is the usual pattern:

    retrieve 20 candidates cheaply (vector + BM25)  ->  score each (query, chunk) pair with a
    cross-encoder that reads both together  ->  keep the best k.

A cross-encoder is slower than vector search but far better at judging "does this chunk
answer this question", and its score is a calibrated 0..1 probability, so it is also what
we use to decide "nothing relevant found -> refuse" (RERANK_MIN_SCORE).

RERANKER=cross-encoder (default, local, needs `uv sync --extra rerank`)
RERANKER=cohere        (hosted, needs COHERE_API_KEY)
RERANKER=none          (disable)
"""
from __future__ import annotations

import logging
import math
from typing import Protocol

log = logging.getLogger("production-api")


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score (0..1, higher = better) per text, same order."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # fail fast with a fix, not silently degrade
            raise RuntimeError(
                "RERANKER=cross-encoder needs sentence-transformers. Run `uv sync --extra rerank` "
                "(or set RERANKER=none to turn the reranker off)."
            ) from e
        self.name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        # some models return logits, some probabilities; squash only if outside 0..1
        out = []
        for x in raw:
            x = float(x)
            out.append(x if 0.0 <= x <= 1.0 else 1.0 / (1.0 + math.exp(-x)))
        return out


class CohereReranker:
    URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: str, model_name: str = "rerank-v3.5", client=None):
        import httpx

        if not api_key:
            raise RuntimeError("RERANKER=cohere needs COHERE_API_KEY")
        self.name = f"cohere:{model_name}"
        self.model_name = model_name
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=30)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        r = self.client.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model_name, "query": query, "documents": texts, "top_n": len(texts)},
        )
        r.raise_for_status()
        scores = [0.0] * len(texts)
        for item in r.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores


def build_reranker(settings) -> Reranker | None:
    kind = settings.reranker.strip().lower()
    if kind in ("", "none", "off", "false"):
        return None
    if kind == "cohere":
        return CohereReranker(settings.cohere_api_key, settings.cohere_model)
    if kind == "cross-encoder":
        log.info("Loading reranker model %s", settings.rerank_model)
        return CrossEncoderReranker(settings.rerank_model)
    raise ValueError(f"unknown RERANKER={settings.reranker!r} (use cross-encoder, cohere or none)")
