"""
# GLUE (not from course): the retrieval layer used by the production app.

lang-production-api has no retrieval at all, so this file is the bridge. Phase 0 re-used the
course's own pieces; phase 1 (HR policy assistant) extends them:

  load      app/loaders.py ................... multi-format (PDF/DOCX/HTML/MD) - phase 1; course: part1/document_loaders.py
  split     app/chunking.py .................. header-aware - phase 1; course: part1/text_splitters.py
  embed     OpenAIEmbeddings (text-embedding-3-small), in batches ... part1/embeddings.py
  store     Chroma, persist_directory ........ part1/vector_stores.py
  score     1 / (1 + distance) ............... part1/vector_stores.py (similarity_search_with_scores)
  hybrid    BM25 + vectors, weights 0.4/0.6 ... part2_debug_optimize/advanced_rag.py (demo_ensemble_hybrid_search)
  rerank    cross-encoder over the candidates . app/reranker.py - phase 1 addition
  prompt    "answer ONLY from context, else refuse" ... part1/rag_pipeline.py (demo_rag_with_fallback + _with_sources)

Retrieval pipeline of search():

    filters (doc_type / category) -> vector top-N  \
                                                     > weighted reciprocal-rank fusion -> rerank -> top-k -> refusal gate
                                  -> BM25 top-N     /

The fusion is the same algorithm as the course's EnsembleRetriever (weighted reciprocal rank
fusion, constant 60, weights 0.4 BM25 / 0.6 vector). It is written out here (12 lines) instead of
calling EnsembleRetriever because EnsembleRetriever throws the scores away, and we want to keep
vector score, BM25 rank, fused score and rerank score on every result to see WHY a chunk was
retrieved and to calibrate the refusal threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.chunking import chunk_document
from app.config import Settings
from app.loaders import SUPPORTED_SUFFIXES, file_sha256, load_document
from app.manifest import IngestManifest
from app.reranker import Reranker, build_reranker

REFUSAL = "I don't have information about that in my knowledge base."
RRF_C = 60  # the constant used by EnsembleRetriever

_STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it my of on or the to was what when "
    "where which who will with you your me we our there their this that these those if get".split()
)


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer: lowercase words/numbers, no punctuation, no stopwords.
    (BM25Retriever's default just splits on spaces, so 'leave?' would not match 'leave'.)"""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


@dataclass
class Retrieved:
    document: Document
    score: float | None            # the score used for gating: rerank score if reranked, else vector score
    vector_score: float | None = None   # 1/(1+distance)
    bm25_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


@dataclass
class SearchResult:
    results: list[Retrieved]            # what is passed to the LLM (after the refusal gate)
    ranked: list[Retrieved]             # top-k before the refusal gate (used by the eval for recall@k)
    top_score: float | None             # best gating score (for threshold calibration)
    refused: bool = False
    reason: str = ""


def format_docs_with_sources(docs: list[Document]) -> str:
    """Course helper from rag_pipeline.py::demo_rag_with_sources, extended with title, section, page, date."""
    formatted = []
    for i, doc in enumerate(docs):
        m = doc.metadata
        where = " > ".join(x for x in (m.get("doc_title") or m.get("source", "unknown"), m.get("section")) if x)
        extra = ", ".join(x for x in (f"page {m['page']}" if m.get("page") else "", f"effective {m['effective_date']}" if m.get("effective_date") else "") if x)
        formatted.append(f"[{i+1}] {where}" + (f" ({extra})" if extra else "") + f":\n{doc.page_content}")
    return "\n\n".join(formatted)


# Prompt = demo_rag_with_fallback + demo_rag_with_sources, extended for policy documents.
RAG_PROMPT = """
You are an HR policy assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, respond with exactly: "{refusal}"

Rules:
- Quote numbers, dates and amounts exactly as written in the context.
- If the context answers only part of the question, answer that part and say what is missing.
- Cite the sources you used as [1], [2] after the statements they support.
- If two sources disagree, prefer the one with the later effective date and say so.

Context:
{context}

Question: {question}

Answer:"""


def _where(doc_types: list[str] | None, categories: list[str] | None) -> dict | None:
    conds = []
    if doc_types:
        conds.append({"doc_type": {"$in": [d.lower() for d in doc_types]}})
    if categories:
        conds.append({"category": {"$in": [c.lower() for c in categories]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class RAGService:
    _UNSET = object()

    def __init__(self, settings: Settings, embeddings=None, reranker=_UNSET):
        self.settings = settings
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_dir,
        )
        self.reranker: Reranker | None = build_reranker(settings) if reranker is RAGService._UNSET else reranker
        self.manifest = IngestManifest(
            Path(settings.chroma_dir) / f"ingest_manifest__{settings.collection_name}.json"
        )
        if self.count() == 0 and self.manifest.files:
            self.manifest.reset()  # vector DB was wiped but the manifest survived -> start clean
        self._bm25_cache: dict = {}

    # ------------------------------------------------------------------ helpers
    def count(self) -> int:
        return self.vectorstore._collection.count()

    def resolve_path(self, rel: str) -> Path:
        """Only allow paths below DATA_DIR (blocks ../../etc/passwd style input)."""
        base = Path(self.settings.data_dir).resolve()
        target = (base / rel).resolve()
        if target != base and base not in target.parents:
            raise ValueError("path must be inside the data directory")
        if not target.exists():
            raise FileNotFoundError(rel)
        return target

    @staticmethod
    def _scan(target: Path) -> list[Path]:
        if target.is_file():
            return [target]
        return sorted(
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(("_", "."))
        )

    def _add_in_batches(self, chunks: list[Document]) -> None:
        n = max(1, self.settings.embed_batch_size)
        for i in range(0, len(chunks), n):
            batch = chunks[i : i + n]
            self.vectorstore.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])

    def _delete_ids(self, ids) -> None:
        ids = list(ids)
        for i in range(0, len(ids), 500):
            self.vectorstore.delete(ids=ids[i : i + 500])

    # ------------------------------------------------------------------- ingest
    def ingest(self, rel_path: str = "hr_docs", force: bool = False, prune: bool = True) -> dict:
        """Incremental ingest of a file or folder below DATA_DIR. See app/manifest.py for the rules."""
        base = Path(self.settings.data_dir).resolve()
        target = self.resolve_path(rel_path)
        files = self._scan(target)
        if not files:
            raise ValueError("no supported files found (.pdf .docx .html .md .txt)")
        if self.count() == 0 and self.manifest.files:
            self.manifest.reset()

        rep = {"files": [], "added": [], "updated": [], "unchanged": [], "duplicates": {}, "deleted": [], "errors": {},
               "pages": 0, "chunks_added": 0, "chunks_deleted": 0}
        on_disk = {p.relative_to(base).as_posix() for p in files}

        if prune and target.is_dir():
            prefix = "" if target == base else target.relative_to(base).as_posix() + "/"
            for key in [k for k in self.manifest.files if k.startswith(prefix) and k not in on_disk
                        and not (base / k).exists()]:
                ids = self.manifest.files.pop(key).get("chunk_ids", [])
                if ids:
                    self._delete_ids(ids)
                rep["chunks_deleted"] += len(ids)
                rep["deleted"].append(Path(key).name)
            self.manifest.save()

        owners = self.manifest.hash_owners()  # after pruning, so a copy can take over from a deleted original
        for path in files:
            key = path.relative_to(base).as_posix()
            rep["files"].append(path.name)
            try:
                sha = file_sha256(path)
                entry = self.manifest.files.get(key)
                if entry and entry["sha256"] == sha and not force:
                    owner = entry.get("duplicate_of")
                    orig = self.manifest.files.get(owner, {}) if owner else {}
                    if not owner or (orig.get("chunk_ids") and orig.get("sha256") == sha):  # a duplicate stays valid while its original is indexed unchanged
                        rep["unchanged"].append(path.name)
                        continue
                owner = owners.get(sha)
                if owner and owner != key and not force:
                    if entry and entry.get("chunk_ids"):  # file became a copy of another one
                        self._delete_ids(entry["chunk_ids"])
                        rep["chunks_deleted"] += len(entry["chunk_ids"])
                    self.manifest.files[key] = {"sha256": sha, "duplicate_of": owner, "chunk_ids": []}
                    self.manifest.save()
                    rep["duplicates"][path.name] = owner
                    continue

                doc = load_document(path)
                chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
                new_ids = [c.metadata["chunk_id"] for c in chunks]
                old_ids = set(entry["chunk_ids"]) if entry else set()
                self._add_in_batches([c for c in chunks if c.metadata["chunk_id"] not in old_ids])  # embed only what is new
                stale = old_ids - set(new_ids)
                if stale:  # delete AFTER adding, so the document is never missing from the index
                    self._delete_ids(stale)
                self.manifest.files[key] = {
                    "sha256": sha, "title": doc.title, "doc_type": doc.doc_type, "category": doc.category,
                    "effective_date": doc.effective_date, "version": doc.version, "fmt": doc.fmt,
                    "pages": doc.pages, "chunk_ids": new_ids,
                }
                self.manifest.save()
                owners[sha] = key
                rep["pages"] += doc.pages
                rep["chunks_added"] += len(set(new_ids) - old_ids)
                rep["chunks_deleted"] += len(stale)
                rep["updated" if entry and entry.get("chunk_ids") else "added"].append(path.name)
            except Exception as e:  # one bad file must not stop the batch
                rep["errors"][path.name] = f"{type(e).__name__}: {e}"

        self._bm25_cache.clear()
        rep["total_chunks_in_store"] = self.count()
        return rep

    def list_documents(self) -> list[dict]:
        out = []
        for key, e in sorted(self.manifest.files.items()):
            out.append({"path": key, "title": e.get("title", ""), "doc_type": e.get("doc_type", ""),
                        "category": e.get("category", ""), "effective_date": e.get("effective_date", ""),
                        "version": e.get("version", ""), "pages": e.get("pages", 0),
                        "chunks": len(e.get("chunk_ids", [])), "duplicate_of": e.get("duplicate_of")})
        return out

    # ----------------------------------------------------------------- retrieval
    def _bm25(self, where: dict | None):
        from langchain_community.retrievers import BM25Retriever

        key = (repr(where), self.count())
        if key not in self._bm25_cache:
            data = self.vectorstore.get(where=where, include=["documents", "metadatas"])
            docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(data["documents"], data["metadatas"])]
            self._bm25_cache[key] = BM25Retriever.from_documents(docs, preprocess_func=tokenize) if docs else None
        return self._bm25_cache[key]

    @staticmethod
    def _key(doc: Document) -> str:
        return doc.metadata.get("chunk_id") or doc.page_content

    def search(self, query: str, k: int | None = None, doc_types: list[str] | None = None,
               categories: list[str] | None = None) -> SearchResult:
        s = self.settings
        k = k or s.retrieval_k
        if self.count() == 0:
            return SearchResult([], [], None, True, "empty index")
        where = _where(doc_types, categories)
        n = max(s.candidates, k)

        # 1. vector search (scores kept: course formula 1/(1+distance))
        kwargs = {"filter": where} if where else {}
        vec = [(d, 1 / (1 + dist)) for d, dist in self.vectorstore.similarity_search_with_score(query, k=n, **kwargs)]
        cands: dict[str, Retrieved] = {}
        for rank, (d, sc) in enumerate(vec, start=1):
            cands[self._key(d)] = Retrieved(d, sc, vector_score=sc, fused_score=s.hybrid_weights_vector / (RRF_C + rank))

        # 2. BM25 + weighted reciprocal-rank fusion (hybrid)
        if s.hybrid_search:
            bm25 = self._bm25(where)
            if bm25 is not None:
                bm25.k = n
                for rank, d in enumerate(bm25.invoke(query), start=1):
                    r = cands.setdefault(self._key(d), Retrieved(d, None, fused_score=0.0))
                    r.bm25_rank = rank
                    r.fused_score = (r.fused_score or 0.0) + s.hybrid_weights_bm25 / (RRF_C + rank)
        ordered = sorted(cands.values(), key=lambda r: r.fused_score or 0.0, reverse=True)

        # 3. rerank the best candidates with a cross-encoder
        if self.reranker is not None and ordered:
            pool = ordered[:n]
            for r, sc in zip(pool, self.reranker.score(query, [r.document.page_content for r in pool])):
                r.rerank_score = sc
                r.score = sc
            ordered = sorted(pool, key=lambda r: r.rerank_score, reverse=True)
        ranked = ordered[:k]

        # 4. refusal gate
        best_vec = max((r.vector_score for r in cands.values() if r.vector_score is not None), default=None)
        if self.reranker is not None:
            top = ranked[0].rerank_score if ranked else None
            results = [r for r in ranked if r.rerank_score >= s.rerank_min_score]
            reason = f"best rerank score {top:.3f} < RERANK_MIN_SCORE {s.rerank_min_score}" if not results and top is not None else ""
        else:
            top = best_vec
            results, reason = ranked, ""
            if s.min_relevance_score > 0 and (best_vec is None or best_vec < s.min_relevance_score):
                results, reason = [], f"best vector score {best_vec} < MIN_RELEVANCE_SCORE {s.min_relevance_score}"
        return SearchResult(results, ranked, top, refused=not results, reason=reason)

    def retrieve(self, query: str, k: int | None = None, **filters) -> list[Retrieved]:
        """Convenience wrapper: just the chunks that pass the refusal gate."""
        return self.search(query, k, **filters).results
