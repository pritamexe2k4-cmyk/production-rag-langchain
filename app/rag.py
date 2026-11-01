"""
# GLUE (not from course): the retrieval layer used by the production app.

lang-production-api has no retrieval at all, so this file is the bridge. It does
not invent new techniques; each step is the course's own code, re-used here:

  load      PyPDFLoader / TextLoader ........ part1/document_loaders.py
  split     RecursiveCharacterTextSplitter
            (chunk_size=500, overlap=50) .... part1/rag_pipeline.py  (create_kb)
  embed     OpenAIEmbeddings
            ("text-embedding-3-small") ...... part1/embeddings.py
  store     Chroma, persist_directory ....... part1/vector_stores.py  (persist_chroma)
  score     1 / (1 + distance) .............. part1/vector_stores.py  (similarity_search_with_scores)
  hybrid    BM25Retriever + EnsembleRetriever
            weights 0.4 / 0.6 ............... part2_debug_optimize/advanced_rag.py (demo_ensemble_hybrid_search)
  prompt    "answer ONLY from context, else
            'I don't have information...'" .. part1/rag_pipeline.py (demo_rag_with_fallback + demo_rag_with_sources)

What IS glue: putting those pieces in one class, idempotent ids so re-ingesting
the same file does not duplicate chunks, path-safety for /ingest, and
shortening `source` to a file name for cleaner citations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

REFUSAL = "I don't have information about that in my knowledge base."
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class Retrieved:
    document: Document
    score: float | None  # 1/(1+distance); None in hybrid mode (BM25 has no distance)


def format_docs_with_sources(docs: list[Document]) -> str:
    """Course helper from rag_pipeline.py::demo_rag_with_sources (page number added)."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source
        formatted.append(f"[{i+1}] {label}:\n{doc.page_content}")
    return "\n\n".join(formatted)


# Prompt = demo_rag_with_fallback + demo_rag_with_sources, merged.
RAG_PROMPT = """
Answer the question based ONLY on the following context.
If the answer is not in the context, respond with: "{refusal}"
Include which sources you used.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_dir,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self._bm25 = None
        self._bm25_count = -1

    # ---------------------------------------------------------------- ingest
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

    def _load(self, path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(path)).load()  # one Document per page
        else:
            docs = TextLoader(str(path), encoding="utf-8").load()
        for d in docs:
            d.metadata["source"] = path.name
        return docs

    def ingest(self, rel_path: str) -> dict:
        target = self.resolve_path(rel_path)
        files = (
            sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            if target.is_dir()
            else [target]
        )
        if not files:
            raise ValueError("no .pdf/.txt/.md files found")
        before = self.count()
        pages = 0
        for f in files:
            docs = self._load(f)
            pages += len(docs)
            chunks = self.splitter.split_documents(docs)
            ids = [
                hashlib.sha1(
                    f"{c.metadata['source']}|{c.metadata.get('page')}|{i}|{c.page_content}".encode()
                ).hexdigest()
                for i, c in enumerate(chunks)
            ]
            self.vectorstore.add_documents(chunks, ids=ids)  # upsert -> idempotent
        self._bm25 = None
        after = self.count()
        return {
            "files": [f.name for f in files],
            "pages": pages,
            "chunks_added": after - before,
            "total_chunks_in_store": after,
        }

    # -------------------------------------------------------------- retrieve
    def _bm25_retriever(self, k: int):
        from langchain_community.retrievers import BM25Retriever

        n = self.count()
        if self._bm25 is None or self._bm25_count != n:
            data = self.vectorstore.get()
            docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(data["documents"], data["metadatas"])
            ]
            self._bm25 = BM25Retriever.from_documents(docs)
            self._bm25_count = n
        self._bm25.k = k
        return self._bm25

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        k = k or self.settings.retrieval_k
        if self.count() == 0:
            return []
        if self.settings.hybrid_search:
            from langchain_classic.retrievers import EnsembleRetriever

            ensemble = EnsembleRetriever(
                retrievers=[
                    self._bm25_retriever(k),
                    self.vectorstore.as_retriever(search_kwargs={"k": k}),
                ],
                weights=[0.4, 0.6],  # 40% keyword, 60% semantic (course values)
            )
            return [Retrieved(d, None) for d in ensemble.invoke(query)[:k]]
        out = []
        for doc, distance in self.vectorstore.similarity_search_with_score(query, k=k):
            score = 1 / (1 + distance)  # course formula: distance -> similarity
            if score >= self.settings.min_relevance_score:
                out.append(Retrieved(doc, score))
        return out
