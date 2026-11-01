"""
# GLUE (not from course): request/response models for the new /ingest and /query endpoints.
# Same style as lang-production-api's app/models.py (which is left untouched).
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(
        default="hr_docs",
        description="File or folder, relative to DATA_DIR (default ./data). Folders are scanned recursively.",
    )
    force: bool = Field(default=False, description="Re-index even if the file hash is unchanged")
    prune: bool = Field(default=True, description="When ingesting a folder, delete chunks of files that no longer exist")


class IngestResponse(BaseModel):
    files: list[str]
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    duplicates: dict[str, str] = Field(default_factory=dict, description="file -> original it duplicates")
    errors: dict[str, str] = Field(default_factory=dict)
    pages: int
    chunks_added: int
    chunks_deleted: int = 0
    total_chunks_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    thread_id: str = Field(default="default")
    doc_types: list[str] | None = Field(default=None, description="Only search these document types, e.g. ['policy']")
    categories: list[str] | None = Field(default=None, description="Only search these categories, e.g. ['leave', 'wfh']")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Chunks to give the model (default RETRIEVAL_K)")


class SourceItem(BaseModel):
    n: int | None = None
    source: str
    doc_title: str = ""
    section: str = ""
    doc_type: str = ""
    effective_date: str = ""
    page: int | None = None
    score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    cited: bool = False
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    thread_id: str
    model_used: str
    cached: bool = False
    rewritten_query: str | None = None
    refused: bool = False
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
