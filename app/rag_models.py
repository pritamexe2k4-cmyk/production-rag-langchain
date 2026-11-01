"""
# GLUE (not from course): request/response models for the new /ingest and /query endpoints.
# Same style as lang-production-api's app/models.py (which is left untouched).
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(
        default="langchain_demo.pdf",
        description="File or folder, relative to DATA_DIR (default ./data/docs)",
    )


class IngestResponse(BaseModel):
    files: list[str]
    pages: int
    chunks_added: int
    total_chunks_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    thread_id: str = Field(default="default")


class SourceItem(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    thread_id: str
    model_used: str
    cached: bool = False
    rewritten_query: str | None = None
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
