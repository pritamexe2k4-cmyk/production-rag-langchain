"""
SOURCE: lang-production-api  app/main.py. Original code is kept; every addition is
marked `# GLUE (not from course)`: the RAG service, /ingest, /query.

Production-Ready FastAPI + LangGraph Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""

import json  # GLUE (not from course)
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent
# GLUE (not from course): retrieval layer + its request/response models
from app.rag import RAGService
from app.rag_models import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceItem,
)

load_dotenv()



# === Global instances (initialized in lifespan) ===
security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
rag: RAGService = None  # GLUE (not from course)
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    global security, cache, metrics, agent, rag

    settings = get_settings()

    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    rag = RAGService(settings)  # GLUE (not from course)
    agent = ProductionAgent(rag=rag)

    logger.info("All components initialized. Ready to serve requests.")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})
    
    
    # === Rate Limiter Setup ===
limiter = Limiter(key_func=get_remote_address)

# === FastAPI App ===
app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


# === Exception Handlers ===

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )
    

# =============================================
# ENDPOINTS
# =============================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # ---- Step 3: Invoke LangGraph Agent ----
        try:
            result = agent.invoke(cleaned_message, use_retrieval=False)  # GLUE: /chat stays plain chat (no retrieval)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
        }})

    logger.info("Request completed", extra={"extra_data": {
        "thread_id": body.thread_id,
        "model_used": model_used,
        "latency_ms": round(timer.elapsed_ms, 2),
    }})

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )
    
    
    
    
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
        "vector_store": rag is not None,  # GLUE (not from course)
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards."""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats


# =============================================
# GLUE (not from course): RAG ENDPOINTS
# =============================================

def _check_ingest_key(x_api_key: str | None):
    """If INGEST_API_KEY is set, /ingest needs `X-API-Key: <that value>`."""
    expected = get_settings().ingest_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(get_settings().rate_limit)
async def ingest(
    request: Request,
    body: IngestRequest,
    x_api_key: str | None = Header(default=None),
):
    """
    Load -> split -> embed -> store files from DATA_DIR into Chroma.
    Re-ingesting the same file does not create duplicates.
    """
    _check_ingest_key(x_api_key)
    try:
        stats = rag.ingest(body.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File or folder not found in data directory")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # e.g. bad OPENAI_API_KEY while embedding
        logger.error(f"Ingest failed: {e}", extra={"extra_data": {"error": str(e)}})
        raise HTTPException(status_code=502, detail="Ingest failed (check OPENAI_API_KEY and server logs).")
    cache.clear()  # answers cached before this ingest may now be stale
    logger.info("Ingest completed", extra={"extra_data": stats})
    return IngestResponse(**stats)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="query_endpoint")
async def query(request: Request, body: QueryRequest):
    """
    RAG endpoint. Same production flow as /chat, but the agent retrieves first:

    1. Security check (injection + PII masking)
    2. Cache lookup
    3. Agent: retrieve (-> grade/rewrite) -> answer from context only
    4. Output validation
    5. Cache store
    6. Return answer + sources
    """
    with RequestTimer() as timer:
        security_notes = []

        is_allowed, cleaned, notes = security.check_input(body.question)
        security_notes.extend(notes)
        if not is_allowed:
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        cache_key = "rag::" + cleaned  # keep separate from /chat answers
        cached = cache.get(cache_key)
        if cached is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            payload = json.loads(cached)
            return QueryResponse(
                answer=payload["answer"],
                sources=[SourceItem(**s) for s in payload["sources"]],
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        try:
            result = agent.invoke(cleaned, use_retrieval=True)
        except Exception as e:
            logger.error(f"RAG invocation failed: {e}", extra={"extra_data": {"error": str(e)}})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        answer, output_warnings = security.check_output(result["response"])
        security_notes.extend(output_warnings)
        # only cache real answers, not the error-handler apology
        if result["model_used"] != "error_handler":
            cache.set(cache_key, json.dumps({"answer": answer, "sources": result["sources"]}))

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=int(len(cleaned.split()) * 1.3),
        output_tokens=int(len(answer.split()) * 1.3),
        cache_hit=False,
    )
    return QueryResponse(
        answer=answer,
        sources=[SourceItem(**s) for s in result["sources"]],
        thread_id=body.thread_id,
        model_used=result["model_used"],
        cached=False,
        rewritten_query=result["rewritten_query"],
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )
