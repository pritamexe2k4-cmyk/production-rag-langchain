"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.

SOURCE: lang-production-api  app/config.py  (course "Production project" chapter)
The original fields are untouched. Every field under the GLUE marker below was
added by us to configure the retrieval layer.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

class Settings(BaseSettings):
    
    # LLM Configuration
    openai_api_key: str
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"
    
    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production-api"
    
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    # ------------------------------------------------------------------
    # GLUE (not from course): settings for the retrieval layer. The values are
    # the ones the course uses in its scripts (text-embedding-3-small from
    # embeddings.py, chunk 500/50 from rag_pipeline.py, hybrid weights 0.4/0.6
    # from advanced_rag.py, max_retries=2 + 0.5 relevance cutoff from
    # 04_agentic_rag.py).
    # ------------------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "./chroma_db"
    collection_name: str = "production_rag"
    data_dir: str = "./data"               # /ingest may only read files below this folder
    chunk_size: int = 1000                 # phase 1: 1000/100 suits policy sections (course demo used 500/50)
    chunk_overlap: int = 100
    embed_batch_size: int = 64             # phase 1: chunks embedded per OpenAI call
    retrieval_k: int = 5                   # chunks given to the LLM
    hybrid_search: bool = True             # Part 2 hybrid: BM25 + vectors (advanced_rag.py); phase 1: ON by default
    hybrid_weights_bm25: float = 0.4       # course values 0.4 / 0.6
    hybrid_weights_vector: float = 0.6
    candidates: int = 20                   # phase 1: candidates fetched before reranking
    agentic_retrieval: bool = False        # Part 6 agentic RAG: grade + rewrite loop (04_agentic_rag.py)
    max_rewrites: int = 2
    min_relevance_score: float = 0.0       # 0 = off. Refuse if the best vector hit is below this; score = 1/(1+distance) as in vector_stores.py
    # phase 1: reranker (see app/reranker.py)
    reranker: str = "cross-encoder"        # cross-encoder | cohere | none
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # e.g. BAAI/bge-reranker-base is slower but stronger
    rerank_min_score: float = 0.002        # refuse when the best reranker score is below this. Calibrated on the HR eval set: lowest answerable top score 0.004, so 0.002 keeps every answerable question; see README "Calibration"
    cohere_api_key: str = ""
    cohere_model: str = "rerank-v3.5"
    ingest_api_key: str = ""               # if set, /ingest requires header X-API-Key
    
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()
