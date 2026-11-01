"""
# GLUE (not from course): command-line ingest so you can fill the vector DB
# without starting the API.

    uv run --env-file .env python -m app.ingest                 # data/docs/langchain_demo.pdf
    uv run --env-file .env python -m app.ingest some_folder     # relative to data/docs
"""
import json
import sys

from app.config import get_settings
from app.rag import RAGService

if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else "langchain_demo.pdf"
    print(json.dumps(RAGService(get_settings()).ingest(rel), indent=2))
