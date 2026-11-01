"""
# GLUE (not from course): command-line ingest so you can fill / refresh the vector DB without
# starting the API. Incremental: unchanged files are skipped, changed files re-indexed,
# deleted files removed (see app/manifest.py).

    uv run --env-file .env python -m app.ingest                      # data/hr_docs (default)
    uv run --env-file .env python -m app.ingest docs/langchain_demo.pdf   # one file below data/
    uv run --env-file .env python -m app.ingest --force              # re-index everything
    uv run --env-file .env python -m app.ingest --list               # show what is indexed
"""
import argparse
import json

from app.config import get_settings
from app.rag import RAGService


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="hr_docs", help="file or folder below DATA_DIR (default: hr_docs)")
    ap.add_argument("--force", action="store_true", help="re-index even if the file hash is unchanged")
    ap.add_argument("--no-prune", action="store_true", help="do not delete chunks of files that disappeared")
    ap.add_argument("--list", action="store_true", help="list indexed documents and exit")
    args = ap.parse_args(argv)

    s = get_settings()
    s.reranker = "none"  # ingest never needs the reranker model
    rag = RAGService(s)
    if args.list:
        for d in rag.list_documents():
            print(f"{d['path']:60s} {d['doc_type']:10s} {d['category']:12s} {d['effective_date']:11s} v{d['version']:5s} chunks={d['chunks']}"
                  + (f"  (duplicate of {d['duplicate_of']})" if d["duplicate_of"] else ""))
        print(f"total chunks: {rag.count()}")
        return
    print(json.dumps(rag.ingest(args.path, force=args.force, prune=not args.no_prune), indent=2))


if __name__ == "__main__":
    main()
