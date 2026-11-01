"""
# GLUE (not from course) - PHASE 1 ADDITION: the ingest manifest.

A small JSON file that remembers, for every ingested file, its SHA-256 hash and the ids of the
chunks it produced. That is what makes re-indexing incremental:

  same hash            -> skip (unchanged)
  different hash       -> re-parse, embed only chunks that are new, delete chunks that are gone
  file no longer there -> delete its chunks
  same hash as another file -> skip, report as duplicate

It lives inside the Chroma folder (one manifest per collection), so deleting `chroma_db/` also
deletes the manifest and the two can never disagree.
"""
from __future__ import annotations

import json
from pathlib import Path


class IngestManifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.files: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.files = json.loads(self.path.read_text(encoding="utf-8")).get("files", {})
            except (json.JSONDecodeError, OSError):
                self.files = {}  # corrupt manifest -> behave like a first ingest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "files": self.files}, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def reset(self) -> None:
        self.files = {}
        self.save()

    def hash_owners(self) -> dict[str, str]:
        """sha256 -> path of the file that actually owns chunks for that content."""
        return {e["sha256"]: k for k, e in self.files.items() if e.get("chunk_ids")}
