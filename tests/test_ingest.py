"""
# GLUE (not from course): tests for the incremental ingestion pipeline (app/rag.py::ingest + app/manifest.py).
Uses a temporary data folder so files can be changed, deleted and duplicated.
"""
import shutil

import pytest

from app.config import get_settings
from app.rag import RAGService
from tests.conftest import EMBED_CALLS, ROOT

HR = ROOT / "data" / "hr_docs"


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = tmp_path / "data"
    (d / "hr").mkdir(parents=True)
    for name in ("Leave_Policy.pdf", "HR_FAQ.md", "Holiday_List_2027.html", "POSH_Policy.docx"):
        shutil.copy(HR / name, d / "hr" / name)
    monkeypatch.setattr(get_settings(), "data_dir", str(d))
    return d


def make():
    return RAGService(get_settings())


def test_full_hr_corpus_ingests_without_errors(monkeypatch):
    rag = make()
    rep = rag.ingest("hr_docs")
    assert len(rep["added"]) == 14 and rep["errors"] == {}
    assert "_ABOUT_THIS_CORPUS.md" not in rep["files"]            # underscore files are skipped
    assert rep["chunks_added"] == rep["total_chunks_in_store"] > 100
    types = {d["doc_type"] for d in rag.list_documents()}
    assert {"policy", "guideline", "handbook", "calendar", "faq"} <= types
    # content-hash incremental: a second run touches nothing and embeds nothing
    EMBED_CALLS.clear()
    again = rag.ingest("hr_docs")
    assert len(again["unchanged"]) == 14 and again["chunks_added"] == 0 and EMBED_CALLS == []


def test_changed_file_only_new_chunks_are_embedded_and_stale_ones_deleted(data):
    rag = make()
    rag.ingest("hr")
    before = rag.count()
    f = data / "hr" / "HR_FAQ.md"
    f.write_text(f.read_text(encoding="utf-8").replace("8 days of Casual Leave and 8 days of Sick Leave", "9 days of Casual Leave and 8 days of Sick Leave"), encoding="utf-8")
    EMBED_CALLS.clear()
    rep = rag.ingest("hr")
    assert rep["updated"] == ["HR_FAQ.md"] and len(rep["unchanged"]) == 3
    assert rep["chunks_added"] == 1 and rep["chunks_deleted"] == 1 and sum(EMBED_CALLS) == 1  # one chunk changed
    assert rag.count() == before
    texts = rag.vectorstore.get()["documents"]
    assert any("9 days of Casual Leave" in t for t in texts) and not any("8 days of Casual Leave and 8 days" in t for t in texts)


def test_deleted_file_is_pruned_from_index_and_manifest(data):
    rag = make()
    rag.ingest("hr")
    n = len(rag.manifest.files["hr/POSH_Policy.docx"]["chunk_ids"])
    before = rag.count()
    (data / "hr" / "POSH_Policy.docx").unlink()
    rep = rag.ingest("hr")
    assert rep["deleted"] == ["POSH_Policy.docx"] and rep["chunks_deleted"] == n
    assert rag.count() == before - n and "hr/POSH_Policy.docx" not in rag.manifest.files
    assert not any(m["source"] == "POSH_Policy.docx" for m in rag.vectorstore.get()["metadatas"])


def test_prune_can_be_switched_off(data):
    rag = make()
    rag.ingest("hr")
    (data / "hr" / "HR_FAQ.md").unlink()
    assert rag.ingest("hr", prune=False)["deleted"] == []


def test_identical_copy_of_a_file_is_reported_as_duplicate_and_not_indexed(data):
    rag = make()
    rag.ingest("hr")
    count = rag.count()
    shutil.copy(data / "hr" / "HR_FAQ.md", data / "hr" / "HR_FAQ_copy.md")
    rep = rag.ingest("hr")
    assert rep["duplicates"] == {"HR_FAQ_copy.md": "hr/HR_FAQ.md"} and rep["chunks_added"] == 0
    assert rag.count() == count
    assert rag.ingest("hr")["unchanged"].count("HR_FAQ_copy.md") == 1   # stays quiet on the next run


def test_copy_takes_over_when_the_original_is_deleted(data):
    rag = make()
    shutil.copy(data / "hr" / "HR_FAQ.md", data / "hr" / "HR_FAQ_copy.md")
    rag.ingest("hr")
    (data / "hr" / "HR_FAQ.md").unlink()
    rep = rag.ingest("hr")
    assert rep["deleted"] == ["HR_FAQ.md"] and rep["added"] == ["HR_FAQ_copy.md"]
    assert any(m["source"] == "HR_FAQ_copy.md" for m in rag.vectorstore.get()["metadatas"])


def test_force_reindexes_everything(data):
    rag = make()
    rag.ingest("hr")
    rep = rag.ingest("hr", force=True)
    assert len(rep["updated"]) == 4 and rep["unchanged"] == [] and rep["chunks_added"] == 0


def test_one_bad_file_does_not_stop_the_batch(data):
    (data / "hr" / "broken.pdf").write_bytes(b"not a pdf at all")
    (data / "hr" / "empty.md").write_text("  ")
    rep = make().ingest("hr")
    assert set(rep["errors"]) == {"broken.pdf", "empty.md"} and len(rep["added"]) == 4


def test_manifest_survives_restart_and_resets_if_vector_db_was_wiped(data):
    make().ingest("hr")
    rag2 = make()                                  # new process, same chroma folder
    assert len(rag2.ingest("hr")["unchanged"]) == 4
    rag2.vectorstore.delete_collection()
    rag3 = make()                                  # collection gone, manifest file still there
    rep = rag3.ingest("hr")
    assert len(rep["added"]) == 4 and rag3.count() > 0


def test_empty_folder_is_an_error(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", str(tmp_path))
    with pytest.raises(ValueError):
        make().ingest("empty")
