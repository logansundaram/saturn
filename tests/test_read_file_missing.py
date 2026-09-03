"""read_file's not-found refusal (2026-09-02).

A missing workspace file used to surface as the raw OSError — `Error calling read_file:
[Errno 2] No such file or directory: 'welcome-to-saturn.md'` — which told the redraft nothing
about WHERE the file is. The common shape (run 24): the planner asked read_file for a document
that lives in the KNOWLEDGE BASE, not the workspace; two judge calls and a replan later it
reached search_knowledge_base. The refusal now names the namespace, and when the name matches
an ingested document, names the tool that reads it. Still an error (the step did fail — the
incident is disclosed); only the text changed. Offline: isolated paths, no embedder.
"""

import pytest

from stores.document_registry import register_rag_document
from tools.files import read_file


def test_missing_file_names_the_workspace(isolated_paths):
    with pytest.raises(FileNotFoundError) as exc:
        read_file.invoke({"file_path": "nothing-here.txt"})
    text = str(exc.value)
    assert text.startswith("File not found in the workspace: nothing-here.txt")
    assert "search_knowledge_base" not in text


def test_missing_file_that_is_a_knowledge_base_document_names_the_tool(isolated_paths):
    register_rag_document("welcome-to-saturn.md", "Welcome to Saturn\n\nSaturn is an agent.")
    with pytest.raises(FileNotFoundError) as exc:
        read_file.invoke({"file_path": "welcome-to-saturn.md"})
    text = str(exc.value)
    assert text.startswith("File not found in the workspace: welcome-to-saturn.md")
    assert "knowledge base" in text and "search_knowledge_base" in text


def test_knowledge_base_match_is_by_basename_case_insensitively(isolated_paths):
    register_rag_document("Notes.PDF", "some notes")
    with pytest.raises(FileNotFoundError) as exc:
        read_file.invoke({"file_path": "docs/notes.pdf"})
    assert "search_knowledge_base" in str(exc.value)


def test_present_file_still_reads(isolated_paths):
    ws = isolated_paths / "database" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    assert read_file.invoke({"file_path": "a.txt"}) == "hello"
