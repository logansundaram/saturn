"""Workspace-manifest reconciliation (2026-09-02).

The workspace manifest — the "Workspace files" block the planner reads every turn — was only
ever written by write_file / edit_file / /undo. Anything that happened to the workspace
OUTSIDE the agent (a file deleted in Finder, a CSV dropped in) left it wrong in one of two
directions: phantom entries the planner then planned read_file on (measured: a turn spent
2.5 minutes reading four files that no longer existed), or real files it never knew about.
`sync_workspace_manifest` reconciles both directions against disk; ground calls it per turn.
"""

import pytest

from stores import document_registry as dr


@pytest.fixture
def manifest_env(isolated_paths, monkeypatch):
    """Isolated workspace + a stubbed summarizer (no LLM)."""
    monkeypatch.setattr(dr, "_summarize", lambda content, filename: "stub summary")
    ws = isolated_paths / "database" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _names() -> list[str]:
    return [e["name"] for e in dr.manifest_entries(dr.read_workspace_manifest())]


def test_phantom_entries_are_dropped_and_present_ones_kept(manifest_env):
    (manifest_env / "keep.txt").write_text("kept", encoding="utf-8")
    dr.register_workspace_file("keep.txt", "kept")
    dr.register_workspace_file("gone.txt", "was here")
    removed, added = dr.sync_workspace_manifest()
    assert removed == ["gone.txt"] and added == []
    assert _names() == ["keep.txt"]


def test_unregistered_files_on_disk_are_added_with_a_first_line_summary(manifest_env):
    (manifest_env / "sub").mkdir()
    (manifest_env / "sub" / "notes.md").write_text("# Meeting notes\n\nbody", encoding="utf-8")
    (manifest_env / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    removed, added = dr.sync_workspace_manifest()
    assert removed == [] and sorted(added) == ["data.csv", "sub/notes.md"]
    assert sorted(_names()) == ["data.csv", "sub/notes.md"]
    entries = {e["name"]: e for e in dr.manifest_entries(dr.read_workspace_manifest())}
    assert entries["sub/notes.md"]["summary"] == "stub summary"


def test_hidden_files_the_manifest_itself_and_binaries_are_skipped(manifest_env):
    (manifest_env / ".hidden").write_text("secret", encoding="utf-8")
    (manifest_env / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    (manifest_env / ".git").mkdir()
    (manifest_env / ".git" / "config").write_text("x", encoding="utf-8")
    (manifest_env / "ok.txt").write_text("fine", encoding="utf-8")
    removed, added = dr.sync_workspace_manifest()
    assert added == ["ok.txt"]
    assert _names() == ["ok.txt"]


def test_sync_is_a_no_op_when_nothing_changed(manifest_env):
    (manifest_env / "a.txt").write_text("a", encoding="utf-8")
    dr.register_workspace_file("a.txt", "a")
    before = dr.read_workspace_manifest()
    assert dr.sync_workspace_manifest() == ([], [])
    assert dr.read_workspace_manifest() == before


def test_missing_workspace_directory_is_not_an_error(isolated_paths):
    assert dr.sync_workspace_manifest() == ([], [])


def test_ground_reconciles_the_workspace_before_building_the_block(monkeypatch, manifest_env):
    from nodes import ground

    calls = []
    monkeypatch.setattr(ground, "sync_workspace_manifest",
                        lambda: calls.append(1) or ([], []))
    ground.grounding_node({"messages": [], "current_query": "q"})
    assert calls == [1]
