"""stores/snapshots.py — the /undo layer: pre-write snapshot, restore, created-file deletion,
and the nothing-to-undo error. The document-registry sync inside undo_last is stubbed out so no
test ever reaches for the LLM summarizer."""

import pytest

from stores import snapshots


@pytest.fixture
def workspace(isolated_paths, monkeypatch):
    """An isolated workspace + a neutered manifest sync (undo_last imports these lazily)."""
    import stores.document_registry as dr

    monkeypatch.setattr(dr, "register_workspace_file", lambda *a, **k: None)
    monkeypatch.setattr(dr, "remove_workspace_file", lambda *a, **k: None)
    ws = isolated_paths / "database" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_undo_restores_overwritten_file(workspace):
    target = workspace / "a.txt"
    target.write_text("original", encoding="utf-8")
    snapshots.begin_turn("overwrite a.txt")
    snapshots.snapshot_file("a.txt", target)
    target.write_text("mutated", encoding="utf-8")

    summary, actions = snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "original"
    assert any("restored a.txt" in a for a in actions)
    assert "overwrite a.txt" in summary


def test_undo_deletes_created_file(workspace):
    target = workspace / "new.txt"
    snapshots.begin_turn("create new.txt")
    snapshots.snapshot_file("new.txt", target)  # recorded as not-existing
    target.write_text("created", encoding="utf-8")

    _, actions = snapshots.undo_last()
    assert not target.exists()
    assert any("deleted new.txt" in a for a in actions)


def test_first_snapshot_wins(workspace):
    """Two writes to the same file in one turn: undo restores the TURN-START bytes."""
    target = workspace / "a.txt"
    target.write_text("turn-start", encoding="utf-8")
    snapshots.begin_turn("double write")
    snapshots.snapshot_file("a.txt", target)
    target.write_text("first write", encoding="utf-8")
    snapshots.snapshot_file("a.txt", target)  # no-op: turn-start already captured
    target.write_text("second write", encoding="utf-8")

    snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "turn-start"


def test_each_undo_pops_one_batch(workspace):
    target = workspace / "a.txt"
    target.write_text("v1", encoding="utf-8")
    snapshots.begin_turn("turn one")
    snapshots.snapshot_file("a.txt", target)
    target.write_text("v2", encoding="utf-8")
    snapshots.begin_turn("turn two")
    snapshots.snapshot_file("a.txt", target)
    target.write_text("v3", encoding="utf-8")

    assert len(snapshots.list_batches()) == 2
    snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "v2"
    snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "v1"
    with pytest.raises(RuntimeError):
        snapshots.undo_last()


def test_unarmed_snapshot_noops(workspace, monkeypatch):
    """A tool firing outside a turn (no begin_turn) must not create batches."""
    monkeypatch.setattr(snapshots, "_pending", None)
    monkeypatch.setattr(snapshots, "_active_dir", None)
    target = workspace / "a.txt"
    target.write_text("x", encoding="utf-8")
    snapshots.snapshot_file("a.txt", target)
    assert snapshots.list_batches() == []
