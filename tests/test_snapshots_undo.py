"""stores/snapshots.py — the /undo layer: pre-write snapshot, restore, created-file deletion,
the nothing-to-undo error, and failed-restore batch retention (a failed undo must never destroy
the only copy of the turn-start bytes). The document-registry sync inside undo_last is stubbed
out so no test ever reaches for the LLM summarizer."""

import shutil

import pytest

from stores import snapshots


@pytest.fixture
def workspace(isolated_paths, monkeypatch):
    """An isolated workspace."""
    ws = isolated_paths / "database" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_undo_restores_overwritten_file(workspace):
    target = workspace / "a.txt"
    target.write_text("original", encoding="utf-8")
    snapshots.begin_turn("overwrite a.txt")
    snapshots.snapshot_file(target)
    target.write_text("mutated", encoding="utf-8")

    summary, actions = snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "original"
    assert any("restored a.txt" in a for a in actions)
    assert "overwrite a.txt" in summary


def test_undo_deletes_created_file(workspace):
    target = workspace / "new.txt"
    snapshots.begin_turn("create new.txt")
    snapshots.snapshot_file(target)  # recorded as not-existing
    target.write_text("created", encoding="utf-8")

    _, actions = snapshots.undo_last()
    assert not target.exists()
    assert any("deleted new.txt" in a for a in actions)


def test_first_snapshot_wins(workspace):
    """Two writes to the same file in one turn: undo restores the TURN-START bytes."""
    target = workspace / "a.txt"
    target.write_text("turn-start", encoding="utf-8")
    snapshots.begin_turn("double write")
    snapshots.snapshot_file(target)
    target.write_text("first write", encoding="utf-8")
    snapshots.snapshot_file(target)  # no-op: turn-start already captured
    target.write_text("second write", encoding="utf-8")

    snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "turn-start"


def test_each_undo_pops_one_batch(workspace):
    target = workspace / "a.txt"
    target.write_text("v1", encoding="utf-8")
    snapshots.begin_turn("turn one")
    snapshots.snapshot_file(target)
    target.write_text("v2", encoding="utf-8")
    snapshots.begin_turn("turn two")
    snapshots.snapshot_file(target)
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
    snapshots.snapshot_file(target)
    assert snapshots.list_batches() == []


# --- failed-restore batch retention -----------------------------------------------------------
# A restore failure (locked file, permissions — common on Windows, the dev platform) must NOT
# cost the batch: the snapshot is the ONLY copy of the turn-start bytes, so a failed undo
# deleting it would destroy exactly the recovery data the layer exists to provide.

def test_failed_restore_keeps_batch_for_retry(workspace, monkeypatch):
    target = workspace / "a.txt"
    target.write_text("original", encoding="utf-8")
    snapshots.begin_turn("locked file")
    snapshots.snapshot_file(target)
    target.write_text("mutated", encoding="utf-8")

    real_copy2 = shutil.copy2

    def locked_copy2(*args, **kwargs):
        raise PermissionError("file is locked")

    monkeypatch.setattr(snapshots.shutil, "copy2", locked_copy2)
    _, actions = snapshots.undo_last()
    assert any("FAILED to restore a.txt" in a for a in actions)
    assert any("kept this snapshot batch" in a for a in actions)
    assert target.read_text(encoding="utf-8") == "mutated"  # nothing restored
    batches = snapshots.list_batches()
    assert len(batches) == 1 and batches[0]["files"] == ["a.txt"]  # the safety copy survives

    # Fault cleared (lock released) → a second /undo restores and pops the batch as usual.
    monkeypatch.setattr(snapshots.shutil, "copy2", real_copy2)
    _, actions = snapshots.undo_last()
    assert any("restored a.txt" in a for a in actions)
    assert target.read_text(encoding="utf-8") == "original"
    assert snapshots.list_batches() == []


def test_partial_failure_shrinks_manifest_to_unresolved(workspace, monkeypatch):
    """Mixed batch: one restore fails, one succeeds. The kept manifest holds ONLY the failed
    entry — a retry must not re-restore the succeeded file over anything written since — and
    the succeeded entry's saved bytes are pruned to match."""
    a = workspace / "a.txt"
    b = workspace / "b.txt"
    a.write_text("a-orig", encoding="utf-8")
    b.write_text("b-orig", encoding="utf-8")
    snapshots.begin_turn("two files")
    snapshots.snapshot_file(a)
    snapshots.snapshot_file(b)
    a.write_text("a-mut", encoding="utf-8")
    b.write_text("b-mut", encoding="utf-8")

    real_copy2 = shutil.copy2

    def selective_copy2(src, dst, *args, **kwargs):
        if str(src).endswith("a.txt"):
            raise PermissionError("a.txt is locked")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(snapshots.shutil, "copy2", selective_copy2)
    _, actions = snapshots.undo_last()
    assert b.read_text(encoding="utf-8") == "b-orig"  # the clean entry restored
    assert a.read_text(encoding="utf-8") == "a-mut"   # the failed one untouched

    batches = snapshots.list_batches()
    assert len(batches) == 1 and batches[0]["files"] == ["a.txt"]  # shrunk to the failure
    batch_dir = snapshots._root() / batches[0]["id"]
    kept = snapshots._load_manifest(batch_dir)["files"]
    assert (snapshots._saved(batch_dir, kept[0])).exists()                       # failed bytes retained
    assert not (batch_dir / snapshots._FILES_DIR / str(b).lstrip("/")).exists()  # resolved bytes pruned

    # State written AFTER the partial undo must survive the retry untouched.
    b.write_text("b-newer", encoding="utf-8")
    monkeypatch.setattr(snapshots.shutil, "copy2", real_copy2)
    snapshots.undo_last()
    assert a.read_text(encoding="utf-8") == "a-orig"
    assert b.read_text(encoding="utf-8") == "b-newer"
    assert snapshots.list_batches() == []


def test_sandbox_skipped_legacy_entry_keeps_batch(workspace):
    """A LEGACY entry (workspace-relative, written before 2026-09-29) whose path escapes the
    configured workspace is unresolved: the batch survives for a retry or removal by hand."""
    target = workspace / "a.txt"
    target.write_text("original", encoding="utf-8")
    snapshots.begin_turn("escapee")
    snapshots.snapshot_file(target)
    batch_dir = snapshots._batch_dirs()[-1]
    manifest = snapshots._load_manifest(batch_dir)
    manifest["files"] = [{"path": "../escapee.txt", "existed": False}]
    snapshots._save_manifest(batch_dir, manifest)

    _, actions = snapshots.undo_last()
    assert any("skipped ../escapee.txt" in a for a in actions)
    assert any("kept this snapshot batch" in a for a in actions)
    assert len(snapshots.list_batches()) == 1


def test_legacy_relative_entry_still_restores(workspace):
    target = workspace / "old.txt"
    target.write_text("original", encoding="utf-8")
    snapshots.begin_turn("legacy")
    snapshots.snapshot_file(target)
    batch_dir = snapshots._batch_dirs()[-1]
    # Rewrite the batch in the pre-2026-09-29 layout: relative path, bytes under files/<rel>.
    saved_new = snapshots._saved(batch_dir, snapshots._load_manifest(batch_dir)["files"][0])
    legacy_saved = batch_dir / snapshots._FILES_DIR / "old.txt"
    legacy_saved.write_bytes(saved_new.read_bytes())
    manifest = snapshots._load_manifest(batch_dir)
    manifest["files"] = [{"path": "old.txt", "existed": True}]
    snapshots._save_manifest(batch_dir, manifest)
    target.write_text("mutated", encoding="utf-8")

    _, actions = snapshots.undo_last()
    assert target.read_text(encoding="utf-8") == "original"
    assert any("restored old.txt" in a for a in actions)


def test_undo_restores_the_recorded_file_from_another_folder(tmp_path, isolated_paths):
    """A write made while Saturn ran in one folder is undone in THAT folder, even after Saturn
    is relaunched somewhere else holding a same-named file."""
    from core import workspace

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace.set_root(first)
    a = first / "notes.txt"
    a.write_text("first-original", encoding="utf-8")
    snapshots.begin_turn("edit in first")
    snapshots.snapshot_file(a)
    a.write_text("first-mutated", encoding="utf-8")

    workspace.set_root(second)
    b = second / "notes.txt"
    b.write_text("second-untouched", encoding="utf-8")
    snapshots.undo_last()
    assert a.read_text(encoding="utf-8") == "first-original"
    assert b.read_text(encoding="utf-8") == "second-untouched"


def test_undo_names_the_file_relative_to_the_current_folder(tmp_path, isolated_paths):
    from core import workspace

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace.set_root(first)
    a = first / "notes.txt"
    a.write_text("orig", encoding="utf-8")
    snapshots.begin_turn("edit in first")
    snapshots.snapshot_file(a)
    a.write_text("mutated", encoding="utf-8")

    workspace.set_root(second)
    listed = snapshots.list_batches()[0]["files"]
    assert listed != ["notes.txt"] and listed[0].endswith("first/notes.txt")
    _, actions = snapshots.undo_last()
    line = next(x for x in actions if x.startswith("restored"))
    assert line != "restored notes.txt" and line.endswith("first/notes.txt")


def test_undo_deletes_a_created_file_in_its_own_folder_only(tmp_path, isolated_paths):
    from core import workspace

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace.set_root(first)
    a = first / "new.txt"
    snapshots.begin_turn("create in first")
    snapshots.snapshot_file(a)
    a.write_text("created", encoding="utf-8")

    workspace.set_root(second)
    b = second / "new.txt"
    b.write_text("keep me", encoding="utf-8")
    _, actions = snapshots.undo_last()
    assert not a.exists()
    assert b.read_text(encoding="utf-8") == "keep me"
    assert any(x.startswith("deleted") and x.split()[1].endswith("first/new.txt") for x in actions)
