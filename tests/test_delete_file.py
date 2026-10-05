"""
`delete_file` — a delete is a move to the Trash, so it can be undone (/undo moves it back) and
recovered from Finder after the undo history has moved on. Before it existed the model deleted
with `rm` through run_shell (2026-10-02, run 38: `rm -f ~/Downloads/test_folder/abg*.txt`),
which nothing could reverse.

Offline: the Trash is a temp folder (`files._trash_dir` is the seam); conftest's throwaway HOME
covers the default.
"""

import os
from pathlib import Path

import pytest

from core import workspace
from stores import snapshots
from tools import files
from tools.files import delete_file
from tools.toolspec import ToolError


@pytest.fixture
def launched(tmp_path, isolated_paths, monkeypatch):
    r = tmp_path / "launch"
    (r / "notes").mkdir(parents=True)
    (r / "notes" / "todo.md").write_text("buy milk\n", encoding="utf-8")
    trash = tmp_path / "Trash"
    monkeypatch.setattr(files, "_trash_dir", lambda: trash)
    workspace.set_root(r)
    return r, trash


def _refused(args) -> str:
    with pytest.raises((ToolError, PermissionError)) as info:
        delete_file.invoke(args)
    return str(info.value)


def test_delete_file_is_gated_and_trusted():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("delete_file") == "side_effecting"
    assert "delete_file" not in _UNTRUSTED


def test_delete_moves_the_file_to_the_trash(launched):
    root, trash = launched
    out = delete_file.invoke({"file_path": "notes/todo.md"})
    assert not (root / "notes" / "todo.md").exists()
    assert (trash / "todo.md").read_text(encoding="utf-8") == "buy milk\n"
    assert "notes/todo.md" in out and "Trash" in out


def test_undo_takes_it_back_out_of_the_trash(launched):
    root, trash = launched
    snapshots.begin_turn("delete todo")
    delete_file.invoke({"file_path": "notes/todo.md"})
    _, actions = snapshots.undo_last()
    assert (root / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"
    assert not (trash / "todo.md").exists()
    assert any("back to notes/todo.md" in a for a in actions)


def test_a_name_already_in_the_trash_is_kept(launched):
    root, trash = launched
    trash.mkdir()
    (trash / "todo.md").write_text("an older todo", encoding="utf-8")
    (trash / "todo 2.md").write_text("another", encoding="utf-8")
    delete_file.invoke({"file_path": "notes/todo.md"})
    assert (trash / "todo.md").read_text(encoding="utf-8") == "an older todo"
    assert (trash / "todo 3.md").read_text(encoding="utf-8") == "buy milk\n"


def test_a_folder_goes_whole_and_comes_back_whole(launched):
    root, trash = launched
    snapshots.begin_turn("delete notes")
    delete_file.invoke({"file_path": "notes"})
    assert not (root / "notes").exists() and (trash / "notes" / "todo.md").exists()
    snapshots.undo_last()
    assert (root / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"


def test_a_symlink_is_deleted_as_a_link(launched):
    root, trash = launched
    os.symlink(root / "notes" / "todo.md", root / "link.md")
    delete_file.invoke({"file_path": "link.md"})
    assert (trash / "link.md").is_symlink()
    assert (root / "notes" / "todo.md").exists()              # what it pointed to is untouched


def test_missing_and_unreachable_paths_are_refused(launched, tmp_path):
    assert "not found" in _refused({"file_path": "nope.md"})
    (tmp_path / "elsewhere.md").write_text("x", encoding="utf-8")
    _refused({"file_path": str(tmp_path / "elsewhere.md")})
    assert (tmp_path / "elsewhere.md").exists()


def test_the_working_folder_itself_is_never_deleted(launched, tmp_path):
    root, _ = launched
    assert "working folder" in _refused({"file_path": "."})
    extra = tmp_path / "added"
    extra.mkdir()
    workspace.add(extra)
    assert "working folder" in _refused({"file_path": str(extra)})
    assert root.is_dir() and extra.is_dir()


def test_a_control_file_and_its_folder_are_never_deleted(launched, monkeypatch):
    root, trash = launched
    control = root / "notes" / "permissions.json"
    control.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(files, "_control_files", lambda: {control: "holds the gate's saved permissions"})
    assert "never" in _refused({"file_path": "notes/permissions.json"})
    assert "never" in _refused({"file_path": "notes"})
    assert control.read_text(encoding="utf-8") == "{}"
    assert not trash.exists()


def test_delete_runs_the_write_hooks(launched, monkeypatch):
    seen = []
    monkeypatch.setattr(files.hooks, "before_write", lambda path, tool: seen.append(("before", tool)) or None)
    monkeypatch.setattr(files.hooks, "run", lambda event, **kw: seen.append((event, kw["tool"])))
    delete_file.invoke({"file_path": "notes/todo.md"})
    assert ("before", "delete_file") in seen and ("after-write", "delete_file") in seen


def test_a_before_write_hook_can_refuse_a_delete(launched, monkeypatch):
    root, _ = launched
    monkeypatch.setattr(files.hooks, "before_write", lambda path, tool: "not today")
    assert "not today" in _refused({"file_path": "notes/todo.md"})
    assert (root / "notes" / "todo.md").exists()


def test_the_default_trash_is_the_users(monkeypatch):
    monkeypatch.setattr(files.sys, "platform", "darwin")
    assert files._trash_dir() == Path.home() / ".Trash"
    monkeypatch.setattr(files.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert files._trash_dir() == Path.home() / ".local" / "share" / "Trash" / "files"


def test_the_prompt_points_deletes_at_the_tool():
    from core.messages import agent_sys_msg
    assert "delete_file" in str(agent_sys_msg().content)


def test_off_macos_the_trash_gets_its_info_record(launched, monkeypatch):
    """The freedesktop Trash lists (and restores) an item by its `info/<name>.trashinfo`; a
    file without one is an orphan the file manager hides."""
    root, trash = launched
    files_dir = trash / "files"
    monkeypatch.setattr(files, "_trash_dir", lambda: files_dir)
    monkeypatch.setattr(files.sys, "platform", "linux")
    delete_file.invoke({"file_path": "notes/todo.md"})
    info = (trash / "info" / "todo.md.trashinfo").read_text(encoding="utf-8")
    assert info.startswith("[Trash Info]\n") and f"Path={root / 'notes' / 'todo.md'}" in info
