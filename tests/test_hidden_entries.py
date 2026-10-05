"""Hidden entries are not workspace content (2026-09-02).

A workspace may carry dotfiles (an old `.manifest.md`, `.git`, editor state), and users' directories carry
`.DS_Store`, `.git`, editor droppings. `list_directory` returned them, so "read all files" on an
empty workspace listed the manifest, read it, and relayed the agent's bookkeeping as the user's
data. One rule across the navigation tools: an entry whose name starts with '.' is never listed,
found, or searched — the same rule the manifest sync applies. read_file by explicit path is
unchanged (the user may name a dotfile on purpose).
"""

import pytest

from tools.files import find_files, list_directory, search_files


@pytest.fixture
def ws(isolated_paths):
    root = isolated_paths / "database" / "workspace"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]", encoding="utf-8")
    (root / ".manifest.md").write_text("# Document manifest\n", encoding="utf-8")
    (root / ".DS_Store").write_bytes(b"\x00\x01")
    (root / "notes.md").write_text("manifest of tasks\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / ".hidden.md").write_text("manifest", encoding="utf-8")
    (root / "sub" / "plan.md").write_text("manifest of plans\n", encoding="utf-8")
    return root


def test_list_directory_hides_dot_entries(ws):
    assert sorted(list_directory.invoke({"directory": "."})) == ["notes.md", "sub"]


def test_find_files_skips_hidden_files_and_directories(ws):
    out = find_files.invoke({"pattern": "*"})
    assert "notes.md" in out and "sub/plan.md" in out
    assert ".manifest.md" not in out and ".git" not in out and ".hidden.md" not in out


def test_search_files_skips_hidden_files_and_directories(ws):
    out = search_files.invoke({"pattern": "manifest"})
    assert "notes.md:1" in out and "sub/plan.md:1" in out
    assert ".manifest.md" not in out and ".git/config" not in out and ".hidden.md" not in out


def test_an_empty_workspace_lists_as_empty(ws):
    for name in ("notes.md", "sub/plan.md", "sub/.hidden.md"):
        (ws / name).unlink()
    (ws / "sub").rmdir()
    assert list_directory.invoke({"directory": "."}) == []
    assert find_files.invoke({"pattern": "*"}).startswith("No files matching")
