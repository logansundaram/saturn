"""core/workspace — where Saturn works (2026-09-29; spec
docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md): the launch folder, the
/add-dir folders, the one containment check, and the consumers that follow them."""

from pathlib import Path

import pytest

from core import workspace


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "proj"
    (r / "sub").mkdir(parents=True)
    (r / "sub" / "a.txt").write_text("a", encoding="utf-8")
    return workspace.set_root(r)


# ── the root ────────────────────────────────────────────────────────────────────────────────


def test_unset_root_is_the_configured_workspace(isolated_paths):
    from config import get_config

    assert workspace.root() == get_config().path("workspace")
    assert workspace.roots() == [get_config().path("workspace")]


def test_set_root_resolves_and_clears_added_folders(tmp_path, root):
    other = tmp_path / "other"
    other.mkdir()
    workspace.add(other)
    assert workspace.set_root(root) == root.resolve()
    assert workspace.extra() == []


def test_filesystem_root_and_missing_folder_fall_back_to_home(tmp_path):
    home = Path.home().resolve()
    assert workspace.set_root("/") == home
    assert workspace.set_root(tmp_path / "nope") == home


def test_root_reached_through_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    workspace.set_root(alias)
    assert workspace.root() == real.resolve()
    assert workspace.resolve(str(alias / "f.txt"))[1] is None
    assert workspace.resolve(str(real / "f.txt"))[1] is None


# ── resolve ─────────────────────────────────────────────────────────────────────────────────


def test_relative_and_absolute_paths_inside_are_allowed(root):
    for p in ("sub/a.txt", str(root / "sub" / "a.txt"), "./sub/../sub/a.txt"):
        target, refusal = workspace.resolve(p)
        assert refusal is None and target == root / "sub" / "a.txt"


def test_tilde_path_inside_a_home_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Desktop").mkdir()
    workspace.set_root(tmp_path)
    target, refusal = workspace.resolve("~/Desktop/x.pdf")
    assert refusal is None and target == tmp_path.resolve() / "Desktop" / "x.pdf"


def test_escapes_are_refused_and_name_the_add_dir_fix(root, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for p in ("../elsewhere/x.txt", str(outside / "x.txt"), str(outside)):
        _target, refusal = workspace.resolve(p)
        assert refusal is not None
        assert refusal.startswith("Outside the folders Saturn can reach (")
        assert f"/add-dir {workspace.display(outside)} to allow it." in refusal


def test_symlink_escape_is_refused(root, tmp_path):
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "k").write_text("x", encoding="utf-8")
    (root / "link").symlink_to(secret)
    assert workspace.resolve("link/k")[1] is not None


def _case_insensitive(tmp_path) -> bool:
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    return (tmp_path / "caseprobe").exists()


def test_case_only_spelling_is_not_refused_on_a_case_insensitive_disk(root, tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    desk = tmp_path / "Desktop"
    desk.mkdir()
    (desk / "x.pdf").write_text("x", encoding="utf-8")
    workspace.add(desk)
    assert workspace.resolve(str(tmp_path / "desktop" / "x.pdf"))[1] is None


# ── add / remove ────────────────────────────────────────────────────────────────────────────


def test_added_folder_is_reachable_until_removed(root, tmp_path):
    desk = tmp_path / "Desktop"
    desk.mkdir()
    folder, added = workspace.add(desk)
    assert added and folder == desk.resolve()
    assert workspace.resolve(str(desk / "new" / "deep.txt"))[1] is None
    assert workspace.remove(str(desk)) is True
    assert workspace.resolve(str(desk / "x"))[1] is not None
    assert workspace.remove(str(desk)) is False


def test_add_refuses_unusable_folders_and_skips_reachable_ones(root, tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        workspace.add(tmp_path / "nope")
    with pytest.raises(ValueError, match="too broad"):
        workspace.add("/")
    assert workspace.add(root / "sub") == ((root / "sub").resolve(), False)
    assert workspace.remove(str(root)) is False
    assert workspace.roots() == [root]


def test_quoted_path_with_spaces(root, tmp_path):
    spaced = tmp_path / "My Folder"
    spaced.mkdir()
    folder, added = workspace.add(f'"{spaced}"')
    assert added and folder == spaced.resolve()


# ── display forms ───────────────────────────────────────────────────────────────────────────


def test_display_and_relative_forms(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path.resolve()
    (home / "proj").mkdir()
    workspace.set_root(home / "proj")
    assert workspace.display(home) == "~"
    assert workspace.display(home / "proj") == "~/proj"
    assert workspace.display(Path("/opt/x")) == "/opt/x"
    assert workspace.relative(home / "proj" / "a" / "b.txt") == "a/b.txt"
    assert workspace.relative(home / "proj") == "."
    assert workspace.relative(home / "Desktop" / "x") == "~/Desktop/x"
