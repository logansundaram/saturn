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


# ── the file tools ──────────────────────────────────────────────────────────────────────────

from tools.files import edit_file, find_files, list_directory, read_file, search_files, write_file  # noqa: E402


@pytest.fixture
def launched(tmp_path, isolated_paths):
    r = tmp_path / "launch"
    (r / "notes").mkdir(parents=True)
    (r / "notes" / "todo.md").write_text("buy milk\n", encoding="utf-8")
    return workspace.set_root(r)


def test_file_tools_work_in_the_launch_folder(launched):
    assert read_file.invoke({"file_path": "notes/todo.md"}) == "buy milk\n"
    assert write_file.invoke({"file_path": "new/idea.txt", "content": "x"}) == "File created successfully"
    assert (launched / "new" / "idea.txt").read_text(encoding="utf-8") == "x"
    assert edit_file.invoke({"file_path": "notes/todo.md", "old_string": "milk",
                             "new_string": "oats"}).startswith("Edited ")
    assert sorted(list_directory.invoke({"directory": "."})) == ["new", "notes"]
    assert "notes/todo.md:1" in search_files.invoke({"pattern": "oats"})
    assert "notes/todo.md" in find_files.invoke({"pattern": "*.md"})


def test_a_path_outside_is_refused_with_the_add_dir_fix(launched, tmp_path):
    (tmp_path / "Desktop").mkdir()
    out = read_file.invoke({"file_path": str(tmp_path / "Desktop" / "x.pdf")})
    assert out.startswith("Outside the folders Saturn can reach") and "/add-dir" in out
    out = write_file.invoke({"file_path": "../escape.txt", "content": "x"})
    assert out.startswith("Outside the folders Saturn can reach")
    assert not (launched.parent / "escape.txt").exists()


def test_a_new_subfolder_inside_an_added_folder_is_writable(launched, tmp_path):
    desk = tmp_path / "Desktop"
    desk.mkdir()
    workspace.add(desk)
    out = write_file.invoke({"file_path": str(desk / "sorted" / "a.txt"), "content": "x"})
    assert out == "File created successfully"
    assert (desk / "sorted" / "a.txt").read_text(encoding="utf-8") == "x"
    assert "a.txt" in find_files.invoke({"pattern": "a.txt", "directory": str(desk)})


def test_a_deleted_launch_folder_degrades_to_refusals(launched):
    import shutil

    shutil.rmtree(launched)
    assert list_directory.invoke({"directory": "."}) == "Path is not a directory."
    assert search_files.invoke({"pattern": "x"}) == "Path is not a directory."
    with pytest.raises(FileNotFoundError):
        read_file.invoke({"file_path": "notes/todo.md"})


def test_walk_prunes_home_library_dependency_and_hidden_folders(monkeypatch, tmp_path, isolated_paths):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for d in ("Library/Caches", "proj/node_modules/pkg", "proj/.git", "proj/src", "Documents/Library"):
        (home / d).mkdir(parents=True)
    for f in ("Library/Caches/c.txt", "proj/node_modules/pkg/i.js", "proj/.git/HEAD",
              "proj/src/main.py", "Documents/Library/kept.txt", "proj/.env"):
        (home / f).write_text("needle", encoding="utf-8")
    workspace.set_root(home)
    found = find_files.invoke({"pattern": "*"})
    assert "proj/src/main.py" in found and "Documents/Library/kept.txt" in found
    assert "\nLibrary/" not in "\n" + found
    for gone in ("Caches", "node_modules", ".git", ".env"):
        assert gone not in found
    hits = search_files.invoke({"pattern": "needle"})
    assert "proj/src/main.py:1" in hits and "Caches" not in hits and "node_modules" not in hits


def test_walk_budget_stops_and_says_so(monkeypatch, launched):
    for i in range(5):
        (launched / f"f{i}.txt").write_text("needle", encoding="utf-8")
    monkeypatch.setattr(workspace, "WALK_MAX_ENTRIES", 3)
    assert "stopped after 3 entries" in find_files.invoke({"pattern": "*.txt"})
    assert "stopped after 3 entries" in search_files.invoke({"pattern": "needle"})


def test_find_files_path_patterns(launched):
    (launched / "notes" / "drafts").mkdir()
    (launched / "notes" / "drafts" / "a.md").write_text("x", encoding="utf-8")
    assert "notes/drafts/a.md" in find_files.invoke({"pattern": "notes/drafts/*.md"})
    assert "notes/drafts/a.md" in find_files.invoke({"pattern": "**/drafts/*.md"})
    assert "notes/drafts/" in find_files.invoke({"pattern": "drafts"})


# ── robustness of resolve / normalize ───────────────────────────────────────────────────────


def test_resolve_never_raises_on_an_invalid_path(root):
    target, refusal = workspace.resolve("a\x00b")
    assert refusal is not None and refusal.startswith("Invalid path:")


def test_normalize_strips_only_a_matched_pair_of_quotes(root):
    assert workspace.normalize("notes'").name == "notes'"
    assert workspace.normalize("'x y'").name == "x y"
    assert workspace.normalize('"x y"').name == "x y"


# ── symlinks in the walk ────────────────────────────────────────────────────────────────────


def test_walk_skips_symlinks_pointing_outside(launched, tmp_path):
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("needle", encoding="utf-8")
    (launched / "link.txt").symlink_to(tmp_path / "secret" / "key.txt")
    assert "link.txt" not in search_files.invoke({"pattern": "needle"})
    assert "link.txt" not in find_files.invoke({"pattern": "*.txt"})


def test_walk_keeps_symlinks_pointing_inside(launched):
    (launched / "inside.txt").symlink_to(launched / "notes" / "todo.md")
    assert "inside.txt" in find_files.invoke({"pattern": "inside.txt"})
    assert "inside.txt:1" in search_files.invoke({"pattern": "milk"})


def test_walk_survives_a_dangling_symlink(launched):
    (launched / "dead.txt").symlink_to(launched / "missing.txt")
    find_files.invoke({"pattern": "*.txt"})
    search_files.invoke({"pattern": "milk"})


# ── the shell, grounding, prompt, cleanup and /init ────────────────────────────────────────


def test_run_shell_runs_in_the_launch_folder(launched):
    from tools.shell import run_shell

    out = run_shell.invoke({"command": "pwd -P"})
    assert out.startswith("[exit code 0]") and str(launched) in out


def test_grounding_names_the_working_folder_and_reads_its_saturn_md(launched, tmp_path, monkeypatch):
    from nodes.ground import stable_grounding

    monkeypatch.setenv("SATURN_HOME", str(tmp_path / "saturn_home"))  # no global SATURN.md
    (launched / "SATURN.md").write_text("be terse", encoding="utf-8")
    desk = tmp_path / "Desktop"
    desk.mkdir()
    workspace.add(desk)
    text = stable_grounding()
    section = text.split("### Working folder\n", 1)[1].split("\n### ", 1)[0]
    assert workspace.display(launched) in section
    assert workspace.display(desk) in section and "/add-dir" in section
    assert "be terse" in text


def test_clean_collapses_the_launch_folder(launched):
    from core.context import clean

    assert clean(f"wrote {launched}/notes/a.txt in {launched}") == "wrote notes/a.txt in ."


def test_agent_prompt_points_at_the_working_folder():
    from core.messages import agent_sys_msg

    text = agent_sys_msg().content
    assert "working folder" in text and "/add-dir" in text
    assert "Workspace files" not in text


def test_init_drafts_into_the_launch_folder(isolated_paths, tmp_path):
    import commands  # noqa: F401 — registers every command
    from commands._framework import CommandContext, dispatch

    empty = tmp_path / "empty"
    empty.mkdir()
    workspace.set_root(empty)
    dispatch("/init", CommandContext(state={}, make_initial_state=dict, db_path=""))
    assert (empty / "SATURDAY.md").is_file()


def test_init_listing_uses_the_pruned_walk(monkeypatch, tmp_path):
    from commands.knowledge import _workspace_listing

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    (home / "Library" / "Caches").mkdir(parents=True)
    (home / "Library" / "Caches" / "c").write_text("x", encoding="utf-8")
    (home / "notes.md").write_text("x", encoding="utf-8")
    listing = _workspace_listing(home.resolve())
    assert "notes.md" in listing and not any("Library" in p for p in listing)


# ── /add-dir and /rm-dir ────────────────────────────────────────────────────────────────────


def _dispatch(line):
    import commands  # noqa: F401
    from commands._framework import CommandContext, dispatch

    dispatch(line, CommandContext(state={}, make_initial_state=dict, db_path=""))


def test_add_dir_adds_lists_and_rm_dir_removes(launched, tmp_path, capsys):
    desk = tmp_path / "My Desk"
    desk.mkdir()
    _dispatch(f"/add-dir {desk}")
    assert "added" in capsys.readouterr().out
    assert workspace.extra() == [desk.resolve()]
    _dispatch("/add-dir")
    out = capsys.readouterr().out
    assert workspace.display(launched) in out and workspace.display(desk) in out
    _dispatch(f'/rm-dir "{desk}"')
    assert "removed" in capsys.readouterr().out
    assert workspace.extra() == []


def test_add_dir_and_rm_dir_refusals(launched, tmp_path, capsys):
    _dispatch(f"/add-dir {tmp_path / 'nope'}")
    assert "not a folder" in capsys.readouterr().out
    _dispatch(f"/rm-dir {launched}")
    assert "started in" in capsys.readouterr().out
    _dispatch(f"/rm-dir {tmp_path}")
    assert "was not added" in capsys.readouterr().out
    _dispatch("/rm-dir")
    assert "usage" in capsys.readouterr().out.lower()
    assert workspace.roots() == [launched]


def test_both_commands_answer_help(capsys):
    _dispatch("/add-dir --help")
    assert "/add-dir" in capsys.readouterr().out
    _dispatch("/rm-dir --help")
    assert "/rm-dir" in capsys.readouterr().out


# ── launch wiring and the benchmark ─────────────────────────────────────────────────────────


def test_main_sets_the_root_to_the_launch_folder(monkeypatch, tmp_path):
    import io

    import agent

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # -p must never wait on piped input

    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setattr("sys.argv", ["saturn", "--replay", str(tmp_path / "none.json")])
    with pytest.raises(SystemExit):
        agent.main()  # the replay path exits before the root is set …
    assert workspace.root() != here.resolve()

    ran = {}
    monkeypatch.setattr("sys.argv", ["saturn", "-p", "hi"])
    monkeypatch.setattr("app.headless.run_headless", lambda args: ran.setdefault("root", workspace.root()))
    agent.main()  # … the headless path runs with it set
    assert ran["root"] == here.resolve()


def test_banner_names_the_working_folder(launched):
    from tui.ui._base import _short_cwd

    assert _short_cwd() == workspace.display(launched)


def test_the_loop_benchmark_never_plants_in_the_launch_folder(monkeypatch, launched):
    import benchmark
    from config import get_config

    monkeypatch.setattr(benchmark, "run_query", lambda graph, q: {"status": "ok", "response": "x",
                        "iterations": 1, "tools_called": [], "hygiene": 0, "capped": False})
    benchmark.run_loop_benchmark(object())
    assert not list(launched.glob("bench_*"))
    assert workspace.root() == get_config().path("workspace")


def test_refusal_names_the_nearest_existing_folder(root, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _t, refusal = workspace.resolve(str(outside / "reports" / "deep" / "q3.md"))
    assert f"/add-dir {workspace.display(outside)} to allow it." in refusal


def test_refusal_at_the_filesystem_root_has_no_add_dir_sentence(root):
    _t, refusal = workspace.resolve("/definitely-not-here.txt")
    assert refusal.startswith("Outside the folders Saturn can reach (")
    assert "/add-dir" not in refusal and refusal.endswith(").")
    _t, refusal = workspace.resolve("/")
    assert "/add-dir" not in refusal
