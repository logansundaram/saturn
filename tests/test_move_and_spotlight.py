"""
`move_file` (rename / move inside the reachable folders, reversible with /undo) and the
Spotlight half of `search_files` (macOS's content index as a candidate source for the files a
direct walk cannot reach in time, and for PDFs / .docx / .xlsx).

Offline: `mdfind` never runs — `tools.files._mdfind` is the process seam, and conftest turns
Spotlight off for every test that does not turn it on here.
"""

import subprocess

import pytest

from core import workspace
from stores import snapshots
from tools import files
from tools.files import move_file, search_files
from tools.toolspec import ToolError


@pytest.fixture
def launched(tmp_path, isolated_paths):
    r = tmp_path / "launch"
    (r / "notes").mkdir(parents=True)
    (r / "notes" / "todo.md").write_text("buy milk\n", encoding="utf-8")
    return workspace.set_root(r)


def _refused(args) -> str:
    with pytest.raises((ToolError, PermissionError)) as info:
        move_file.invoke(args)
    return str(info.value)


# ── move_file ────────────────────────────────────────────────────────────────────────────────

def test_move_file_is_gated_and_trusted():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("move_file") == "side_effecting"
    assert "move_file" not in _UNTRUSTED


def test_move_renames_a_file_and_keeps_its_bytes(launched):
    out = move_file.invoke({"source": "notes/todo.md", "destination": "notes/shopping.md"})
    assert out == "Moved notes/todo.md to notes/shopping.md"
    assert not (launched / "notes" / "todo.md").exists()
    assert (launched / "notes" / "shopping.md").read_text(encoding="utf-8") == "buy milk\n"


def test_move_into_an_existing_folder_keeps_the_name(launched):
    (launched / "archive").mkdir()
    out = move_file.invoke({"source": "notes/todo.md", "destination": "archive"})
    assert out == "Moved notes/todo.md to archive/todo.md"
    assert (launched / "archive" / "todo.md").is_file()


def test_move_creates_the_destination_folder(launched):
    move_file.invoke({"source": "notes/todo.md", "destination": "2026/09/todo.md"})
    assert (launched / "2026" / "09" / "todo.md").is_file()


def test_move_a_folder(launched):
    move_file.invoke({"source": "notes", "destination": "old-notes"})
    assert (launched / "old-notes" / "todo.md").is_file() and not (launched / "notes").exists()


def test_move_refuses_to_replace_an_existing_file_unless_told(launched):
    (launched / "b.md").write_text("keep me", encoding="utf-8")
    why = _refused({"source": "notes/todo.md", "destination": "b.md"})
    assert "already exists" in why and "overwrite=true" in why
    assert (launched / "b.md").read_text(encoding="utf-8") == "keep me"
    move_file.invoke({"source": "notes/todo.md", "destination": "b.md", "overwrite": True})
    assert (launched / "b.md").read_text(encoding="utf-8") == "buy milk\n"


def test_move_never_replaces_a_folder(launched):
    (launched / "a.txt").write_text("x", encoding="utf-8")
    (launched / "box" / "a.txt").mkdir(parents=True)          # a FOLDER named a.txt in the way
    why = _refused({"source": "a.txt", "destination": "box", "overwrite": True})
    assert "folder" in why
    assert (launched / "a.txt").is_file()


def test_move_missing_source_and_same_place(launched):
    assert "not found" in _refused({"source": "nope.md", "destination": "x.md"})
    assert "same" in _refused({"source": "notes/todo.md", "destination": "notes/todo.md"})


def test_move_refuses_a_folder_into_itself(launched):
    assert "inside itself" in _refused({"source": "notes", "destination": "notes/deeper/notes"})


def test_move_refuses_a_folder_into_itself_under_another_spelling(launched):
    # On a case-insensitive disk Notes/ IS notes/: the rename fails, and shutil.move would then
    # copy the folder into itself and delete the original — the copy with it.
    if not (launched / "NOTES").exists():
        pytest.skip("needs a case-insensitive disk")
    assert "inside itself" in _refused({"source": "notes", "destination": "Notes/deeper"})
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"


def test_move_outside_the_reachable_folders_is_refused(launched, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.txt").write_text("x", encoding="utf-8")
    assert "Outside the folders" in _refused({"source": "notes/todo.md", "destination": str(outside / "t.md")})
    assert "Outside the folders" in _refused({"source": str(outside / "x.txt"), "destination": "x.txt"})
    assert (launched / "notes" / "todo.md").exists() and (outside / "x.txt").exists()


def test_move_never_touches_a_control_file(launched, monkeypatch):
    control = launched / "permissions.json"
    control.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(files, "_control_files", lambda: {control: "holds the gate's saved permissions"})
    assert "never" in _refused({"source": "permissions.json", "destination": "p.bak"})
    assert "never" in _refused({"source": "notes/todo.md", "destination": "permissions.json", "overwrite": True})
    assert control.read_text(encoding="utf-8") == "{}"


def test_undo_moves_the_file_back_without_a_byte_copy(launched):
    snapshots.begin_turn("rename todo")
    move_file.invoke({"source": "notes/todo.md", "destination": "done/todo-old.md"})
    batch = snapshots._batch_dirs()[-1]
    assert not any((batch / "files").rglob("*.md"))           # a move is its own undo record
    _, actions = snapshots.undo_last()
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"
    assert not (launched / "done" / "todo-old.md").exists()
    assert any("moved done/todo-old.md back to notes/todo.md" in a for a in actions)


def test_undo_of_an_overwriting_move_restores_both_files(launched):
    (launched / "b.md").write_text("keep me", encoding="utf-8")
    snapshots.begin_turn("clobber")
    move_file.invoke({"source": "notes/todo.md", "destination": "b.md", "overwrite": True})
    snapshots.undo_last()
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"
    assert (launched / "b.md").read_text(encoding="utf-8") == "keep me"


def test_undo_keeps_the_batch_when_the_old_place_is_taken(launched):
    snapshots.begin_turn("rename")
    move_file.invoke({"source": "notes/todo.md", "destination": "moved.md"})
    (launched / "notes" / "todo.md").write_text("someone wrote a new one", encoding="utf-8")
    _, actions = snapshots.undo_last()
    assert any(a.startswith("FAILED") and "in the way" in a for a in actions)
    assert (launched / "moved.md").exists()                    # nothing was clobbered
    assert snapshots._batch_dirs()                             # ...and the record survives for a retry


def test_move_runs_the_write_hooks(launched, monkeypatch):
    seen = []
    monkeypatch.setattr(files.hooks, "before_write", lambda path, tool: seen.append(("before", tool)) or None)
    monkeypatch.setattr(files.hooks, "run", lambda event, **kw: seen.append((event, kw["tool"])))
    move_file.invoke({"source": "notes/todo.md", "destination": "t.md"})
    assert ("before", "move_file") in seen and ("after-write", "move_file") in seen


def test_no_hook_fires_for_a_move_refused_at_its_destination(launched, monkeypatch):
    control = launched / "hooks.yaml"
    control.write_text("hooks: []", encoding="utf-8")
    seen = []
    monkeypatch.setattr(files, "_control_files", lambda: {control: "holds the user's hook commands"})
    monkeypatch.setattr(files.hooks, "before_write", lambda path, tool: seen.append(str(path)) or None)
    assert "never" in _refused({"source": "notes/todo.md", "destination": "hooks.yaml", "overwrite": True})
    assert seen == [] and control.read_text(encoding="utf-8") == "hooks: []"


# ── Spotlight behind search_files ────────────────────────────────────────────────────────────

@pytest.fixture
def spotlight(monkeypatch):
    """Turn the Spotlight seam on with a canned candidate list (None = no index answer)."""
    state = {"paths": [], "queries": []}

    def fake(literal, directory):
        state["queries"].append((literal, directory))
        return state["paths"]

    monkeypatch.setattr(files, "_spotlight", fake)
    return state


def test_spotlight_query_is_scoped_to_the_folder_and_parsed(monkeypatch, tmp_path):
    seen = {}

    def fake_mdfind(argv, timeout):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, f"{tmp_path}/a.pdf\n{tmp_path}/b.txt\n", "")

    monkeypatch.setattr(files, "_platform", lambda: "darwin")
    monkeypatch.setattr(files, "_mdfind", fake_mdfind)
    out = files._spotlight_query("tax return", tmp_path)
    assert out == [tmp_path / "a.pdf", tmp_path / "b.txt"]
    assert seen["argv"] == ["mdfind", "-onlyin", str(tmp_path), 'kMDItemTextContent == "*tax return*"cd']


def test_spotlight_gives_no_answer_off_macos_or_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(files, "_platform", lambda: "linux")
    assert files._spotlight_query("x y z", tmp_path) is None
    monkeypatch.setattr(files, "_platform", lambda: "darwin")
    monkeypatch.setattr(files, "_mdfind", lambda argv, timeout: subprocess.CompletedProcess(argv, 1, "", "boom"))
    assert files._spotlight_query("x y z", tmp_path) is None

    def gone(argv, timeout):
        raise FileNotFoundError("mdfind")

    monkeypatch.setattr(files, "_mdfind", gone)
    assert files._spotlight_query("x y z", tmp_path) is None


def test_only_a_plain_phrase_is_asked_of_spotlight(launched, spotlight):
    search_files.invoke({"pattern": "milk|bread"})             # a regex: Spotlight cannot answer it
    search_files.invoke({"pattern": "mi"})                     # too short to be worth the index
    search_files.invoke({"pattern": "milk+bread"})             # `+` is a quantifier, not a character
    assert spotlight["queries"] == []
    search_files.invoke({"pattern": "buy milk"})
    assert spotlight["queries"] == [("buy milk", launched)]


def test_a_complete_walk_is_not_repeated_through_spotlight(launched, spotlight):
    spotlight["paths"] = [launched / "notes" / "todo.md"]
    out = search_files.invoke({"pattern": "buy milk"})
    assert out == "notes/todo.md:1: buy milk"                  # once, and no Spotlight note


def test_spotlight_finds_text_inside_documents_the_walk_cannot_read(launched, spotlight, monkeypatch):
    pdf = launched / "tax-2025.pdf"
    pdf.write_bytes(b"%PDF-1.4\0binary")
    monkeypatch.setattr(files.doctext, "extract",
                        lambda p: "Form 1040\nU.S. Individual Income Tax Return 2025" if str(p).endswith(".pdf") else None)
    spotlight["paths"] = [pdf]
    out = search_files.invoke({"pattern": "tax return"})
    assert out == "tax-2025.pdf:2: U.S. Individual Income Tax Return 2025"


def test_a_cut_walk_hands_the_rest_to_spotlight_and_says_so(launched, spotlight, monkeypatch):
    far = launched / "zz" / "deep.txt"
    far.parent.mkdir()
    far.write_text("line\nthe quarterly report is here\n", encoding="utf-8")
    spotlight["paths"] = [far]
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)   # the direct walk is out of time at once
    out = search_files.invoke({"pattern": "quarterly report"})
    assert out.splitlines()[0] == "zz/deep.txt:2: the quarterly report is here"
    assert "Spotlight" in out.splitlines()[-1]


def test_a_cut_walk_with_nothing_in_the_index_says_what_was_searched(launched, spotlight, monkeypatch):
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)
    out = search_files.invoke({"pattern": "no such phrase"})
    assert out.startswith("No matches for /no such phrase/") and "Spotlight" in out


def test_without_an_index_answer_the_walk_keeps_its_full_budget(launched, spotlight, monkeypatch):
    spotlight["paths"] = None
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)
    out = search_files.invoke({"pattern": "buy milk"})
    assert out == "notes/todo.md:1: buy milk"


def test_spotlight_candidates_pass_the_same_filters_as_the_walk(launched, spotlight, monkeypatch, tmp_path):
    outside = tmp_path / "elsewhere" / "leak.txt"
    outside.parent.mkdir()
    outside.write_text("secret phrase here", encoding="utf-8")
    hidden = launched / ".cache" / "h.txt"
    hidden.parent.mkdir()
    hidden.write_text("secret phrase here", encoding="utf-8")
    heavy = launched / "node_modules" / "m.txt"
    heavy.parent.mkdir()
    heavy.write_text("secret phrase here", encoding="utf-8")
    other = launched / "notes" / "log.txt"
    other.write_text("secret phrase here", encoding="utf-8")
    kept = launched / "notes" / "keep.md"
    kept.write_text("secret phrase here", encoding="utf-8")
    spotlight["paths"] = [outside, hidden, heavy, other, kept, launched / "gone.md"]
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)
    out = search_files.invoke({"pattern": "secret phrase", "file_glob": "*.md"})
    assert out.splitlines()[0] == "notes/keep.md:1: secret phrase here"
    assert "leak" not in out and "h.txt" not in out and "m.txt" not in out and "log.txt" not in out


def test_a_spotlight_candidate_must_really_contain_the_pattern(launched, spotlight, monkeypatch):
    # The index matches words across line breaks and stale content; the regex is the judge.
    stale = launched / "stale.txt"
    stale.write_text("nothing relevant", encoding="utf-8")
    spotlight["paths"] = [stale]
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)
    assert search_files.invoke({"pattern": "quarterly report"}).startswith("No matches")


def test_documents_past_the_parse_budget_are_listed_unopened(launched, spotlight, monkeypatch):
    # Opening a PDF is a full parse (a search over ~ took 11 s on fifteen of them, 2026-10-01):
    # past the budget a document is reported as an index match, with its date, and not opened.
    import os
    old, new = launched / "old.pdf", launched / "new.pdf"
    for p, stamp in ((old, 1_700_000_000), (new, 1_750_000_000)):
        p.write_bytes(b"%PDF-1.4\0")
        os.utime(p, (stamp, stamp))
    opened = []
    monkeypatch.setattr(files.doctext, "extract",
                        lambda p: opened.append(p.name) or "the tax return for 2025")
    monkeypatch.setattr(files, "_SPOTLIGHT_DOC_SECONDS", -1.0)       # no time to open any
    spotlight["paths"] = [old, new]
    out = search_files.invoke({"pattern": "tax return"}).splitlines()
    assert opened == []
    assert out[0].startswith("new.pdf: in Spotlight's index, not opened (modified 2025-06-15")
    assert out[1].startswith("old.pdf: in Spotlight's index, not opened (modified 2023-11-1")


def test_documents_are_opened_newest_first_and_after_the_text_files(launched, spotlight, monkeypatch):
    import os
    old, new, txt = launched / "old.pdf", launched / "new.pdf", launched / "zz" / "late.txt"
    txt.parent.mkdir()
    txt.write_text("my tax return notes", encoding="utf-8")
    for p, stamp in ((old, 1_700_000_000), (new, 1_750_000_000)):
        p.write_bytes(b"%PDF-1.4\0")
        os.utime(p, (stamp, stamp))
    monkeypatch.setattr(files.doctext, "extract", lambda p: "Tax Return" if str(p).endswith(".pdf") else None)
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)
    spotlight["paths"] = [old, new, txt]
    out = search_files.invoke({"pattern": "tax return"}).splitlines()
    assert [line.split(":")[0] for line in out[:3]] == ["zz/late.txt", "new.pdf", "old.pdf"]


def test_a_move_that_fails_loses_nothing(launched):
    # A folder cannot replace a file: the refusal must leave both exactly as they were.
    (launched / "b.md").write_text("keep me", encoding="utf-8")
    why = _refused({"source": "notes", "destination": "b.md", "overwrite": True})
    assert "could not move" in why
    assert (launched / "b.md").read_text(encoding="utf-8") == "keep me"
    assert (launched / "notes" / "todo.md").is_file()


# ── the 2026-10-01 review: the guard, symlinks, case-only renames, the walk budget ───────────

def test_move_never_moves_or_replaces_a_control_files_folder(launched, monkeypatch):
    # The guard on the exact file is not enough: swapping its FOLDER swaps the file.
    home = launched / ".saturn"
    home.mkdir()
    control = home / "hooks.yaml"
    control.write_text("hooks: []", encoding="utf-8")
    (launched / "staged").mkdir()
    (launched / "staged" / "hooks.yaml").write_text("planted", encoding="utf-8")
    monkeypatch.setattr(files, "_control_files", lambda: {control: "holds the user's hook commands"})
    assert "never" in _refused({"source": ".saturn", "destination": ".saturn-old"})
    assert control.read_text(encoding="utf-8") == "hooks: []"
    control.unlink()
    home.rmdir()                                               # the folder is not there (yet)
    assert "never" in _refused({"source": "staged", "destination": ".saturn"})
    assert not home.exists() and (launched / "staged" / "hooks.yaml").is_file()


def test_a_control_file_that_does_not_exist_yet_is_protected_under_any_spelling(launched, monkeypatch):
    # On macOS's case-insensitive disk HOOKS.YAML would BE hooks.yaml once written.
    control = launched / ".saturn" / "hooks.yaml"
    monkeypatch.setattr(files, "_control_files", lambda: {control: "holds the user's hook commands"})
    with pytest.raises(PermissionError):
        files.write_file.invoke({"file_path": ".SATURN/HOOKS.YAML", "content": "planted"})
    assert not (launched / ".SATURN").exists()


def test_undo_never_restores_over_a_file_whose_move_back_failed(launched):
    # b.md holds the moved file until it is back at its old place: restoring the replaced
    # b.md over it would destroy the only copy.
    (launched / "b.md").write_text("old b", encoding="utf-8")
    snapshots.begin_turn("clobber")
    move_file.invoke({"source": "notes/todo.md", "destination": "b.md", "overwrite": True})
    (launched / "notes" / "todo.md").write_text("a new one", encoding="utf-8")
    _, actions = snapshots.undo_last()
    assert (launched / "b.md").read_text(encoding="utf-8") == "buy milk\n"
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "a new one"
    assert any("in the way" in a for a in actions)
    (launched / "notes" / "todo.md").unlink()                  # the way is clear: the retry does both
    snapshots.undo_last()
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"
    assert (launched / "b.md").read_text(encoding="utf-8") == "old b"
    assert not snapshots._batch_dirs()


def test_move_moves_a_symlink_itself_not_what_it_points_to(launched):
    (launched / "link.md").symlink_to(launched / "notes" / "todo.md")
    (launched / "archive").mkdir()
    out = move_file.invoke({"source": "link.md", "destination": "archive"})
    assert out == "Moved link.md to archive/link.md"
    assert (launched / "notes" / "todo.md").is_file()          # the real file stayed put
    assert (launched / "archive" / "link.md").is_symlink() and not (launched / "link.md").is_symlink()


def test_move_renames_by_case_alone_and_undo_reverses_it(launched):
    # On a case-insensitive disk TODO.md "exists" and is the same file: still a rename.
    snapshots.begin_turn("rename")
    out = move_file.invoke({"source": "notes/todo.md", "destination": "notes/TODO.md"})
    assert out == "Moved notes/todo.md to notes/TODO.md"
    assert [p.name for p in (launched / "notes").iterdir()] == ["TODO.md"]
    snapshots.undo_last()
    assert [p.name for p in (launched / "notes").iterdir()] == ["todo.md"]
    assert (launched / "notes" / "todo.md").read_text(encoding="utf-8") == "buy milk\n"


def test_an_unindexed_folder_is_no_index_answer(monkeypatch, tmp_path):
    # mdfind exits 0 with nothing for a folder Spotlight does not index. "Nothing matches"
    # only counts when the index knows the folder at all.
    known = {"n": "(null)"}

    def fake_mdfind(argv, timeout):
        return subprocess.CompletedProcess(argv, 0, known["n"] if argv[0] == "mdls" else "", "")

    monkeypatch.setattr(files, "_platform", lambda: "darwin")
    monkeypatch.setattr(files, "_mdfind", fake_mdfind)
    assert files._spotlight_query("tax return", tmp_path) is None
    known["n"] = "2026-09-27 16:05:11 +0000"
    assert files._spotlight_query("tax return", tmp_path) == []


def test_the_walk_budget_starts_once_the_index_has_answered(launched, monkeypatch):
    # A slow mdfind must not eat the walk's time: the nearest files are still read directly.
    import time as real_time
    from types import SimpleNamespace

    clock = {"t": 100.0}
    monkeypatch.setattr(files, "time", SimpleNamespace(
        monotonic=lambda: clock["t"], strftime=real_time.strftime, localtime=real_time.localtime))

    def slow_index(literal, directory):
        clock["t"] += files._SEARCH_WALK_SECONDS + 1.0
        return []

    monkeypatch.setattr(files, "_spotlight", slow_index)
    assert search_files.invoke({"pattern": "buy milk"}) == "notes/todo.md:1: buy milk"


def test_an_index_that_gives_no_answer_does_not_eat_the_walk_budget(launched, monkeypatch):
    # mdfind hung until its timeout: the walk is the only search there is, on its full budget.
    import time as real_time
    from types import SimpleNamespace

    clock = {"t": 100.0}
    monkeypatch.setattr(files, "time", SimpleNamespace(
        monotonic=lambda: clock["t"], strftime=real_time.strftime, localtime=real_time.localtime))

    def dead_index(literal, directory):
        clock["t"] += files._SEARCH_MAX_SECONDS + 1.0
        return None

    monkeypatch.setattr(files, "_spotlight", dead_index)
    assert search_files.invoke({"pattern": "buy milk"}) == "notes/todo.md:1: buy milk"


def test_index_answers_are_recognized_under_another_spelling_of_the_folder(launched, spotlight, monkeypatch):
    # `NOTES` is `notes` on a case-insensitive disk, and the index answers with the disk's
    # spelling: its candidates are still inside the folder, and still files the walk has read.
    if not (launched / "NOTES").exists():
        pytest.skip("needs a case-insensitive disk")
    spotlight["paths"] = [launched / "notes" / "todo.md"]
    out = search_files.invoke({"pattern": "buy milk", "directory": "NOTES"})
    assert out == "notes/todo.md:1: buy milk"                  # once
    monkeypatch.setattr(files, "_SEARCH_WALK_SECONDS", -1.0)   # the walk reads nothing
    out = search_files.invoke({"pattern": "buy milk", "directory": "NOTES"})
    assert out.splitlines()[0] == "notes/todo.md:1: buy milk"


# ── a match on a long line ───────────────────────────────────────────────────────────────────

def test_search_files_shows_the_match_on_a_long_line(launched):
    """Loop benchmark, 2026-10-07: a one-line 30k-character file, the fact 16k characters in.
    Each matched line was cut to its FIRST 200 characters, so searching for the very word
    returned filler without it — seven searches by a 9b, and the answer "no bird is named".
    The line is cut around the match instead, each cut marked."""
    filler = "Levels held steady and the weather stayed overcast. " * 300
    (launched / "log.txt").write_text(filler + "The bird was a kestrel. " + filler, encoding="utf-8")
    (launched / "short.txt").write_text("a kestrel at dawn\n", encoding="utf-8")

    out = search_files.invoke({"pattern": "kestrel", "file_glob": "log.txt"})
    assert out.startswith("log.txt:1: …") and "The bird was a kestrel." in out
    assert out.endswith("…") and len(out) <= len("log.txt:1: ") + files._SEARCH_MAX_LINE
    # a line that fits is shown whole, with no marks
    assert search_files.invoke({"pattern": "kestrel", "file_glob": "short.txt"}) == \
        "short.txt:1: a kestrel at dawn"
    # a match near the start of a long line is not cut before it
    head = search_files.invoke({"pattern": "levels held", "file_glob": "log.txt"}).splitlines()[0]
    assert head.startswith("log.txt:1: Levels held steady") and head.endswith("…")


def test_search_files_still_matches_a_pattern_on_the_indentation(launched):
    (launched / "code.py").write_text("class A:\n    def run(self):\n        pass\n", encoding="utf-8")
    assert search_files.invoke({"pattern": r"^\s+def", "file_glob": "code.py"}) == "code.py:2: def run(self):"


def test_search_files_shows_each_different_match_on_a_long_line(launched):
    """The 9b never searched for the bird's name — it searched "bird|duck|heron|…", which the
    filler matches 16k characters before the fact (live probe, 2026-10-07). One entry per line
    showed only that first match. A long line now gives one entry per match whose surroundings
    differ: repeated filler collapses, the different passage shows."""
    filler = "Levels held steady and the birds were quiet. " * 300
    (launched / "log.txt").write_text(filler + "The bird was a kestrel. " + filler, encoding="utf-8")

    out = search_files.invoke({"pattern": "bird|heron", "file_glob": "log.txt"})
    assert "kestrel" in out
    assert len(out.splitlines()) <= 8                       # 601 matches, a handful of entries
    assert all(line.startswith("log.txt:1: ") for line in out.splitlines())
