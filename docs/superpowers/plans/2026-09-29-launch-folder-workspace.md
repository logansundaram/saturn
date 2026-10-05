# Launch-Folder Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `saturn` from any folder and have its file tools, shell, `/undo`, `/init` and the folder's `SATURN.md` work in that folder, with `/add-dir` and `/rm-dir` to reach other folders for the session.

**Architecture:** A new `core/workspace.py` owns where Saturn works: the launch folder (set once in `agent.main`), the session's added folders, the ONE containment check (`resolve`), and a pruned directory walk. Every consumer that read `config.path("workspace")` calls it instead; when nothing set a launch folder it falls back to that configured path, so tests and the benchmark behave exactly as before. Snapshots record absolute paths so `/undo` restores the right file from any folder.

**Tech Stack:** Python 3.11+, pathlib, pytest (offline), LangChain tools (`@register_tool`), the `commands/_framework.py` slash-command registry.

**Spec:** `docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md`

## Global Constraints

- Python 3.11+; CI runs ubuntu + macOS × 3.11–3.13. A test that depends on a case-insensitive disk must skip on a case-sensitive one.
- Tests are fully offline: no model, no network, no embedder. Use the `isolated_paths` fixture whenever a test touches configured paths.
- The gate does not change. `write_file`/`edit_file` stay `side_effecting`, `run_shell` stays `destructive`.
- Never edit the user's `config.yaml`. Defaults live in `config.default.yaml` (this plan changes none).
- Refusal text, verbatim: `Outside the folders Saturn can reach ({reach}). Ask the user to run /add-dir {folder} to allow it.`
- Walk budget constant: `WALK_MAX_ENTRIES = 50_000`. Heavy folders skipped anywhere: `node_modules`, `.git`, `__pycache__`, `.venv`, `venv`, `Pods`, `DerivedData`. `Library` is skipped only directly under home.
- Added folders are session-only, never persisted.
- Commit messages: lowercase `area: what changed`, ending with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Stage only the files a task names (`git add <paths>`); never `git add -A`, never interactive staging, never bare `git stash`.
- Run tests with the worktree venv: `.venv/bin/python -m pytest …` from `/Users/Logan/Documents/saturn-v2`.

## Prerequisite (before Task 1)

The worktree carries uncommitted work from earlier sessions in files this plan also edits: `nodes/ground.py`, `commands/knowledge.py`, `commands/system.py`, `app/repl.py`, `benchmark.py`, `CHANGELOG.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `pivot.md`. Staging those files in a task commit would sweep the earlier work into it. The human decides how that earlier work is committed; do not start Task 1 until `git status --short` shows none of those files modified, or the human says to proceed anyway.

## Review Focus

1. **A case-only spelling on macOS** (`~/desktop/x.pdf` after `/add-dir ~/Desktop`): must be allowed, not refused. Pinned in Task 1.
2. **A folder name with spaces, typed with or without quotes** (`/add-dir ~/My Folder`, `/add-dir "~/My Folder"`): must add that folder. Pinned in Tasks 1 and 5.
3. **The launch folder deleted or renamed mid-session**: tools must answer with a refusal or not-found, never crash the turn. Pinned in Task 3.
4. **A launch folder reached through a symlink** (macOS `/tmp` is `/private/tmp`): paths given in either spelling must be allowed. Pinned in Task 1.
5. **A write into a folder that doesn't exist yet inside an added folder** (`~/Desktop/sorted/a.txt`): the containment check must pass although the target and its parent don't exist. Pinned in Task 3.

---

### Task 1: The workspace module and its containment check

**Files:**
- Create: `core/workspace.py`
- Create: `tests/test_workspace.py`
- Modify: `tests/conftest.py` (add one autouse fixture after `_reset_grant_lifecycle`)

**Interfaces:**
- Consumes: `config.get_config().path("workspace") -> Path`
- Produces (used by every later task):
  - `set_root(path) -> Path`, `reset() -> None`, `root() -> Path`, `extra() -> list[Path]`, `roots() -> list[Path]`
  - `normalize(path) -> Path` — quotes stripped, `~` expanded, relative joined onto `root()`, resolved
  - `resolve(path) -> tuple[Path, str | None]` — `(target, refusal)`; refusal `None` when reachable
  - `add(path) -> tuple[Path, bool]` — `(folder, added)`; raises `ValueError(message)` when unusable
  - `remove(path) -> bool`
  - `display(path) -> str` — `~`, `~/…`, or absolute
  - `relative(target) -> str` — root-relative when inside the root (`.` for the root itself), else `display`

- [ ] **Step 1: Add the autouse reset to `tests/conftest.py`**

Append after the `_reset_grant_lifecycle` fixture:

```python
@pytest.fixture(autouse=True)
def _reset_workspace():
    """core/workspace holds the launch folder and the /add-dir folders as process state —
    clear it around every test so a root set in one test never leaks into another."""
    from core import workspace

    workspace.reset()
    yield
    workspace.reset()
```

- [ ] **Step 2: Write the failing tests** in `tests/test_workspace.py`

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'core.workspace'` (and every test in the file fails).

- [ ] **Step 4: Write `core/workspace.py`**

```python
"""
Where Saturn works (2026-09-29; spec docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md).

The launch folder is the workspace: `agent.main` calls `set_root(cwd)` once, and the file tools,
the shell's working directory, the folder's SATURN.md and `/init` follow it. Folders added with
`/add-dir` are reachable too, for this session only. `resolve()` is the ONE containment check —
tools/files._resolve is a thin wrapper over it.

Nothing set (the offline tests, the benchmark, a tool imported outside the app) → the root is the
configured `paths.workspace`, exactly the pre-2026-09-29 behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path

from config import get_config

_root: "Path | None" = None
_extra: "list[Path]" = []


def _home() -> Path:
    return Path.home().resolve()


def _is_fs_root(p: Path) -> bool:
    return p == Path(p.anchor)


def set_root(path) -> Path:
    """Make `path` the launch folder and clear the added folders. `/` is too broad and a path that
    is not a readable directory is unusable: both fall back to home. Returns the root actually
    used, so the caller can say when it differs from the folder Saturn was started in."""
    global _root
    try:
        p = Path(path).expanduser().resolve()
        usable = p.is_dir() and not _is_fs_root(p)
    except (OSError, RuntimeError):
        usable = False
    _root = p if usable else _home()
    _extra.clear()
    return _root


def reset() -> None:
    """Forget the launch folder and the added folders (tests)."""
    global _root
    _root = None
    _extra.clear()


def root() -> Path:
    return _root if _root is not None else get_config().path("workspace")


def extra() -> list[Path]:
    return list(_extra)


def roots() -> list[Path]:
    return [root(), *_extra]


def normalize(path) -> Path:
    """A user- or model-supplied path as an absolute resolved Path: surrounding quotes stripped
    (a dragged folder arrives quoted), `~` expanded, a relative path joined onto the root."""
    raw = Path(str(path).strip().strip("\"'")).expanduser()
    return (raw if raw.is_absolute() else root() / raw).resolve()


def _same(a: Path, b: Path) -> bool:
    try:
        return a == b or (a.exists() and b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def _inside(target: Path, folder: Path) -> bool:
    """Whether `target` is `folder` or under it. The string check answers almost every call; the
    fallback compares each existing ancestor by file identity, so a case-only spelling on macOS's
    case-insensitive disk (`~/desktop` vs `~/Desktop`) is not refused. A target that does not exist
    yet (a new file in a new subfolder) is judged by its nearest existing ancestors."""
    if target.is_relative_to(folder):
        return True
    return any(_same(anc, folder) for anc in (target, *target.parents) if anc.exists())


def display(path) -> str:
    """A folder or file as a person reads it: `~`, `~/…` under home, else absolute."""
    p = Path(path)
    home = _home()
    if p == home:
        return "~"
    if p.is_relative_to(home):
        return "~/" + p.relative_to(home).as_posix()
    return p.as_posix()


def relative(target) -> str:
    """A path as the tools report it: relative to the root when inside it (`.` for the root
    itself), else `display`."""
    p = Path(target)
    r = root()
    if p.is_relative_to(r):
        return p.relative_to(r).as_posix()
    return display(p)


def resolve(path) -> "tuple[Path, str | None]":
    """The ONE containment check. Returns (target, refusal): `refusal` is None when the target is
    inside the root or an added folder, else the observation the model relays — it names the
    exact /add-dir command that would allow the path."""
    target = normalize(path)
    if any(_inside(target, r) for r in roots()):
        return target, None
    folder = target if target.is_dir() else target.parent
    reach = ", ".join(display(r) for r in roots())
    return target, (f"Outside the folders Saturn can reach ({reach}). "
                    f"Ask the user to run /add-dir {display(folder)} to allow it.")


def add(path) -> "tuple[Path, bool]":
    """/add-dir: make an existing folder reachable for this session. Returns (folder, added);
    `added` is False when it was already reachable (inside the root or an earlier /add-dir).
    Raises ValueError carrying the message to print when the folder is unusable."""
    p = normalize(path)
    if _is_fs_root(p):
        raise ValueError("the whole disk is too broad — add a specific folder")
    if not p.is_dir():
        raise ValueError(f"not a folder: {display(p)}")
    if any(_inside(p, r) for r in roots()):
        return p, False
    _extra.append(p)
    return p, True


def remove(path) -> bool:
    """/rm-dir: forget a folder added with /add-dir. False when it was not one — the launch folder
    included, which cannot be removed."""
    p = normalize(path)
    for i, folder in enumerate(_extra):
        if _same(p, folder):
            del _extra[i]
            return True
    return False
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: all pass (the case-insensitive test passes on macOS, skips on Linux).

- [ ] **Step 6: Run the full suite** (the autouse fixture touches every test)

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add core/workspace.py tests/test_workspace.py tests/conftest.py
git commit -m "workspace: the launch folder, /add-dir folders and one containment check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Snapshots record absolute paths

**Files:**
- Modify: `stores/snapshots.py` (`snapshot_file`, `undo_last`, `_shrink_batch`; add `_key`, `_saved`, `_target`)
- Modify: `tools/files.py:122` and `tools/files.py:186` (the two `snapshot_file` calls)
- Modify: `tests/test_snapshots_undo.py` (new call signature; the legacy and cross-folder cases)

**Interfaces:**
- Consumes: `core.workspace.relative(target) -> str`, `core.workspace.set_root`
- Produces: `snapshot_file(target: Path) -> None` (the one-argument form Task 3 relies on). Manifest entries: new `{"path": <display>, "abs": <absolute posix>, "existed": bool}`; legacy `{"path": <workspace-relative>, "existed": bool}` still restore.

- [ ] **Step 1: Update the existing tests to the new signature and add the new cases**

In `tests/test_snapshots_undo.py`, replace every `snapshots.snapshot_file("<name>", <var>)` call with `snapshots.snapshot_file(<var>)` (lines 25, 37, 50, 52, 63, 66, 84, 97, 130, 131, 168). In `test_partial_failure_shrinks_manifest_to_unresolved`, replace the two saved-bytes assertions with:

```python
    kept = snapshots._load_manifest(batch_dir)["files"]
    assert (snapshots._saved(batch_dir, kept[0])).exists()                       # failed bytes retained
    assert not (batch_dir / snapshots._FILES_DIR / str(b).lstrip("/")).exists()  # resolved bytes pruned
```

Replace `test_sandbox_skipped_entry_keeps_batch` with a legacy-entry version, and append the two new tests:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_snapshots_undo.py -q`
Expected: FAIL — `TypeError: snapshot_file() missing 1 required positional argument: 'target'`, and `AttributeError: module 'stores.snapshots' has no attribute '_saved'`.

- [ ] **Step 3: Implement in `stores/snapshots.py`**

Replace `snapshot_file` with:

```python
def _key(entry: dict) -> str:
    """An entry's identity: its absolute path (since 2026-09-29), else the legacy
    workspace-relative path."""
    return entry.get("abs") or entry["path"]


def _saved(batch_dir: Path, entry: dict) -> Path:
    """Where an entry's turn-start bytes live inside its batch."""
    rel = entry["abs"].lstrip("/") if entry.get("abs") else entry["path"]
    return batch_dir / _FILES_DIR / rel


def _target(entry: dict) -> "tuple[Path | None, str | None]":
    """The file an entry restores, or (None, why) when it is skipped. A new entry names its
    absolute path: /undo is typed by the user, so it restores exactly the file that was written,
    whatever folder Saturn runs in now (like @file mentions, it is not a model action, so the
    containment check does not apply). A legacy entry resolves against the configured workspace
    it was recorded under and keeps the old containment check."""
    if entry.get("abs"):
        return Path(entry["abs"]), None
    workspace = get_config().path("workspace")
    target = (workspace / entry["path"]).resolve()
    if not target.is_relative_to(workspace):
        return None, "outside the current workspace"
    return target, None


def snapshot_file(target: Path) -> None:
    """Record `target` (an absolute, containment-checked path) before it is mutated. Existing file
    -> its bytes are copied into the batch; missing file -> recorded as not-existing so an undo
    deletes the file the tool is about to create. First snapshot of a path in a batch wins (it is
    the turn-start state); later writes to the same file are no-ops. Best-effort: any failure is
    logged and swallowed — the write itself must not be blocked."""
    from core import workspace

    try:
        batch_dir = _ensure_batch()
        if batch_dir is None:
            return
        target = Path(target)
        entry = {"path": workspace.relative(target), "abs": target.as_posix(),
                 "existed": target.exists()}
        manifest = _load_manifest(batch_dir)
        if any(_key(f) == entry["abs"] for f in manifest["files"]):
            return  # turn-start state already captured
        if entry["existed"]:
            saved = _saved(batch_dir, entry)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)  # byte copy — user files may not be UTF-8
        manifest["files"].append(entry)
        _save_manifest(batch_dir, manifest)
    except Exception as exc:
        diag.log(f"snapshot_file failed for {target}: {exc}")
```

In `undo_last`, replace the per-entry target lookup and the saved-bytes path. The loop becomes:

```python
    batch_dir = batches[-1]
    manifest = _load_manifest(batch_dir)
    actions: list[str] = []
    # Entries that did NOT resolve this pass (the restore raised — a locked file, permissions —
    # or a legacy path fell outside the configured workspace). Their saved bytes are the only
    # copy of the turn-start state, so they decide below whether the batch may be deleted.
    unresolved: list[dict] = []
    for entry in reversed(manifest.get("files", [])):
        rel = entry["path"]
        target, problem = _target(entry)
        if target is None:
            actions.append(f"skipped {rel} ({problem})")
            unresolved.append(entry)
            continue
        try:
            if entry.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_saved(batch_dir, entry), target)
                actions.append(f"restored {rel}")
            else:
                if target.exists():
                    target.unlink()
                actions.append(f"deleted {rel} (was created by that turn)")
        except Exception as exc:
            actions.append(f"FAILED to restore {rel}: {exc}")
            unresolved.append(entry)
            continue
```

(The lines above replace everything from `batch_dir = batches[-1]` through the loop's closing `continue`; delete the now-unused `workspace = get_config().path("workspace")` line. Leave the docstring's first paragraph; change its sentence "Restore paths are re-resolved against the CURRENT workspace and sandbox-checked, mirroring the file tools." to "A new entry restores its recorded absolute path; a legacy entry is re-resolved against the configured workspace and containment-checked.")

In `_shrink_batch`, key by `_key` and prune with `_saved`:

```python
    keep = {_key(e) for e in unresolved}  # keys are unique per batch (first snapshot wins)
    for entry in manifest.get("files", []):
        if _key(entry) in keep or not entry.get("existed"):
            continue
        try:  # best-effort: a leftover saved file is harmless once it left the manifest
            _saved(batch_dir, entry).unlink(missing_ok=True)
        except OSError as exc:
            diag.log(f"undo saved-bytes prune failed for {entry['path']}: {exc}")
    manifest["files"] = [e for e in manifest.get("files", []) if _key(e) in keep]
```

In the module docstring, change "before `write_file` / `edit_file` changes a workspace file" to "before `write_file` / `edit_file` changes a file".

- [ ] **Step 4: Update the two callers in `tools/files.py`**

Line 122 (in `write_file`) and line 186 (in `edit_file`): replace
`snapshot_file(str(target_path.relative_to(workspace)), target_path)`
with
`snapshot_file(target_path)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_snapshots_undo.py tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add stores/snapshots.py tools/files.py tests/test_snapshots_undo.py
git commit -m "undo: snapshots record absolute paths so /undo restores the right folder

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The file tools resolve through the workspace, over a pruned walk

**Files:**
- Modify: `core/workspace.py` (append `WALK_MAX_ENTRIES`, `_HEAVY_DIRS`, `Walk`)
- Modify: `tools/files.py` (`_resolve`, `_resolve_dir`, `search_files`, `find_files`, six docstrings, module docstring)
- Modify: `tests/test_gate_ux.py:747` (the refusal wording)
- Modify: `tests/test_workspace.py` (append the file-tool tests)

**Interfaces:**
- Consumes: Task 1's `resolve`, `root`, `relative`, `add`, `set_root`; Task 2's `snapshot_file(target)`
- Produces: `core.workspace.Walk(top: Path, *, dirs: bool = False)` — iterable of absolute `Path`s, attribute `capped: bool`; `core.workspace.WALK_MAX_ENTRIES` (read at iteration time, so tests may monkeypatch it). `tools.files._resolve(path) -> (root, target, error)` keeps its tuple shape (the approval preview relies on it).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_workspace.py`:

```python
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
```

In `tests/test_gate_ux.py:747`, change `"outside the workspace" in (v["note"] or "")` to `"Outside the folders Saturn can reach" in (v["note"] or "")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/test_gate_ux.py -q`
Expected: FAIL — the file tools still resolve against the configured workspace (`read_file` of `notes/todo.md` raises `FileNotFoundError`), and `AttributeError: module 'core.workspace' has no attribute 'WALK_MAX_ENTRIES'`.

- [ ] **Step 3: Append the walk to `core/workspace.py`**

```python
# ── the pruned walk (search_files, find_files, /init) ────────────────────────────────────────
# Launched from ~, a search walks the whole home directory: macOS's ~/Library alone holds
# hundreds of thousands of files, and dependency folders are noise, not the user's content.
WALK_MAX_ENTRIES = 50_000
_HEAVY_DIRS = frozenset({"node_modules", ".git", "__pycache__", ".venv", "venv", "Pods", "DerivedData"})


class Walk:
    """The entries under `top` in a stable (sorted) order, pruned: hidden entries (a dot-name —
    the rule tools/files._hidden applies), heavy dependency/build folders anywhere, and a
    `Library` folder directly under home. Yields files, and directories too when `dirs=True`, as
    absolute Paths. Stops after WALK_MAX_ENTRIES entries and sets `capped`, so a caller can say
    the result is partial."""

    def __init__(self, top: Path, *, dirs: bool = False):
        self.top = Path(top)
        self.dirs = dirs
        self.capped = False

    def __iter__(self):
        home = _home()
        seen = 0
        for dirpath, dirnames, filenames in os.walk(self.top):
            here = Path(dirpath)
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in _HEAVY_DIRS
                and not (d == "Library" and here == home)
            )
            entries = [here / d for d in dirnames] if self.dirs else []
            entries += [here / f for f in sorted(filenames) if not f.startswith(".")]
            for entry in entries:
                if seen >= WALK_MAX_ENTRIES:
                    self.capped = True
                    return
                seen += 1
                yield entry


def walk_note() -> str:
    """The line a capped walk appends to its observation."""
    return f"… stopped after {WALK_MAX_ENTRIES:,} entries — narrow the directory or pattern."
```

- [ ] **Step 4: Route `tools/files.py` through the workspace**

Add the import next to the others: `from core import workspace as _ws`.

Replace `_resolve` and `_resolve_dir`:

```python
def _resolve(path: str):
    """Resolve a path the model gave against the folders Saturn can reach
    (core/workspace.resolve — the ONE containment check). Returns (root, target, error): `error`
    is the refusal string when the path is outside every reachable folder (then `target` must
    not be used), else None. Every tool below starts here, and so does the approval preview."""
    target, refusal = _ws.resolve(path)
    return _ws.root(), target, refusal


def _resolve_dir(path: str):
    """`_resolve` for tools that need an existing directory to walk."""
    root, target, error = _resolve(path)
    if error is None and not target.is_dir():
        error = "Path is not a directory."
    return root, target, error
```

In `write_file` and `edit_file`, the first line becomes `_, target_path, error = _resolve(file_path)` (the root is no longer needed there).

Replace the body of `search_files` after the regex compile with:

```python
    matches: list[str] = []
    truncated = False
    walk = _ws.Walk(target_path)
    for path in walk:
        if len(matches) >= _SEARCH_MAX_MATCHES:
            truncated = True
            break
        if not fnmatch.fnmatch(path.name, file_glob):
            continue
        try:
            if path.stat().st_size > _SEARCH_MAX_FILE_BYTES or _is_binary(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _ws.relative(path)
        in_file = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if not rx.search(line):
                continue
            matches.append(f"{rel}:{lineno}: {truncate(line.strip(), _SEARCH_MAX_LINE)}")
            in_file += 1
            if in_file >= _SEARCH_MAX_PER_FILE:
                matches.append(f"{rel}: … more matches in this file (capped at {_SEARCH_MAX_PER_FILE})")
                break
            if len(matches) >= _SEARCH_MAX_MATCHES:
                truncated = True
                break
        if truncated:
            break

    if not matches:
        out = f"No matches for /{pattern}/ in {directory!r} (files matching {file_glob!r})."
        return out + ("\n" + _ws.walk_note() if walk.capped else "")
    out = "\n".join(matches)
    if truncated:
        out += f"\n… stopped at {_SEARCH_MAX_MATCHES} matches — narrow the pattern, directory, or file_glob."
    elif walk.capped:
        out += "\n" + _ws.walk_note()
    return out
```

Replace the body of `find_files` after the `_resolve_dir` check with:

```python
    # A bare name pattern means "anywhere under here" — that's what the asker wants from '*.md'.
    # A pattern containing '/' matches the path relative to `directory`; a leading '**/' also
    # matches at the top level (glob's zero-directory reading).
    walk = _ws.Walk(target_path, dirs=True)
    results = []
    for p in walk:
        rel_to_dir = p.relative_to(target_path).as_posix()
        if "/" in pattern:
            hit = fnmatch.fnmatch(rel_to_dir, pattern) or (
                pattern.startswith("**/") and fnmatch.fnmatch(rel_to_dir, pattern[3:]))
        else:
            hit = fnmatch.fnmatch(p.name, pattern)
        if hit:
            results.append(_ws.relative(p) + ("/" if p.is_dir() else ""))
    results.sort()
    if not results:
        out = f"No files matching {pattern!r} under {directory!r}."
        return out + ("\n" + _ws.walk_note() if walk.capped else "")
    if len(results) > _FIND_MAX_RESULTS:
        extra = len(results) - _FIND_MAX_RESULTS
        results = results[:_FIND_MAX_RESULTS] + [f"… {extra} more — narrow the pattern."]
    if walk.capped:
        results.append(_ws.walk_note())
    return "\n".join(results)
```

`_has_hidden_part` loses its last caller; delete it.

Replace the six tool docstrings' path sentences (the rest of each docstring stays):
- `read_file`: `"Reads the contents of a file and returns it as a string. file_path is relative to the working folder; an absolute or ~ path inside a folder Saturn can reach also works."`
- `write_file`: first two sentences become `"Writes content to a file. file_path is relative to the working folder; an absolute or ~ path inside a folder Saturn can reach also works."`
- `list_directory`: `"Lists the files and folders inside a directory. directory is relative to the working folder (an absolute or ~ path inside a reachable folder also works). Use '.' to list the working folder."`
- `edit_file`: `"…to an existing file in the workspace…file_path is relative to the workspace root."` becomes `"…to an existing file…file_path is relative to the working folder; an absolute or ~ path inside a reachable folder also works."`
- `search_files`: `"Searches the CONTENTS of workspace files"` → `"Searches the CONTENTS of files"`, and `"directory is a workspace-relative path to search under ('.' = whole workspace)"` → `"directory is relative to the working folder ('.' = the whole working folder)"`.
- `find_files`: `"Finds workspace files by NAME … returns their workspace-relative paths."` → `"Finds files and folders by NAME … returns their paths relative to the working folder."`

In the module docstring, replace the first two paragraphs' "`config.path("workspace")`" wording with: "Every path is resolved per call through `core/workspace.resolve` — the launch folder plus any `/add-dir` folders — so a tool call can never reach anything else."

- [ ] **Step 5: Verify no stale wording remains**

Run: `grep -n 'workspace root\|workspace-relative\|config.path("workspace")' tools/files.py`
Expected: no output.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/test_gate_ux.py tests/test_hidden_entries.py tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add core/workspace.py tools/files.py tests/test_workspace.py tests/test_gate_ux.py
git commit -m "files: resolve through the launch folder, walk home without ~/Library

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The shell, the grounding, the prompt, path cleanup and /init follow the root

**Files:**
- Modify: `tools/shell.py:118-124` and its docstrings (module lines 12-13, `run_shell`)
- Modify: `nodes/ground.py` (`_read_instructions`, `stable_grounding`; add `_working_folder_section`)
- Modify: `core/messages.py:27-29` (the agent prompt's file rule)
- Modify: `core/context.py:24-35` (`clean`)
- Modify: `commands/knowledge.py` (`_workspace_listing`, `/init`'s handler and help text)
- Modify: `tests/test_workspace.py` (append consumer tests)

**Interfaces:**
- Consumes: Task 1's `root`, `extra`, `display`, `add`, `set_root`; Task 3's `Walk`
- Produces: the stable grounding carries a `### Working folder` section (first section after `## Grounding context`).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_workspace.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k "shell or grounding or clean or prompt or init"`
Expected: FAIL — `pwd` prints the configured workspace, no `### Working folder` section, `clean` leaves the path, the prompt still says `Workspace files`, `/init` writes into the configured workspace, the listing includes `Library`.

- [ ] **Step 3: `tools/shell.py`**

Replace lines 118-119 and the `cwd=` argument:

```python
        from core import workspace

        cwd = workspace.root()
        cwd.mkdir(parents=True, exist_ok=True)  # the configured fallback may not exist yet
```

and `cwd=str(workspace),` → `cwd=str(cwd),`. In the module docstring (lines 12-13) replace "every call runs inside `config.path("workspace")` by default, resolved per call so a live `/config paths.workspace` change is honored, matching the file tools' sandbox" with "every call runs inside the working folder (core/workspace.root(): the folder Saturn was launched from)". In `run_shell`'s docstring replace "It runs inside the workspace directory by default" with "It runs inside the working folder".

- [ ] **Step 4: `nodes/ground.py`**

In `_read_instructions`, replace `workspace = get_config().path("workspace")` with:

```python
    from core import workspace as _ws

    workspace = _ws.root()
```

Add above `stable_grounding`:

```python
def _working_folder_section() -> str:
    """Where Saturn is working (core/workspace): the launch folder and the session's /add-dir
    folders. In the STABLE half — the root is fixed for the session, so the prefix cache holds;
    /add-dir and /rm-dir miss it once, like editing SATURN.md."""
    from core import workspace as _ws

    lines = [f"You are working in {_ws.display(_ws.root())}. Relative paths resolve here."]
    extra = _ws.extra()
    if extra:
        lines.append("Also reachable this session (added with /add-dir): "
                     + ", ".join(_ws.display(p) for p in extra))
    lines.append("Any other folder needs the user to run /add-dir <folder> first.")
    return "### Working folder\n" + "\n".join(lines)
```

In `stable_grounding`, change `sections = ["## Grounding context"]` to `sections = ["## Grounding context", _working_folder_section()]`.

- [ ] **Step 5: `core/messages.py`**

Replace lines 27-29:

```
- The user's own notes, documents, mail and calendar come from the matching reader tools. A \
file listed under "Workspace files" is read with read_file; the knowledge base with \
search_knowledge_base.
```

with:

```
- The user's own notes, documents, mail and calendar come from the matching reader tools. \
Files are read with read_file; relative paths are in the working folder shown in the grounding. \
For a folder outside it, ask the user to run /add-dir <folder>. The knowledge base is searched \
with search_knowledge_base.
```

- [ ] **Step 6: `core/context.py::clean`**

Replace the body with:

```python
    s = str(text)
    try:
        from core import workspace

        raw = str(workspace.root())
    except Exception:
        return s
    for form in {raw, raw.replace("\\", "/")}:
        if form:
            s = s.replace(form + "/", "").replace(form + "\\", "").replace(form, ".")
    return s
```

and in its docstring replace "absolute workspace paths … collapse to workspace-relative" with "absolute paths under the working folder (run_shell output routinely embeds them) collapse to relative ones".

- [ ] **Step 7: `commands/knowledge.py`**

Replace `_workspace_listing`:

```python
def _workspace_listing(workspace: Path) -> list[str]:
    """Paths relative to `workspace`, capped, over the pruned walk (core/workspace.Walk) — run
    from home, a sorted rglob would crawl all of ~/Library first. Best-effort."""
    from core import workspace as _ws

    out = []
    try:
        for p in _ws.Walk(workspace, dirs=True):
            if len(out) >= _MAX_LISTING:
                out.append("… (listing capped)")
                break
            out.append(p.relative_to(workspace).as_posix() + ("/" if p.is_dir() else ""))
    except OSError:
        pass
    return out
```

In `_init`, replace `workspace = get_config().path("workspace")` with:

```python
    from core import workspace as _ws

    workspace = _ws.root()
```

(delete the now-unused `from config import get_config` line in `_init` if nothing else in the function uses it). Change the command summary to `"Survey the working folder and draft SATURDAY.md (standing per-folder instructions)."` and replace the first help paragraph ("The workspace is Saturn's sandboxed working area … copy them into the workspace.") with:

```
The working folder is the folder you launched Saturn from — the directory the file tools read
and write and the shell runs in. /add-dir reaches another folder for the session.
```

and in the second paragraph replace "at that workspace root" with "in the working folder".

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/ -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add tools/shell.py nodes/ground.py core/messages.py core/context.py commands/knowledge.py tests/test_workspace.py
git commit -m "workspace: the shell, grounding, prompt and /init follow the launch folder

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: /add-dir and /rm-dir

**Files:**
- Create: `commands/workspace_dirs.py`
- Modify: `commands/__init__.py` (`_COMMAND_MODULES`)
- Modify: `commands/system.py:35` (`_GROUPS`)
- Modify: `tests/test_workspace.py` (append command tests)

**Interfaces:**
- Consumes: Task 1's `add`, `remove`, `normalize`, `root`, `extra`, `display`
- Produces: `/add-dir [path]`, `/rm-dir <path>` registered in `COMMANDS`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_workspace.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k "add_dir or rm_dir or help"`
Expected: FAIL — the dispatcher prints an unknown-command error and `workspace.extra()` stays empty.

- [ ] **Step 3: Create `commands/workspace_dirs.py`**

```python
"""
/add-dir and /rm-dir — reaching folders beyond the launch folder (2026-09-29; spec
docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md). Session only: the next
launch starts from its own folder with nothing added.
"""

from __future__ import annotations

from commands._framework import command, _print


def _path_arg(args: list[str]) -> str:
    """The rest of the line as one path, so `~/My Folder` works without quotes."""
    return " ".join(args).strip()


@command(
    "add-dir",
    "Let Saturn reach another folder for this session.",
    usage="/add-dir [path]",
    details="""
Saturn works in the folder you launched it from. /add-dir lets the file tools reach one more
folder until you quit — the agent suggests it when you ask about a file outside the folders it
can reach. Writes there still face the approval gate.

  /add-dir ~/Desktop     reach ~/Desktop for this session
  /add-dir               list the working folder and every added folder

/rm-dir takes an added folder away again.
""",
)
def _add_dir(ctx, args):
    from core import workspace

    path = _path_arg(args)
    if not path:
        _print(f"  working folder: {workspace.display(workspace.root())}")
        extra = workspace.extra()
        for folder in extra:
            _print(f"  also reachable:  {workspace.display(folder)}")
        if not extra:
            _print("  no other folders added — /add-dir <path> adds one for this session.")
        return
    try:
        folder, added = workspace.add(path)
    except ValueError as exc:
        _print(f"  {exc}")
        return
    if added:
        _print(f"  added {workspace.display(folder)} for this session.")
    else:
        _print(f"  {workspace.display(folder)} is already reachable.")


@command(
    "rm-dir",
    "Stop Saturn reaching a folder added with /add-dir.",
    usage="/rm-dir <path>",
    details="""
Takes back a folder /add-dir made reachable. The folder Saturn was launched from cannot be
removed — quit and start Saturn somewhere else instead.

  /rm-dir ~/Desktop
""",
)
def _rm_dir(ctx, args):
    from core import workspace

    path = _path_arg(args)
    if not path:
        _print("  usage: /rm-dir <path> — /add-dir lists the added folders.")
        return
    target = workspace.normalize(path)
    if target == workspace.root():
        _print("  that's the folder Saturn was started in — it can't be removed.")
        return
    if workspace.remove(path):
        _print(f"  removed {workspace.display(target)}.")
    else:
        _print(f"  {workspace.display(target)} was not added with /add-dir.")
```

- [ ] **Step 4: Register the module and group the commands**

In `commands/__init__.py`, add to `_COMMAND_MODULES` after `"trace",`:

```python
    "workspace_dirs",  # /add-dir, /rm-dir — folders beyond the launch folder
```

In `commands/system.py`, change the knowledge row of `_GROUPS` to:

```python
    ("knowledge & workspace", ("add-dir", "docs", "init", "memory", "rm-dir", "undo")),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/test_help.py tests/ -q`
Expected: all pass (`test_grouping_table_exactly_covers_live_registry` confirms both commands are grouped).

- [ ] **Step 6: Commit**

```bash
git add commands/workspace_dirs.py commands/__init__.py commands/system.py tests/test_workspace.py
git commit -m "commands: /add-dir and /rm-dir reach folders beyond the launch folder

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Launch wiring, the banner, the benchmark, and the docs

**Files:**
- Modify: `agent.py` (`main`; add `from pathlib import Path`)
- Modify: `tui/ui/_base.py:223-229` (`_short_cwd`)
- Modify: `app/repl.py` (a fallback note right after `ui.banner(...)`)
- Modify: `benchmark.py` (`_loop_fixtures.__enter__`, `grade_loop_task`, `run_trust_benchmark`)
- Modify: `tests/test_workspace.py` (append launch and benchmark tests)
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `pivot.md`, the spec's Testing bullet

**Interfaces:**
- Consumes: Task 1's `set_root`, `root`, `display`, `reset`
- Produces: nothing new for other tasks.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_workspace.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k "main or banner or benchmark"`
Expected: FAIL — `ran["root"]` is the configured workspace, the banner shows `os.getcwd()`, and the benchmark plants `bench_*` in the launch folder (it reads the root while `launched` set it).

- [ ] **Step 3: `agent.py::main`**

Add `from pathlib import Path` beside `import sys`. After the `--replay` block and before the `--yolo` block, insert:

```python
    # The launch folder is the workspace (2026-09-29, core/workspace): the file tools, the shell
    # and the folder's SATURN.md follow it in both modes. Set once, before either runs.
    from core import workspace

    try:
        launched = Path.cwd()
    except OSError:  # the folder was deleted under the shell
        launched = Path.home()
    workspace.set_root(launched)
```

- [ ] **Step 4: The banner and the fallback note**

In `tui/ui/_base.py`, replace `_short_cwd`:

```python
def _short_cwd() -> str:
    """The working folder (core/workspace.root()), with $HOME collapsed to ~, for the banner."""
    from core import workspace

    return workspace.display(workspace.root())
```

In `app/repl.py`, directly after the `ui.banner(...)` call, insert:

```python
    # Launched from "/" or an unreadable folder, core/workspace fell back to home — say so once.
    from core import workspace as _ws
    try:
        _started = Path.cwd().resolve()
    except OSError:
        _started = None
    if _started != _ws.root():
        ui.warn(f"working in {_ws.display(_ws.root())}, not the folder Saturn was started in — "
                "cd to a specific folder and restart to work there")
```

(add `from pathlib import Path` to `app/repl.py`'s imports if it is not already imported).

- [ ] **Step 5: `benchmark.py`**

The benchmark plants and grades files in the configured scratch workspace, never where it was run from. In `_loop_fixtures.__enter__`, replace `self.workspace = get_config().path("workspace")` with:

```python
        from core import workspace

        workspace.reset()  # the configured scratch folder, never the folder it was run from
        self.workspace = workspace.root()
```

`grade_loop_task` keeps `get_config().path("workspace") / target`: that is the configured folder, where the fixtures now always live. At the top of `run_trust_benchmark`'s body, add:

```python
    from core import workspace

    workspace.reset()  # probes write into the configured scratch folder, never the launch folder
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/ -q`
Expected: all pass.

- [ ] **Step 7: Docs**

`CHANGELOG.md`, under `## [Unreleased]` → `### Added`, first bullet:

```markdown
- **Saturn works where you launch it.** `cd` into any folder and run `saturn`: the file tools,
  the shell, `/undo`, `/init` and the folder's `SATURN.md` all work there, the way Claude Code
  works in a repo. Launched from `~`, your home folder is the workspace. The tools can't reach
  anything outside it on their own: ask about a file elsewhere and Saturn suggests `/add-dir
  <folder>`, which makes that folder reachable for the session; `/rm-dir` takes it away. Every
  write and shell command still faces the gate. Searches skip `~/Library`, dependency folders
  and hidden folders, and stop at 50,000 entries. `/undo` restores the exact file a turn wrote,
  whatever folder you run it from.
```

`CLAUDE.md`, in the `### Tools` section, after the first paragraph add:

```markdown
File tools, `run_shell`'s working directory, the workspace `SATURN.md` and `/init` follow
`core/workspace.py`: the launch folder (`agent.main` sets it from the cwd) plus folders added
with `/add-dir`. `workspace.resolve` is the ONE containment check (`tools/files._resolve` wraps
it); unset, the root falls back to `paths.workspace`, which is what tests and the benchmark use.
Snapshots record absolute paths, so `/undo` restores the right file from any folder.
```

`docs/ARCHITECTURE.md`, in the `core/` table after the `mentions.py` row:

```markdown
| `workspace.py` | Where Saturn works: the launch folder, `/add-dir` folders, the one containment check (`resolve`), and the pruned walk. |
```

`pivot.md`, append to the heading of item 1 (`### 1. Work where you launched — …`): ` — shipped 2026-09-29 (launch folder + /add-dir, /rm-dir; no always-listed roots)`.

The spec's Testing bullet "**Benchmark**: the loop benchmark sets its root to a temporary folder" becomes "**Benchmark**: both benchmarks reset the root, so they plant and grade in the configured scratch workspace, never the folder they were run from."

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add agent.py tui/ui/_base.py app/repl.py benchmark.py tests/test_workspace.py CHANGELOG.md CLAUDE.md docs/ARCHITECTURE.md pivot.md docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md
git commit -m "workspace: saturn works in the folder it was launched from

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 10: Live check on the 9b** (needs a running Ollama; report results, don't gate the commit on them)

From a scratch folder containing a `notes.md`, and again from `~`:
1. `saturn` — the banner names the folder.
2. "What's in notes.md?" — read in two passes.
3. "Summarize the file x.pdf on my desktop" — the answer suggests `/add-dir ~/Desktop`.
4. `/add-dir ~/Desktop`, then ask again — it reads the file.
5. "Create todo.txt with 'buy milk'" (approve at the gate), `cd` elsewhere, relaunch, `/undo` — the original folder's `todo.txt` is deleted.
