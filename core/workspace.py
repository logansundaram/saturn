"""
Where Saturn works (spec: docs/superpowers/specs/2026-09-29-launch-folder-workspace-design.md).

The launch folder is the workspace: `agent.main` calls `set_root(cwd)` once, and the file tools,
the shell's working directory, the folder's SATURN.md and `/init` follow it. Folders added with
`/add-dir` are reachable too, for this session only. `resolve()` is the ONE containment check —
tools/files._resolve is a thin wrapper over it.

Nothing set (the offline tests, the benchmark, a tool imported outside the app) → the root is the
configured `paths.workspace`.
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


def normalize(path, *, follow: bool = True) -> Path:
    """A user- or model-supplied path as an absolute resolved Path: surrounding quotes stripped
    (a dragged folder arrives quoted), `~` expanded, a relative path joined onto the root.
    `follow=False` resolves the folder but keeps the last name as given — the directory ENTRY,
    so a symlink is itself and not what it points to (the source of a move)."""
    s = str(path).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    raw = Path(s).expanduser()
    p = raw if raw.is_absolute() else root() / raw
    if follow or p.name in ("", ".."):
        return p.resolve()
    return p.parent.resolve() / p.name


def same(a: Path, b: Path) -> bool:
    """Whether two paths name the same file: equal, or (both existing) the same inode — a
    case-only spelling on macOS's case-insensitive disk is the same file."""
    try:
        return a == b or (a.exists() and b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def case_only(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are ONE directory entry spelled in two cases (`readme.md` and
    `README.md` on macOS's case-insensitive disk): renaming one to the other is a real rename,
    not a move onto an existing file."""
    return (a != b and a.parent == b.parent and a.name.casefold() == b.name.casefold()
            and same(a, b))


def _inside(target: Path, folder: Path) -> bool:
    """Whether `target` is `folder` or under it. The string check answers almost every call; the
    fallback compares each existing ancestor by file identity, so a case-only spelling on macOS's
    case-insensitive disk (`~/desktop` vs `~/Desktop`) is not refused. A target that does not exist
    yet (a new file in a new subfolder) is judged by its nearest existing ancestors."""
    if target.is_relative_to(folder):
        return True
    return any(same(anc, folder) for anc in (target, *target.parents) if anc.exists())


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


def resolve(path, *, follow: bool = True) -> "tuple[Path, str | None]":
    """The ONE containment check. Returns (target, refusal): `refusal` is None when the target is
    inside the root or an added folder, else the observation the model relays — it names the
    exact /add-dir command that would allow the path. `follow=False` checks where the entry
    itself sits (`normalize`), for a tool that acts on a symlink and not through it."""
    try:
        target = normalize(path, follow=follow)
    except (ValueError, RuntimeError, OSError) as exc:  # NUL byte, symlink loop: a refusal, never a raise
        return Path(str(path)), f"Invalid path: {exc}"
    if any(_inside(target, r) for r in roots()):
        return target, None
    reach = ", ".join(display(r) for r in roots())
    refusal = f"Outside the folders Saturn can reach ({reach})."
    # Suggest a folder /add-dir will accept: the nearest ancestor that exists, never the disk root.
    folder = target
    while not folder.is_dir() and folder.parent != folder:
        folder = folder.parent
    if folder.is_dir() and not _is_fs_root(folder):
        refusal += f" Ask the user to run /add-dir {display(folder)} to allow it."
    return target, refusal


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
        if same(p, folder):
            del _extra[i]
            return True
    return False


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

    @staticmethod
    def _reachable(entry: Path) -> bool:
        """A symlink is listed only when its resolved target is inside a reachable folder."""
        try:
            resolved = entry.resolve()
        except (OSError, RuntimeError):
            return False
        return any(_inside(resolved, r) for r in roots())

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
                if entry.is_symlink() and not self._reachable(entry):
                    continue
                if seen >= WALK_MAX_ENTRIES:
                    self.capped = True
                    return
                seen += 1
                yield entry


def walk_note() -> str:
    """The line a capped walk appends to its observation."""
    return f"… stopped after {WALK_MAX_ENTRIES:,} entries — narrow the directory or pattern."
