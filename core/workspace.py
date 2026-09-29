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
