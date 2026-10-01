"""
/add-dir and /rm-dir — reaching folders beyond the launch folder (spec
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
