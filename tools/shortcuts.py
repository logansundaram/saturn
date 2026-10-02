"""
Shortcuts tools — list_shortcuts, run_shortcut.

The user's own Shortcuts (HomeKit scenes, Focus modes, any app's Shortcuts actions) through
macOS's `shortcuts` command line: one gated tool reaches everything they have already built.

Trust: a shortcut is a program Saturn cannot see inside — it may change anything, and it may
use the network — so `run_shortcut` is `destructive`, each run is recorded on the egress ledger
as UNTRACKED (never a send, never absent), and under air-gap `policy.airgap_holds` keeps it
from being auto-approved, exactly as for `run_shell`. The gate is relaxed one shortcut at a
time (`/policy shortcut <name>`), never by a blanket always-allow. What a shortcut prints is
`untrusted=True`: it may have fetched it.

Two tools, not one per shortcut: the bound tool schemas are part of the cached prompt prefix,
and a tool per shortcut would rewrite it every time the user edits their Shortcuts. The name is
checked against `shortcuts list` before anything runs. `shortcuts list` is 0.1s (2026-10-01).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from tools import applescript
from tools.toolspec import ToolError, register_tool
from trust import egress

_LIST_TIMEOUT = 15.0
_RUN_TIMEOUT = 120.0


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    # stdin is never the terminal: the Esc watcher is reading it (cf. tools/shell.py).
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


def names() -> list[str]:
    """The names of this Mac's shortcuts, as the Shortcuts app spells them (`shortcuts list`)."""
    applescript.mac_only()
    try:
        proc = _run(["shortcuts", "list"], _LIST_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"the shortcuts command could not list this Mac's shortcuts: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError((proc.stderr or proc.stdout or "").strip() or "the shortcuts command failed")
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


@register_tool("read_only")
def list_shortcuts():
    """List the names of the user's Shortcuts on this Mac (the Shortcuts app). Call it before
    run_shortcut to find the one that does what the user asked ("lights off", "focus mode")."""
    found = names()
    return found or "No shortcuts on this Mac."


def _read_and_remove(path: str) -> str:
    """The text a shortcut wrote to its output file, stripped; '' when it wrote nothing or the
    file is not text. The file is removed either way."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@register_tool("destructive", untrusted=True)
def run_shortcut(name: str, input: str = ""):
    """Run one of the user's Shortcuts by name and return what it outputs. `name` must be a
    name from list_shortcuts; `input` is optional text handed to the shortcut. A shortcut can do
    anything it was built to do, so this always asks the user first unless they allowlisted that
    shortcut."""
    wanted = str(name or "").strip()
    have = names()
    match = next((n for n in have if n.lower() == wanted.lower()), None) if wanted else None
    if match is None:
        raise ToolError(f"no shortcut named {wanted!r}; the shortcuts on this Mac: "
                        + (", ".join(have) or "none"))
    # The result comes back through --output-path as plain text: stdout is the CLI's own
    # channel, and a shortcut's output does not reliably land there.
    fd, result_path = tempfile.mkstemp(prefix="saturn-shortcut-", suffix=".out")
    os.close(fd)
    argv = ["shortcuts", "run", match, "--output-path", result_path, "--output-type", "public.plain-text"]
    tmp = None
    text = str(input or "")
    if text:
        fd, tmp = tempfile.mkstemp(prefix="saturn-shortcut-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        argv += ["-i", tmp]
    # Recorded before it starts, like a shell command: whether the shortcut touches the
    # network cannot be seen from here, so the ledger never claims the boundary stayed closed.
    egress.record("shortcut", "?", match, status=egress.UNTRACKED)
    try:
        proc = _run(argv, _RUN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"the shortcut {match!r} did not finish in {int(_RUN_TIMEOUT)}s; it may "
                        "still be running or waiting for an answer on screen — do not run it "
                        "again") from exc
    except OSError as exc:
        raise ToolError(f"the shortcuts command could not start: {exc}") from exc
    finally:
        written = _read_and_remove(result_path)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if proc.returncode != 0:
        raise ToolError((proc.stderr or proc.stdout or "").strip() or f"the shortcut {match!r} failed")
    out = written or (proc.stdout or "").strip()
    return {"shortcut": match, "output": out or "(the shortcut ran and returned nothing)"}
