"""
System commands — the app itself, in one module (the /help "system" theme; consolidated from
one-file-per-command 2026-06-11):

  /help    the themed command list (+ per-command detail)
  /quit    exit (autosaving the session)
  /update  self-update via git pull at the install root

(/config stays in commands/config.py — it is large and owns the persist seam other commands
import.)
"""

import subprocess
import sys
from pathlib import Path

from commands._framework import (
    COMMANDS,
    _ALIASES,
    _HELP_FLAGS,
    _print,
    _print_renamed,
    _show_help,
    command,
)
from commands._session import write_autosave

# ── /help ────────────────────────────────────────────────────────────────────────────────────
# The grouping table /help renders from. Static and hand-placed (deliberately NOT a new @command
# field): ≤6 themes, alphabetical inside each, and every registered built-in appears exactly
# once — tests/test_help.py cross-checks this against the live registry, so a future command
# can't silently vanish from /help.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conversation", ("clear", "resume")),
    ("knowledge & workspace", ("docs", "init", "memory", "undo")),
    ("trust & control", ("plan", "policy", "privacy")),
    ("observability", ("mcp", "models", "tools", "trace")),
    ("system", ("config", "help", "quit", "update")),
)
# (The legacy gate spellings — /risk · /allow · /autoapprove — were CUT 2026-07-06: they were
# thin delegations to /policy's subcommands and now land on _RENAMED pointers, so the listing
# carries ONE gate-policy surface instead of four. The 2026-07-07 command fold likewise dropped
# /glass and /source into /trace subviews and /context into /config — all three are _RENAMED
# pointers now, so the observability row lists one process surface (/trace) and the runtime
# inventory, not five.)

# The three-line trust-stack map /help opens with: where the boundary POSTURE is set, where the
# live ACTIVITY shows, and where the shareable RECORD comes from.
_TRUST_MAP = (
    ("posture", "/privacy · /policy"),
    ("activity", "receipt · /trace answer · /trace"),
    ("record", "/trace export · replay"),
)


def _names(cmd) -> str:
    out = "/" + cmd.name
    if cmd.aliases:
        out += " (" + ", ".join("/" + a for a in cmd.aliases) + ")"
    return out


@command(
    "help",
    "List all slash commands by theme, or detail one.",
    aliases=("?", "h"),
    usage="/help [command]",
    details="""
With no argument, opens with the trust-stack map (posture · activity · record) then lists every
command grouped by theme.

With a command name, prints its detailed help — identical to `/<command> --help`. Renamed
commands answer here too: `/help why` prints the same pointer as typing /why.

Every command also accepts a standalone --help / -h token as its FIRST or LAST argument; it
shows this detail view instead of executing (`/trace export --help` explains export, never
runs it). A mid-position token is data, so `/memory add prefer -h over --help in docs` stores
the fact.

Examples:
  /help              the grouped command list
  /help policy       detail one command
  /policy --help     same thing, the git-style way
""",
)
def _help(ctx, args):
    if args and args[0].lower() not in _HELP_FLAGS:
        key = args[0].lstrip("/").lower()
        name = key if key in COMMANDS else _ALIASES.get(key)
        cmd = COMMANDS.get(name) if name else None
        if cmd is None:
            # Same moved-pointer dispatch prints for the bare legacy name — /help why must
            # land exactly where /why does, not on "unknown command".
            if not _print_renamed(key):
                _print(f"  unknown command: /{key} - try /help")
            return
        _show_help(cmd)
        return

    from tui import ui

    ui.section("slash commands", "/help <command> or /<command> --help for details on one")
    ui.table(list(_TRUST_MAP), styles=("dim", "accent"))

    for group, names in _GROUPS:
        rows = [
            (_names(COMMANDS[n]), (COMMANDS[n].summary, "dim"))
            for n in names
            if n in COMMANDS
        ]
        if not rows:
            continue
        _print("")
        _print(f"  {group}")
        ui.table(rows)

    _print("")


# ── /quit ────────────────────────────────────────────────────────────────────────────────────
@command(
    "quit",
    "Exit the agent.",
    aliases=("exit", "q"),
    details="""
Ends the interactive session and returns you to the shell. In-process conversation
memory is discarded; the trace DB and RAG corpus on disk are untouched.

Example:
  /quit
""",
)
def _quit(ctx, args):
    if write_autosave(ctx.state):
        _print("  session autosaved — type /resume next launch to continue.")
    ctx.should_quit = True


# ── /update ──────────────────────────────────────────────────────────────────────────────────
# Saturday ships as a git clone (install.sh / install.ps1), so the repo root IS the install.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, timeout: float = 60):
    """Run a git command at the repo root; (returncode, stdout, stderr), never raises for the
    command failing (only for git itself being absent, handled by the caller)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


@command(
    "update",
    "Update Saturday to the latest version (git pull at the install root).",
    usage="/update [--check]",
    details="""
Pulls the latest Saturday from the git remote it was installed from (the install scripts clone
the repo, so the repo root IS the install).

  /update          fast-forward pull; lists what came in; reinstalls Python dependencies if
                   requirements.txt changed in the pull
  /update --check  fetch only and report how many commits behind you are, without changing
                   anything

Fast-forward only (`git pull --ff-only`): if you have local commits or edits that diverge from
the remote, the pull refuses rather than merging on its own — resolve that in git yourself.
After a successful update, restart (/quit and relaunch) to run the new code; the running
process keeps its already-imported version.
""",
)
def _update(ctx, args):
    try:
        rc, _, _ = _git("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        _print("  git is not installed (or not on PATH) — cannot self-update.")
        return
    except subprocess.TimeoutExpired:
        _print("  git did not respond — cannot self-update.")
        return
    if rc != 0:
        _print(f"  {_REPO_ROOT} is not a git repository — was Saturday installed by hand?")
        _print("  installed via pipx/uv? update with `pipx upgrade saturn-agent` "
               "(or `uv tool upgrade saturn-agent`).")
        _print("  otherwise re-install with the install script, or replace the files yourself.")
        return

    try:
        if any(a in ("--check", "-c") for a in args):
            _print("  checking for updates…")
            rc, _, err = _git("fetch", "--quiet", timeout=120)
            if rc != 0:
                _print(f"  fetch failed: {err or 'unknown error'}")
                return
            rc, behind, err = _git("rev-list", "--count", "HEAD..@{u}")
            if rc != 0:
                _print(f"  no upstream configured for this branch: {err}")
                return
            n = int(behind or 0)
            if n == 0:
                _print("  up to date.")
            else:
                _print(f"  {n} commit(s) behind — run /update to pull them.")
            return

        _, old, _ = _git("rev-parse", "HEAD")

        rc, dirty, _ = _git("status", "--porcelain")
        if rc == 0 and dirty:
            _print("  note: you have local uncommitted changes — the pull will refuse if they conflict.")

        _print("  pulling latest…")
        rc, out, err = _git("pull", "--ff-only", timeout=300)
        if rc != 0:
            _print(f"  pull failed: {err or out or 'unknown error'}")
            _print("  (diverged from the remote? resolve it in git, then retry /update.)")
            return

        _, new, _ = _git("rev-parse", "HEAD")
        if new == old:
            _print("  already up to date.")
            return

        rc, log, _ = _git("log", "--oneline", f"{old}..{new}")
        commits = log.splitlines() if rc == 0 else []
        _print(f"  updated {old[:7]} -> {new[:7]} ({len(commits)} commit(s)):")
        for line in commits[:15]:
            _print(f"    {line}")
        if len(commits) > 15:
            _print(f"    … {len(commits) - 15} more")

        # If the pull changed the dependency list, install it — an updated module importing a
        # package that isn't there yet would otherwise greet the next launch with a stack trace.
        rc, changed, _ = _git("diff", "--name-only", old, new)
        if rc == 0 and "requirements.txt" in changed.splitlines():
            _print("  requirements.txt changed — installing dependencies (this can take a minute)…")
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if proc.returncode == 0:
                _print("  dependencies installed.")
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
                _print(f"  pip install failed: {tail[0] if tail else 'unknown error'}")
                _print(f"  run it yourself: {sys.executable} -m pip install -r requirements.txt")

        _print("  restart Saturday (/quit and relaunch) to run the new version.")
    except subprocess.TimeoutExpired:
        _print("  update timed out — check your network and try again.")
