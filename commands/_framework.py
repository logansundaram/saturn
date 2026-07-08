"""
Command dispatch framework: the @command decorator, registry, and dispatch loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable


@dataclass
class CommandContext:
    """Everything a handler is allowed to touch. Handlers mutate `state` in place (or
    reassign via `ctx.state = ...`), and flip `should_quit` to end the loop.

    `make_initial_state` is injected so handlers don't import from `agent.py` (which would
    be circular) — it's how `/reset` gets a clean state without knowing its shape."""

    state: dict
    make_initial_state: Callable[[], dict]
    db_path: str
    show_ui: bool = True
    should_quit: bool = False
    # ISO timestamp of when THIS process's REPL started — the session boundary for /cost.
    session_started_at: str = ""
    # Persistent plan-review mode: when on, every turn pauses at the first plan_gate.
    review_plan: bool = False


Handler = Callable[["CommandContext", list[str]], None]


@dataclass
class SlashCommand:
    name: str
    summary: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    usage: str = ""
    details: str = ""


COMMANDS: dict[str, SlashCommand] = {}
_ALIASES: dict[str, str] = {}


def command(
    name: str,
    summary: str,
    *,
    aliases: tuple[str, ...] = (),
    usage: str = "",
    details: str = "",
) -> Callable[[Handler], Handler]:
    """Register a slash command."""
    def register(fn: Handler) -> Handler:
        cmd = SlashCommand(
            name=name,
            summary=summary,
            handler=fn,
            aliases=aliases,
            usage=usage,
            details=details,
        )
        COMMANDS[name] = cmd
        for alias in aliases:
            _ALIASES[alias] = name
        return fn
    return register


def is_command(line: str) -> bool:
    """A command is any non-empty line whose first non-space character is `/`."""
    return line.lstrip().startswith("/")


@cache
def command_completions() -> list[tuple[str, str]]:
    """(token, summary) pairs for every invocable command — canonical names and aliases."""
    out: list[tuple[str, str]] = []
    for cmd in COMMANDS.values():
        out.append((cmd.name.lower(), cmd.summary))
        for alias in cmd.aliases:
            out.append((alias.lower(), f"alias for /{cmd.name}"))
    return sorted(out)


def _print(line: str = "") -> None:
    print(line)


_HELP_FLAGS = {"--help", "-h"}

# Old command names -> where the behaviour lives now. Typing one prints a pointer instead of a
# bare "unknown command", so muscle memory from before the /docs consolidation lands softly.
_RENAMED = {
    "ingest": "docs add",
    "forget": "docs remove",
    "remove": "docs remove",
    "reingest": "docs sync --force",
    # June 2026 focus pass: overlapping readouts + session commands consolidated.
    "workspace": "docs",
    "ws": "docs",
    "system": "config context",
    "sys": "config context",
    "save": "resume save",
    "load": "resume",
    # June 2026 trust-surface consolidation: the boundary commands fold into the /privacy front
    # door; /why becomes a /trace subview (both read the same trace DB).
    "egress": "privacy egress",
    "airgap": "privacy airgap",
    "redact": "privacy redact",
    "why": "trace why",
    # /commands was never a real command — point the habit at the command list.
    "commands": "help",
    "cmds": "help",
    # /dryrun CUT 2026-07-03 — /plan review + the per-call gate are the surviving
    # intent-before-execution surfaces.
    "dryrun": "plan review",
    "dry": "plan review",
    # 2026-07-06 surface trim: the legacy top-level gate spellings fold into /policy for good.
    # They were already thin delegations to the /policy handlers; now the pointer is the only
    # stub — one spelling to learn, zero parallel registrations to audit.
    "risk": "policy risk",
    "allow": "policy allow",
    "autoapprove": "policy open",
    "yolo": "policy open",
    # 2026-07-07 command fold: three surfaces collapsed into their canonical homes — the Glass Box
    # and the citation drill-down become /trace subviews (both already read the same trace DB /
    # accumulators), and the runtime readout joins /config (the one runtime-settings front door).
    "glass": "trace answer",
    "glassbox": "trace answer",
    "source": "trace source",
    "sources": "trace source",
    "context": "config context",
    "ctx": "config context",
}

# A second, parenthesized line for redirects whose one-line pointer doesn't tell the whole story.
_DRYRUN_NOTE = ("dry-run mode was removed — /plan review pauses at every step boundary, and the "
                "approval gate shows every gated call before it runs")
_RENAMED_NOTES: dict[str, str] = {"dryrun": _DRYRUN_NOTE, "dry": _DRYRUN_NOTE}


def _print_renamed(key: str) -> bool:
    """Print the moved-pointer for a legacy command name — the SAME line whether it arrives via
    dispatch (`/why`) or `/help why`, so neither spelling dead-ends. True when `key` was renamed."""
    moved = _RENAMED.get(key)
    if not moved:
        return False
    _print(f"  /{key} moved — use /{moved} (see /{moved.split()[0]} --help)")
    note = _RENAMED_NOTES.get(key)
    if note:
        _print(f"  ({note})")
    return True


def _show_help(cmd: SlashCommand) -> None:
    """`git <cmd> --help`-style detail view for one command."""
    title = "/" + cmd.name
    if cmd.aliases:
        title += "   aliases: " + ", ".join("/" + a for a in cmd.aliases)
    _print("")
    _print(f"  {title}")
    _print(f"  {'─' * min(len(title), 60)}")
    _print(f"  {cmd.summary}")
    _print("")
    _print(f"  usage:  {cmd.usage or ('/' + cmd.name)}")
    if cmd.details:
        _print("")
        for line in cmd.details.strip("\n").splitlines():
            _print(f"  {line}" if line.strip() else "")
    _print("")


def dispatch(line: str, ctx: CommandContext) -> None:
    """Parse and run a slash command. Always returns; signals exit via `ctx.should_quit`."""
    parts = line.lstrip().lstrip("/").split()
    if not parts:
        _print("  empty command — try /help")
        return

    key = parts[0].lower()
    args = parts[1:]

    name = key if key in COMMANDS else _ALIASES.get(key)
    cmd = COMMANDS.get(name) if name else None
    if cmd is None:
        if not _print_renamed(key):
            _print(f"  unknown command: /{key} - try /help")
        return

    # A standalone --help / -h token at the START or as the FINAL argument shows help instead of
    # executing — `/trace export --help` must explain export, never run it; `/resume save --help`
    # must not create a session named "--help". First-or-last ONLY: a mid-position token is DATA,
    # not a flag — `/memory add prefer -h over --help in CLI docs` must store the fact. Exact-token
    # match only (args are whitespace-split), so an argument merely containing the substring still
    # executes.
    if args and (args[0].lower() in _HELP_FLAGS or args[-1].lower() in _HELP_FLAGS):
        _show_help(cmd)
        return

    try:
        cmd.handler(ctx, args)
    except Exception as exc:
        _print(f"  /{cmd.name} failed: {exc}")
