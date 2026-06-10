"""
Command dispatch framework: the @command decorator, registry, and dispatch loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable, Optional


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
    auto_approve: bool = False
    should_quit: bool = False
    # ISO timestamp of when THIS process's REPL started — the session boundary for /cost.
    session_started_at: str = ""
    # Persistent plan-review mode: when on, every turn pauses at the first plan_gate.
    review_plan: bool = False
    # A query a command wants run as an agent turn IMMEDIATELY after it returns (today only
    # /retry full, which rewinds the last turn and re-runs its question). The REPL loop consumes
    # and clears it instead of returning to the prompt.
    requeue: Optional[str] = None


Handler = Callable[["CommandContext", list[str]], None]


@dataclass
class SlashCommand:
    name: str
    summary: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    usage: str = ""
    implemented: bool = True
    details: str = ""


COMMANDS: dict[str, SlashCommand] = {}
_ALIASES: dict[str, str] = {}


def command(
    name: str,
    summary: str,
    *,
    aliases: tuple[str, ...] = (),
    usage: str = "",
    implemented: bool = True,
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
            implemented=implemented,
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
        summary = cmd.summary + ("" if cmd.implemented else "  (scaffold)")
        out.append((cmd.name.lower(), summary))
        for alias in cmd.aliases:
            out.append((alias.lower(), f"alias for /{cmd.name}"))
    return sorted(out)


def _print(line: str = "") -> None:
    print(line)


def _todo(cmd: SlashCommand, args: list[str]) -> None:
    """Uniform 'not wired yet' notice for scaffolded commands."""
    _print(f"  /{cmd.name} is scaffolded but not implemented yet.")
    _print(f"  intended: {cmd.summary}")
    if cmd.usage:
        _print(f"  usage:    {cmd.usage}")
    _print(f"  see /{cmd.name} --help for the full spec.")


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
    "system": "context",
    "sys": "context",
    "save": "resume save",
    "load": "resume",
}


def _show_help(cmd: SlashCommand) -> None:
    """`git <cmd> --help`-style detail view for one command."""
    title = "/" + cmd.name
    if cmd.aliases:
        title += "   aliases: " + ", ".join("/" + a for a in cmd.aliases)
    _print("")
    _print(f"  {title}")
    _print(f"  {'─' * min(len(title), 60)}")
    _print(f"  {cmd.summary}")
    if not cmd.implemented:
        _print("  (scaffolded — prints intended behaviour only; not yet wired)")
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
        moved = _RENAMED.get(key)
        if moved:
            _print(f"  /{key} moved — use /{moved} (see /{moved.split()[0]} --help)")
        else:
            _print(f"  unknown command: /{key} - try /help")
        return

    if args and args[0].lower() in _HELP_FLAGS:
        _show_help(cmd)
        return

    if not cmd.implemented:
        _todo(cmd, args)
        return

    try:
        cmd.handler(ctx, args)
    except Exception as exc:
        _print(f"  /{cmd.name} failed: {exc}")
