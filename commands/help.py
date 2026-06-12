from commands._framework import (
    COMMANDS,
    _ALIASES,
    _HELP_FLAGS,
    _print,
    _print_renamed,
    _show_help,
    command,
)

# ── the /help layout ──────────────────────────────────────────────────────────────────────────
# The grouping table /help renders from. Static and hand-placed (deliberately NOT a new @command
# field): ≤6 themes, alphabetical inside each, and every registered built-in appears exactly
# once — tests/test_help.py cross-checks this against the live registry, so a future command
# can't silently vanish from /help. User-defined templates render in their own trailing "user"
# section, built live from the loader.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conversation", ("clear", "compact", "resume", "retry", "rewind")),
    ("knowledge & workspace", ("docs", "init", "memory", "undo")),
    ("trust & control", ("allow", "autoapprove", "dryrun", "plan", "policy", "privacy", "risk")),
    ("observability & proof", ("context", "glass", "mcp", "models", "source", "tools", "trace")),
    ("system", ("config", "help", "quit", "update")),
)

# The legacy gate spellings stay dispatchable, but render as ONE compact line under
# trust & control — they are views of the one policy object (policy.py), not three more
# surfaces to learn.
_GATE_VIEWS = ("risk", "allow", "autoapprove")

# The three-line trust-stack map /help opens with: where the boundary POSTURE is set, where the
# live ACTIVITY shows, and where the verifiable PROOF comes from.
_TRUST_MAP = (
    ("posture", "/privacy · /policy"),
    ("activity", "receipt · /glass · /trace"),
    ("proof", "/trace export · verify · /privacy report"),
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
With no argument, opens with the trust-stack map (posture · activity · proof) then lists every
command grouped by theme; user-defined templates (database/commands/*.md) appear under `user`.
The legacy gate spellings (/risk · /allow · /autoapprove) fold into one line — they are views
of /policy.

With a command name, prints its detailed help — identical to `/<command> --help`. Renamed
commands answer here too: `/help why` prints the same pointer as typing /why.

Every command also accepts a standalone --help / -h token anywhere in its arguments; it shows
this detail view instead of executing (`/trace export --help` explains export, never runs it).

Examples:
  /help              the grouped command list
  /help risk         detail one command
  /risk --help       same thing, the git-style way
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

    from commands.user_commands import registered_names
    from tui import ui

    ui.section("slash commands", "/help <command> or /<command> --help for details on one")
    ui.table(list(_TRUST_MAP), styles=("dim", "accent"))

    for group, names in _GROUPS:
        rows = [
            (_names(COMMANDS[n]), (COMMANDS[n].summary, "dim"))
            for n in names
            if n in COMMANDS and n not in _GATE_VIEWS
        ]
        views = [v for v in _GATE_VIEWS if v in names and v in COMMANDS]
        if not rows and not views:
            continue
        _print("")
        _print(f"  {group}")
        ui.table(rows)
        if views:
            ui.table([[("views of /policy: " + " · ".join("/" + v for v in views), "dim")]])

    user_names = sorted(n for n in registered_names() if n in COMMANDS)
    if user_names:
        _print("")
        _print("  user")
        ui.table([(_names(COMMANDS[n]), (COMMANDS[n].summary, "dim")) for n in user_names])
    _print("")
