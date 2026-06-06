from commands._framework import command, _print, COMMANDS, _ALIASES, _show_help, _HELP_FLAGS


@command(
    "help",
    "List all slash commands, or detail one.",
    aliases=("?", "h"),
    usage="/help [command]",
    details="""
With no argument, prints the full command list (scaffolds marked `*`).
With a command name, prints its detailed help — identical to `/<command> --help`.

Every command also accepts --help / -h directly.

Examples:
  /help              list all commands
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
            _print(f"  unknown command: /{key} - try /help")
            return
        _show_help(cmd)
        return

    _print("  slash commands:")
    for cmd in sorted(COMMANDS.values(), key=lambda c: c.name):
        mark = " " if cmd.implemented else "*"
        names = "/" + cmd.name
        if cmd.aliases:
            names += " (" + ", ".join("/" + a for a in cmd.aliases) + ")"
        _print(f"   {mark} {names:<22} {cmd.summary}")
    _print("   * = scaffolded, not yet implemented")
    _print("  /help <command> or /<command> --help for details on one.")
