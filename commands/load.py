from commands._framework import command, _print
from commands._session import _sessions_dir, _session_file, _read_session, _swap_to_messages


@command(
    "load",
    "Load a previously saved session.",
    usage="/load [name]",
    details="""
Restores a session written by /save: rebuilds a clean state and injects the saved message
history, so the conversation continues where it left off (like /reset, but seeded with the saved
messages instead of empty).

With no name, lists the available saves. Config, model bindings, and the RAG corpus are
unaffected — only the conversation is replaced.

Examples:
  /load                 list saved sessions
  /load research-thread restore one
""",
)
def _load(ctx, args):
    if not args:
        files = sorted(f for f in _sessions_dir().glob("*.json") if not f.stem.startswith("_"))
        if not files:
            _print("  no saved sessions yet — use /save [name] first.")
            return
        _print("  saved sessions:")
        for f in files:
            _print(f"    {f.stem}")
        _print("  restore one with /load <name>")
        return

    path = _session_file(" ".join(args))
    if not path.exists():
        _print(f"  no saved session named {path.stem!r} (run /load with no args to list).")
        return

    messages, saved_at = _read_session(path)
    _swap_to_messages(ctx, messages)
    _print(f"  loaded {len(messages)} message(s) from {path.name} (saved {saved_at}).")
    _print("  fresh state — conversation history restored.")
