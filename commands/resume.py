from commands._framework import command, _print
from commands._session import _autosave_file, _read_session, _swap_to_messages


@command(
    "resume",
    "Resume your most recent session (autosaved on quit / each turn).",
    aliases=("continue",),
    usage="/resume",
    details="""
Restores the last conversation from the autosave slot, so you can pick up where you left off
across restarts. Unlike /load, it needs no explicit /save: the live conversation is autosaved to a
reserved slot on /quit and after every turn (the per-turn db.sqlite checkpoints are pruned, so this
slot is what survives a quit, crash, or Ctrl-C).

Like /load, it rebuilds a fresh state seeded with the restored messages — config, model bindings,
and the RAG corpus are untouched. Typically the first thing you type in a new session.

Example:
  /resume               continue your previous session
""",
)
def _resume(ctx, args):
    path = _autosave_file()
    if not path.exists():
        _print("  no previous session to resume — nothing has been autosaved yet.")
        _print("  (a session autosaves on /quit and after each turn; or use /save then /load.)")
        return

    messages, saved_at = _read_session(path)
    if not messages:
        _print("  the autosaved session is empty — nothing to resume.")
        return
    _swap_to_messages(ctx, messages)
    _print(f"  resumed {len(messages)} message(s) from your last session (saved {saved_at}).")
    _print("  conversation history restored — continue where you left off.")
