from commands._framework import command, _print
from commands._session import _session_file, _session_payload


@command(
    "save",
    "Save the current session's messages to disk.",
    usage="/save [name]",
    details="""
Serializes this session's conversation (the message history) to a JSON file under the sessions
directory (database/sessions/ by default; configurable via paths.sessions). Reload it later with
/load to continue where you left off.

With no name, a timestamped one is generated. Names are sanitized to a safe filename, and a
matching name overwrites the existing save. Only messages are persisted — per-turn scratch
(plan, iteration, tool results) is rebuilt fresh on the next turn anyway.

Examples:
  /save                 timestamped autosave
  /save research-thread named save
""",
)
def _save(ctx, args):
    import json
    from datetime import datetime

    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  nothing to save — no messages in this session yet.")
        return

    name = " ".join(args) if args else "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _session_file(name)
    existed = path.exists()
    path.write_text(json.dumps(_session_payload(messages), indent=2), encoding="utf-8")
    note = " (overwrote existing)" if existed else ""
    _print(f"  saved {len(messages)} message(s) -> {path.name}{note}")
    _print(f"  restore it with /load {path.stem}")
