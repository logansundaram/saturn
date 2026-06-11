"""
Session persistence helpers shared by /resume (autosave + named save/load/list) and /quit.
"""
from __future__ import annotations

from pathlib import Path

from textutil import safe_stem

_SESSION_VERSION = 1
# Reserved slot that autosaves the live conversation so /resume can restore it.
# Underscore-prefixed so it's hidden from /resume list and unreachable as a save name.
_AUTOSAVE_NAME = "_autosave"


def _sessions_dir() -> Path:
    from config import get_config
    d = get_config().path("sessions")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_file(name: str) -> Path:
    """Resolve a user-supplied name to a safe `<dir>/<stem>.json` path (textutil.safe_stem —
    the same sanitizer recipes use, so the two stores can't drift onto different naming rules)."""
    return _sessions_dir() / f"{safe_stem(name, 'session')}.json"


def _autosave_file() -> Path:
    return _sessions_dir() / f"{_AUTOSAVE_NAME}.json"


def _session_payload(messages) -> dict:
    from datetime import datetime
    from langchain_core.messages import messages_to_dict

    return {
        "version": _SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages_to_dict(messages),
    }


def _read_session(path: Path):
    """Read + validate a session file -> (messages, saved_at)."""
    import json
    from langchain_core.messages import messages_from_dict
    from commands._framework import _print

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != _SESSION_VERSION:
        _print(
            f"  warning: session format v{payload.get('version')} != v{_SESSION_VERSION}; "
            "attempting to load anyway."
        )
    return messages_from_dict(payload.get("messages", [])), payload.get("saved_at", "?")


def _swap_to_messages(ctx, messages) -> None:
    """Rebuild a fresh state seeded with `messages` (mirrors /reset)."""
    state = ctx.make_initial_state()
    state["messages"] = messages
    ctx.state = state


def clear_autosave() -> bool:
    """Drop the autosave slot. For callers that have deliberately emptied the conversation
    (e.g. /rewind walking back the only turn) — write_autosave refuses to write an empty
    message list (see below), so without this the slot would keep the rewound turn and a
    crash/quit + /resume would resurrect it. Best-effort; True if a file was removed."""
    import diag

    try:
        path = _autosave_file()
        if path.exists():
            path.unlink()
            return True
    except Exception as exc:
        diag.log(f"autosave clear failed: {exc}")
    return False


def write_autosave(state: dict) -> bool:
    """Persist the live conversation to the reserved autosave slot so /resume can restore it.
    Best-effort: a write failure must never break a turn or the quit. Returns True if it wrote.
    An empty message list is a no-op ON PURPOSE: launching the app and quitting before any turn
    must not wipe the previous session's autosave — a caller that has deliberately emptied the
    conversation uses clear_autosave() instead."""
    import json
    import diag

    messages = (state or {}).get("messages", [])
    if not messages:
        return False
    try:
        _autosave_file().write_text(
            json.dumps(_session_payload(messages), indent=2), encoding="utf-8"
        )
        return True
    except Exception as exc:
        diag.log(f"autosave failed: {exc}")
        return False
