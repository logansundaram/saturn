"""
Scheduled-notification tool — schedule_notification.

Hands a one-shot desktop notification to the operating system's own scheduler (`notify/`), so it
fires at the requested time whether or not Saturn is still running. Registered `side_effecting`:
it changes OS state (a launchd job on macOS), so it faces the approval gate, and the gate shows
the exact title/body/time the human is agreeing to. It is NOT egress — nothing leaves the
machine — so the egress ledger is untouched.

The observation carries the resolved local time and the id so the answer can report exactly
what was scheduled (and the human can `/notify cancel <id>` it). Every failure — an
unparseable or past time, an unsupported platform, launchctl refusing — raises ToolError: the
model sees an `Error: …` observation it must relay, never invent around, and the round is
stamped error.
"""

from __future__ import annotations

from datetime import datetime

import notify
from tools.toolspec import ToolError, register_tool


def _now() -> datetime:
    return datetime.now().astimezone()


@register_tool("side_effecting")
def schedule_notification(when: str, title: str, body: str = ""):
    """Schedule a native desktop notification (a reminder) to appear at a future time, even if
    the assistant has been closed by then. `when` is a future time: ISO 8601 like
    '2026-09-06T09:00', a relative offset like 'in 20 minutes' / 'in 2 hours', or
    'tomorrow at 09:00' / '16:00'. `title` is the short headline; `body` the optional detail.
    Use for "remind me", "notify me", "ping me at …" — the reminder is delivered by the OS."""
    title = str(title or "").strip()
    if not title:
        raise ToolError("a notification needs a non-empty title")
    try:
        at = notify.parse_when(str(when or ""), now=_now())
        n = notify.Notification(id=notify.new_id(), when=at, title=title, body=str(body or "").strip())
        notify.backend().schedule(n)
    except notify.NotifyError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "id": n.id,
        "scheduled_for": at.isoformat(timespec="minutes"),
        "title": n.title,
        "body": n.body,
        "note": "delivered by the OS notifier at that time; cancel with /notify cancel " + n.id,
    }
