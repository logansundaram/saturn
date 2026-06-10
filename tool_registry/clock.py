"""
Clock tool — the time-grounding primitive.

`calculate` exists so the model never does arithmetic from memory; `current_time` is the same
idea applied to time. Local models confabulate dates constantly ("today" resolved against a
training cutoff), and without this tool the only cure was a pointless web_search. One read-only
call grounds every "today / now / this week / how long until" question in the machine's own
clock — nothing leaves the machine.
"""

from datetime import datetime, timezone

from toolspec import register_tool


@register_tool("read_only")
def current_time():
    """The current local date and time, with timezone, UTC equivalent, and weekday. Use this
    whenever the answer depends on 'today', 'now', or any relative date — never guess the
    current date from memory."""
    now = datetime.now().astimezone()
    return {
        "local": now.isoformat(timespec="seconds"),
        "utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "timezone": str(now.tzinfo),
        "weekday": now.strftime("%A"),
    }
