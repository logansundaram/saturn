"""
Apple Calendar tools — list_calendar_events, create_calendar_event.

Two tools over the Calendar app via AppleScript (`tools/applescript.py`). The reader is
`read_only` AND `untrusted=True`: an invitation's title, location and notes are written by
whoever sent it, so event text is treated like a web fetch — scanned and fenced by the
quarantine before the model reads it. `create_calendar_event` changes user data, so it is
`side_effecting` and faces the approval gate with the exact calendar/title/times shown.

Why not EventKit: a terminal-launched Python only gets calendar access if the terminal app
itself carries Apple's usage-description key, and the request fails silently otherwise (probed
2026-09-06 — see tools/applescript.py). AppleScript prompts uniformly. It is slow for range
queries (6.5s warm, 15s cold, for a window across eight calendars; 0.7s narrowed to two), so
the tool lets the planner narrow by calendar name, and the default window is one week.

Times go through the notify grammar (`notify.parse_when`, `allow_past` so a query can look
backward). AppleScript dates are built field by field (`mkdate`) — the `date "…"` literal form
is locale-dependent and never used. Every failure comes back as an `Error: …` string.
Nothing here is egress.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import notify
from tools import applescript
from tools.applescript import AS_RS, AS_US, ISO_HANDLERS, AppleScriptError, quote, records
from tools.toolspec import register_tool

_HANDLERS = ISO_HANDLERS + """
on mkdate(y, m, d, secs)
  set dt to current date
  set day of dt to 1
  set year of dt to y
  set month of dt to m
  set day of dt to d
  set time of dt to secs
  return dt
end mkdate
"""

_QUERY_TIMEOUT = 90.0   # the whose-filter is slow on big calendars; one planner step may wait


def _now() -> datetime:
    return datetime.now().astimezone()


def _mkdate(dt: datetime) -> str:
    """The AppleScript expression building `dt` as a local Calendar date."""
    secs = dt.hour * 3600 + dt.minute * 60 + dt.second
    return f"my mkdate({dt.year}, {dt.month}, {dt.day}, {secs})"


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")[:16]


def _when(text: str, default: datetime) -> datetime:
    """Parse a user time, allowing the past; empty means `default`."""
    text = str(text or "").strip()
    return notify.parse_when(text, now=_now(), allow_past=True) if text else default


@register_tool("read_only", untrusted=True)
def list_calendar_events(start: str = "", end: str = "", calendars: str = ""):
    """List Apple Calendar events on this Mac between `start` and `end` (ISO 8601, 'today',
    'tomorrow at 09:00', 'in 3 days'; default: the coming week from today). `calendars` is an
    optional comma-separated list of calendar names to search; empty means every calendar.
    Returns calendar, uid, title, start, end, all_day, location for each event, soonest first."""
    try:
        today = _now().replace(hour=0, minute=0, second=0, microsecond=0)
        d1 = _when(start, today)
        d2 = _when(end, d1 + timedelta(days=7))
    except notify.NotifyError as exc:
        return f"Error: {exc}"
    if d2 <= d1:
        return f"Error: end {_iso(d2)} is at or before start {_iso(d1)}; put the earlier time first"
    names = [n.strip() for n in str(calendars or "").split(",") if n.strip()]
    if names:
        source = "repeat with cn in {" + ", ".join(quote(n) for n in names) + "}\n    set c to calendar cn"
    else:
        source = "repeat with c in (every calendar)"
    script = f"""
set d1 to {_mkdate(d1)}
set d2 to {_mkdate(d2)}
set out to ""
tell application "Calendar"
  {source}
    repeat with r in (every event of c whose start date ≤ d2 and end date ≥ d1)
      set e to contents of r
      set loc to location of e
      if loc is missing value then set loc to ""
      set out to out & (name of c) & {AS_US} & (uid of e) & {AS_US} & (summary of e) ¬
        & {AS_US} & my iso(start date of e) & {AS_US} & my iso(end date of e) ¬
        & {AS_US} & (allday event of e as string) & {AS_US} & loc & {AS_RS}
    end repeat
  end repeat
end tell
return out
{_HANDLERS}"""
    try:
        rows = records(applescript.run(script, timeout=_QUERY_TIMEOUT, app="Calendar"))
    except AppleScriptError as exc:
        return f"Error: {exc}"
    events = [
        {"calendar": r[0], "uid": r[1], "title": r[2], "start": r[3], "end": r[4],
         "all_day": r[5] == "true", "location": r[6]}
        for r in rows if len(r) == 7
    ]
    if not events:
        return f"No events between {_iso(d1)} and {_iso(d2)}."
    events.sort(key=lambda e: e["start"])
    return events


@register_tool("side_effecting")
def create_calendar_event(calendar: str, title: str, start: str, end: str = "",
                          location: str = "", notes: str = ""):
    """Create an event in the named Apple Calendar (`calendar` must be an existing calendar's
    name, e.g. 'Home' or 'Work'). `start`/`end` accept ISO 8601 or 'tomorrow at 14:00'; `end`
    defaults to one hour after `start`. Returns the new event's uid."""
    calendar = str(calendar or "").strip()
    title = str(title or "").strip()
    if not calendar:
        return "Error: create_calendar_event needs the name of an existing calendar"
    if not title:
        return "Error: an event needs a non-empty title"
    try:
        d1 = notify.parse_when(str(start or ""), now=_now(), allow_past=True)
        d2 = _when(end, d1 + timedelta(hours=1))
    except notify.NotifyError as exc:
        return f"Error: {exc}"
    if d2 <= d1:
        return f"Error: end {_iso(d2)} is at or before start {_iso(d1)}"
    props = [f"summary:{quote(title)}", f"start date:{_mkdate(d1)}", f"end date:{_mkdate(d2)}"]
    if str(location or "").strip():
        props.append(f"location:{quote(str(location).strip())}")
    if str(notes or "").strip():
        props.append(f"description:{quote(str(notes).strip())}")
    script = f"""
tell application "Calendar"
  tell calendar {quote(calendar)}
    set e to make new event with properties {{{", ".join(props)}}}
    return uid of e
  end tell
end tell
{_HANDLERS}"""
    try:
        uid = applescript.run(script, app="Calendar")
    except AppleScriptError as exc:
        return f"Error: {exc}"
    return {"uid": uid, "calendar": calendar, "title": title, "start": _iso(d1), "end": _iso(d2)}
