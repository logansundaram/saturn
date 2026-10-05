"""
Apple Calendar tools — list_calendar_events, create_calendar_event, update_calendar_event,
delete_calendar_event.

Four tools over the Calendar app via AppleScript (`tools/applescript.py`). The reader is
`read_only` AND `untrusted=True`: an invitation's title, location and notes are written by
whoever sent it, so event text is treated like a web fetch — scanned and fenced by the
quarantine before the model reads it. `create_calendar_event` and `update_calendar_event`
change user data, so they are `side_effecting` and face the approval gate with the exact
calendar/title/times shown; `delete_calendar_event` is `destructive` — nothing restores a
deleted event, so its observation carries the whole event.

An event is addressed by `uid` + `calendar`, both from list_calendar_events (a uid lookup in
one named calendar is ~0.5s; measured 2026-10-01). A script reaches a recurring event only as
the whole series, so update and delete refuse one unless `whole_series` is passed — the gate
then shows that argument. An event with attendees may send them a notice from the calendar
server; the observation says how many.

Why not EventKit: a terminal-launched Python only gets calendar access if the terminal app
itself carries Apple's usage-description key, and the request fails silently otherwise (probed
2026-09-06 — see tools/applescript.py). AppleScript prompts uniformly. It is slow for range
queries (6.5s warm, 15s cold, for a window across eight calendars; 0.7s narrowed to two), so
the tool lets the agent narrow by calendar name, and the default window is one week.

Times go through the notify grammar (`notify.parse_when`, `allow_past` so a query can look
backward; the list window also takes a bare day). AppleScript dates are built field by field (`mkdate`) — the `date "…"` literal form
is locale-dependent and never used. Every failure raises ToolError (an `Error: …` observation, stamped error).
Nothing here is egress.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import notify
from tools import applescript
from tools.applescript import (AS_RS, AS_US, ISO_HANDLERS, MKDATE_HANDLER, AppleScriptError,
                                quote, records)
from tools.toolspec import ToolError, register_tool

# `d` with its time of day replaced: a bare clock time applied to a day the event already has.
_ATCLOCK_HANDLER = """
on atclock(d, secs)
  copy d to dt
  set time of dt to secs
  return dt
end atclock
"""

_HANDLERS = ISO_HANDLERS + MKDATE_HANDLER + _ATCLOCK_HANDLER

_QUERY_TIMEOUT = 90.0   # the whose-filter is slow on big calendars; one tool call may wait


def _now() -> datetime:
    return datetime.now().astimezone()


_mkdate = applescript.mkdate
_iso = applescript.local_iso


def _when(text: str, default: datetime, *, whole_day: bool = False) -> datetime:
    """Parse a user time, allowing the past; empty means `default`. `whole_day` accepts a bare
    day ('today', 'next monday') as that day's first minute — a window bound, not an event."""
    text = str(text or "").strip()
    if not text:
        return default
    return notify.parse_when(text, now=_now(), allow_past=True, whole_day=whole_day)


@register_tool("read_only", untrusted=True)
def list_calendar_events(start: str = "", end: str = "", calendars: str = ""):
    """List Apple Calendar events on this Mac between `start` and `end` (ISO 8601, 'today',
    'tomorrow', 'next monday', 'tomorrow at 09:00', 'in 3 days'; a bare day means its first
    minute, so today's events are start='today', end='tomorrow'; default: the coming week from
    today). `calendars` is an
    optional comma-separated list of calendar names to search; empty means every calendar.
    Returns calendar, uid, title, start, end, all_day, location, recurring for each event,
    soonest first."""
    try:
        today = _now().replace(hour=0, minute=0, second=0, microsecond=0)
        d1 = _when(start, today, whole_day=True)
        d2 = _when(end, d1 + timedelta(days=7), whole_day=True)
    except notify.NotifyError as exc:
        raise ToolError(str(exc)) from exc
    if d2 <= d1:
        raise ToolError(f"end {_iso(d2)} is at or before start {_iso(d1)}; put the earlier time first")
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
      set rec to recurrence of e
      if rec is missing value then set rec to ""
      set out to out & (name of c) & {AS_US} & (uid of e) & {AS_US} & (summary of e) ¬
        & {AS_US} & my iso(start date of e) & {AS_US} & my iso(end date of e) ¬
        & {AS_US} & (allday event of e as string) & {AS_US} & loc & {AS_US} & rec & {AS_RS}
    end repeat
  end repeat
end tell
return out
{_HANDLERS}"""
    try:
        rows = records(applescript.run(script, timeout=_QUERY_TIMEOUT, app="Calendar"))
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    events = [
        {"calendar": r[0], "uid": r[1], "title": r[2], "start": r[3], "end": r[4],
         "all_day": r[5] == "true", "location": r[6], "recurring": bool(r[7])}
        for r in rows if len(r) == 8
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
        raise ToolError("create_calendar_event needs the name of an existing calendar")
    if not title:
        raise ToolError("an event needs a non-empty title")
    try:
        d1 = notify.parse_when(str(start or ""), now=_now(), allow_past=True)
        d2 = _when(end, d1 + timedelta(hours=1))
    except notify.NotifyError as exc:
        raise ToolError(str(exc)) from exc
    if d2 <= d1:
        raise ToolError(f"end {_iso(d2)} is at or before start {_iso(d1)}")
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
        raise ToolError(str(exc)) from exc
    return {"uid": uid, "calendar": calendar, "title": title, "start": _iso(d1), "end": _iso(d2)}


_RECURRING = ("this is a recurring event and a change here applies to every occurrence. Pass "
              "whole_series=true only if the user wants the whole series changed; one "
              "occurrence can only be changed in Calendar")


def _find(uid: str, calendar: str, whole_series: bool) -> str:
    """The script lines that bind `e` to the event, `n` to its attendee count — and return
    "recurring" from the script for a repeating event the caller did not ask to change whole."""
    guard = "" if whole_series else '''
  set rec to recurrence of e
  if rec is missing value then set rec to ""
  if rec is not "" then return "recurring"'''
    return f"""set e to first event of calendar {quote(calendar)} whose uid is {quote(uid)}{guard}
  set n to count of attendees of e"""


def _event_error(exc: AppleScriptError, uid: str, calendar: str) -> ToolError:
    if "Invalid index" in str(exc) or "-1719" in str(exc) or "-1728" in str(exc):
        return ToolError(f"no event with uid {uid!r} in calendar {calendar!r} "
                         "(list_calendar_events gives each event's uid and calendar)")
    return ToolError(str(exc))


def _attendee_note(result: dict, count: str, verb: str) -> dict:
    if count.isdigit() and int(count) > 0:
        result["note"] = (f"this event has {int(count)} attendee(s); the calendar server may "
                          f"notify them that it was {verb}")
    return result


def _ref(uid: str, calendar: str, tool: str) -> "tuple[str, str]":
    uid, calendar = str(uid or "").strip(), str(calendar or "").strip()
    if not uid or not calendar:
        raise ToolError(f"{tool} needs the event's uid and its calendar name, both from "
                        "list_calendar_events")
    return uid, calendar


@register_tool("side_effecting")
def update_calendar_event(uid: str, calendar: str, start: str = "", end: str = "",
                          title: str = "", location: str = "", notes: str = "",
                          whole_series: bool = False):
    """Change an existing Apple Calendar event: move it (`start`, and `end` if its length
    changes — without `end` it keeps its length), retitle it, set its location, or replace its
    `notes` (the event's description). `uid` and
    `calendar` come from list_calendar_events. Times accept ISO 8601 or 'tomorrow at 14:00'; a
    bare clock time ('15:00', '3pm') keeps the event on its own day.
    A recurring event is refused unless whole_series=true (every occurrence changes).
    Use this to move or rename an event; do not create a second one."""
    uid, calendar = _ref(uid, calendar, "update_calendar_event")
    start, end = str(start or "").strip(), str(end or "").strip()
    title, location = str(title or "").strip(), str(location or "").strip()
    notes = str(notes or "").strip()
    if not (start or end or title or location or notes):
        raise ToolError("nothing to change: pass a new start, end, title, location or notes")
    try:
        # A bare clock time is a time on the EVENT's day ("move Friday's review to 3pm"), which
        # only the script knows; parse_when would make it the next 15:00 from now.
        c1, c2 = notify.clock_only(start), notify.clock_only(end)
        d1 = notify.parse_when(start, now=_now(), allow_past=True) if start and not c1 else None
        d2 = notify.parse_when(end, now=_now(), allow_past=True) if end and not c2 else None
    except notify.NotifyError as exc:
        raise ToolError(str(exc)) from exc
    if d1 and d2 and d2 <= d1:
        raise ToolError(f"end {_iso(d2)} is at or before start {_iso(d1)}")
    secs1, secs2 = (c[0] * 3600 + c[1] * 60 if c else None for c in (c1, c2))
    lines = []
    if d1 or c1:
        if c2:      # an end time on the new start's day; 23:00 to 01:00 ends the day after
            new_end = f"my atclock(d1, {secs2})\n  if d2 ≤ d1 then set d2 to d2 + (1 * days)"
        elif d2:
            new_end = _mkdate(d2) + ('\n  if d2 ≤ d1 then return "order"' if c1 else "")
        else:
            new_end = "d1 + dur"
        # Calendar refuses a save whose start is not before its end, so which date is written
        # first depends on the direction of the move (probed live 2026-10-01).
        lines.append(f"""set dur to (end date of e) - (start date of e)
  set d1 to {_mkdate(d1) if d1 else f"my atclock(start date of e, {secs1})"}
  set d2 to {new_end}
  if d1 ≥ (end date of e) then
    set end date of e to d2
    set start date of e to d1
  else
    set start date of e to d1
    set end date of e to d2
  end if""")
    elif c2:
        lines.append(f"""set d2 to my atclock(end date of e, {secs2})
  if d2 ≤ (start date of e) then return "order"
  set end date of e to d2""")
    elif d2:
        lines.append(f"set end date of e to {_mkdate(d2)}")
    if title:
        lines.append(f"set summary of e to {quote(title)}")
    if location:
        lines.append(f"set location of e to {quote(location)}")
    if notes:
        lines.append(f"set description of e to {quote(notes)}")
    changes = "\n  ".join(lines)
    script = f"""
tell application "Calendar"
  {_find(uid, calendar, whole_series)}
  {changes}
  return "ok" & {AS_US} & (summary of e) & {AS_US} & my iso(start date of e) ¬
    & {AS_US} & my iso(end date of e) & {AS_US} & (n as string)
end tell
{_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Calendar")
    except AppleScriptError as exc:
        raise _event_error(exc, uid, calendar) from exc
    if out == "recurring":
        raise ToolError(_RECURRING)
    if out == "order":
        raise ToolError("the new end is at or before the event's start; nothing was changed — "
                        "pass both start and end as full dates and times")
    parts = out.split(applescript.US)
    if len(parts) != 5 or parts[0] != "ok":
        raise ToolError(f"Calendar did not confirm the change: {out or 'no reply'}")
    result = {"uid": uid, "calendar": calendar, "title": parts[1], "start": parts[2], "end": parts[3]}
    if notes:
        result["notes"] = notes
    return _attendee_note(result, parts[4], "changed")


@register_tool("destructive")
def delete_calendar_event(uid: str, calendar: str, whole_series: bool = False):
    """Delete an Apple Calendar event for good. `uid` and `calendar` come from
    list_calendar_events. A recurring event is refused unless whole_series=true (every
    occurrence is deleted). Returns the deleted event's details."""
    uid, calendar = _ref(uid, calendar, "delete_calendar_event")
    script = f"""
tell application "Calendar"
  {_find(uid, calendar, whole_series)}
  set loc to location of e
  if loc is missing value then set loc to ""
  set res to "ok" & {AS_US} & (summary of e) & {AS_US} & my iso(start date of e) ¬
    & {AS_US} & my iso(end date of e) & {AS_US} & (n as string) & {AS_US} & loc
  delete e
  return res
end tell
{_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Calendar")
    except AppleScriptError as exc:
        raise _event_error(exc, uid, calendar) from exc
    if out == "recurring":
        raise ToolError(_RECURRING)
    parts = out.split(applescript.US)
    if len(parts) != 6 or parts[0] != "ok":
        raise ToolError(f"Calendar did not confirm the deletion: {out or 'no reply'}")
    result = {"deleted": {"uid": uid, "calendar": calendar, "title": parts[1], "start": parts[2],
                          "end": parts[3], "location": parts[5]}}
    return _attendee_note(result, parts[4], "cancelled")
