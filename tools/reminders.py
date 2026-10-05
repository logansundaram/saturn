"""
Apple Reminders tools — list_reminders, create_reminder, complete_reminder.

Three tools over the Reminders app via AppleScript (`tools/applescript.py`). The reader is
`read_only` AND `untrusted=True` (a shared list carries text other people wrote); the two
writers change user data, so they are `side_effecting` and face the gate with the exact
title/time/list shown. Nothing here is egress.

`create_reminder` is where "remind me to …" belongs: the reminder lives in the Reminders app,
reaches the user's other Apple devices, and shows up in `list_reminders` until it is ticked off.
`schedule_notification` stays for a one-off alert on this Mac.

What the scripting dictionary does NOT offer (read 2026-10-01): recurrence and location. "Every
Sunday" and "when I get home" cannot be set from here — the tools say so instead of dropping
the condition silently.

Measured 2026-10-01: each Apple event to Reminders costs about a second, so a per-reminder
loop is ~1.4s a reminder; `properties of (reminders of <list> whose completed is false)` is
one event per list (~2s) and the records are formatted without further events. This Mac has no
default list (only a shared one), so `create_reminder` names the lists when it cannot pick one.
"""

from __future__ import annotations

from datetime import datetime

import notify
from tools import applescript
from tools.applescript import (AS_GS, AS_RS, AS_US, GS, ISO_HANDLERS, MKDATE_HANDLER,
                               AppleScriptError, mkdate, quote, records)
from tools.applescript import local_iso as _iso
from tools.toolspec import ToolError, register_tool

_QUERY_TIMEOUT = 90.0
_ID_PREFIX = "x-apple-reminder://"
_HANDLERS = ISO_HANDLERS + MKDATE_HANDLER


def _now() -> datetime:
    return datetime.now().astimezone()


def _no_list(exc: AppleScriptError, name: str) -> ToolError:
    if name and ("Can’t get list" in str(exc) or "Can't get list" in str(exc)):
        return ToolError(f"no Reminders list named {name!r}")
    return ToolError(str(exc))


@register_tool("read_only", untrusted=True)
def list_reminders(list: str = ""):
    """List the open (not yet completed) reminders in Apple Reminders — every list, or only the
    one named by `list`. Returns id, list, title, and when set: due, overdue, notes, flagged;
    dated ones first, soonest first. Use it for "what's on my list", "what's overdue", "what
    did I ask to be reminded of"."""
    name = str(list or "").strip()
    source = f"set ls to {{list {quote(name)}}}" if name else "set ls to lists"
    script = f"""
set out to ""
tell application "Reminders"
  {source}
  repeat with l in ls
    set ln to name of l
    set ps to properties of (reminders of l whose completed is false)
    repeat with p in ps
      set d to due date of p
      if d is missing value then
        set ds to ""
      else
        set ds to my iso(d)
      end if
      set b to body of p
      if b is missing value then set b to ""
      set out to out & (id of p) & {AS_US} & ln & {AS_US} & (name of p) & {AS_US} & ds ¬
        & {AS_US} & b & {AS_US} & (flagged of p as string) & {AS_RS}
    end repeat
  end repeat
end tell
return out
{ISO_HANDLERS}"""
    try:
        rows = records(applescript.run(script, timeout=_QUERY_TIMEOUT, app="Reminders"))
    except AppleScriptError as exc:
        raise _no_list(exc, name) from exc
    now = _iso(_now())
    items = []
    for r in rows:
        if len(r) != 6:
            continue
        item = {"id": r[0], "list": r[1], "title": r[2]}
        if r[3]:
            item["due"] = r[3]
            # A reminder with a date and no time is due at that day's midnight: it is late
            # once the day is over, not all through the day it is due.
            day_only = r[3].endswith("T00:00")
            if (r[3][:10] < now[:10]) if day_only else (r[3] < now):
                item["overdue"] = True
        if r[4]:
            item["notes"] = r[4]
        if r[5] == "true":
            item["flagged"] = True
        items.append(item)
    if not items:
        return f"No open reminders in {name}." if name else "No open reminders."
    items.sort(key=lambda i: (0, i["due"]) if "due" in i else (1, ""))
    return items


@register_tool("side_effecting")
def create_reminder(title: str, due: str = "", list: str = "", notes: str = ""):
    """Add a reminder to Apple Reminders: use this for "remind me to …" and for to-dos. `due` is
    optional — ISO 8601, 'tomorrow at 09:00', 'in 2 hours' — and is when the reminder alerts.
    `list` names an existing Reminders list (empty = the default list). It cannot repeat and
    cannot trigger at a place: tell the user if they asked for either."""
    title = str(title or "").strip()
    if not title:
        raise ToolError("a reminder needs a non-empty title")
    name = str(list or "").strip()
    notes = str(notes or "").strip()
    props = [f"name:{quote(title)}"]
    at = None
    if str(due or "").strip():
        try:
            at = notify.parse_when(str(due), now=_now())
        except notify.NotifyError as exc:
            raise ToolError(str(exc)) from exc
        props.append(f"due date:{mkdate(at)}")
    if notes:
        props.append(f"body:{quote(notes)}")
    pick = f"set l to list {quote(name)}" if name else "set l to default list"
    script = f"""
tell application "Reminders"
  try
    {pick}
    set ln to name of l
  on error
    set names to ""
    repeat with x in (name of lists)
      set names to names & x & {AS_GS}
    end repeat
    return "nolist" & {AS_US} & names
  end try
  set r to make new reminder at end of l with properties {{{", ".join(props)}}}
  return "ok" & {AS_US} & (id of r) & {AS_US} & ln
end tell
{_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Reminders")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    parts = out.split(applescript.US)
    if parts[0] == "nolist":
        have = ", ".join(n for n in (parts[1] if len(parts) > 1 else "").split(GS) if n) or "none"
        what = f"no Reminders list named {name!r}" if name else "Reminders has no default list on this Mac"
        raise ToolError(f"{what}; pass list= one of: {have}")
    if len(parts) != 3 or parts[0] != "ok":
        raise ToolError(f"Reminders did not confirm the new reminder: {out or 'no reply'}")
    result = {"id": parts[1], "list": parts[2], "title": title}
    if at is not None:
        result["due"] = _iso(at)
    return result


@register_tool("side_effecting")
def complete_reminder(reminder: str):
    """Mark a reminder in Apple Reminders as completed. `reminder` is an id from list_reminders
    (preferred) or the reminder's exact title."""
    ref = str(reminder or "").strip()
    if not ref:
        raise ToolError("complete_reminder needs a reminder id or title")
    if ref.startswith(_ID_PREFIX):
        locate = f"set r to reminder id {quote(ref)}"
    else:
        locate = f"set r to first reminder whose name is {quote(ref)} and completed is false"
    script = f"""
tell application "Reminders"
  {locate}
  set completed of r to true
  return name of r
end tell"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Reminders")
    except AppleScriptError as exc:
        if "-1728" in str(exc) or "-1719" in str(exc):
            raise ToolError(f"no open reminder matches {ref!r} (list_reminders gives the ids)") from exc
        raise ToolError(str(exc)) from exc
    return {"completed": out}
