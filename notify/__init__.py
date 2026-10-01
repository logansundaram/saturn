"""
Native scheduled notifications — the platform seam.

A notification is a one-shot desktop alert handed to the OPERATING SYSTEM's own scheduler, so it
fires whether or not Saturn is still running. Saturn itself never runs in the background
(PLAN.md, verification pass #4): the OS does the waiting, and Saturn only writes the schedule
entry, lists it, and removes it.

  Notification  — the plain record: id, when (an aware datetime), title, body.
  Backend       — what a platform must provide: schedule / cancel / pending / fire_now.
  backend()     — picks the backend for `sys.platform`. Only macOS is implemented
                  (`notify/macos.py`, launchd + osascript); every other platform resolves to
                  `Unsupported`, whose methods raise an honest NotifyError instead of crashing.
                  Adding Linux is one new module plus one branch here.
  parse_when()  — the small deterministic time grammar the tool accepts (ISO 8601, relative
                  offsets, today/tomorrow/a weekday at HH:MM, a bare clock time; a bare day
                  only for whole-day callers). Refuses the past.

Nothing here is egress: every byte stays on this machine (tests/test_no_new_egress.py needs no
allowlist entry). This package imports nothing project-side, so tools/ and commands/ can import
it freely.
"""

from __future__ import annotations

import re
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol


class NotifyError(Exception):
    """A notification could not be scheduled, listed, or cancelled. The message is written for
    the model/user to read verbatim."""


@dataclass
class Notification:
    id: str
    when: datetime          # timezone-aware
    title: str
    body: str = ""


class Backend(Protocol):
    name: str

    def schedule(self, n: Notification) -> None: ...
    def cancel(self, id: str) -> bool: ...
    def pending(self) -> list[Notification]: ...
    def fire_now(self, title: str, body: str) -> None: ...


class Unsupported:
    """The honest no-op backend for platforms without an implementation yet."""

    name = "unsupported"

    def __init__(self, platform: str):
        self.platform = platform

    def _refuse(self):
        raise NotifyError(
            f"scheduled notifications are not supported on {self.platform} yet (macOS only)"
        )

    def schedule(self, n: Notification) -> None:
        self._refuse()

    def cancel(self, id: str) -> bool:
        self._refuse()

    def pending(self) -> list[Notification]:
        self._refuse()

    def fire_now(self, title: str, body: str) -> None:
        self._refuse()


def backend() -> Backend:
    """The backend for this platform. Tests and callers monkeypatch this one function."""
    if sys.platform == "darwin":
        from notify.macos import LaunchdBackend
        return LaunchdBackend()
    return Unsupported(sys.platform)


def new_id() -> str:
    """A short random id — the handle shown in the trace, the answer, and `/notify cancel`."""
    return secrets.token_hex(4)


# ── time grammar ─────────────────────────────────────────────────────────────────────────────

_UNITS = {
    "m": "minutes", "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
    "d": "days", "day": "days", "days": "days",
    "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
}
_RELATIVE = re.compile(r"^(?:in\s+|\+)?(\d+)\s*([a-z]+)$")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAY = r"today|tomorrow|yesterday|(?:next\s+)?(?:" + "|".join(_WEEKDAYS) + ")"
_CLOCK = re.compile(
    rf"^(?:({_DAY})\s*)?(?:at\s+)?(\d{{1,2}})(?::(\d{{2}}))?\s*(am|pm)?$"
)
_BARE_DAY = re.compile(rf"^({_DAY})$")


def _day_offset(day: "str | None", now: datetime) -> int:
    """Days from today to the named day. A weekday is its NEXT occurrence — said on that very
    weekday it means a week from now, never today."""
    if day in (None, "today"):
        return 0
    if day in ("tomorrow", "yesterday"):
        return 1 if day == "tomorrow" else -1
    return (_WEEKDAYS.index(day.split()[-1]) - now.weekday() - 1) % 7 + 1


def parse_when(text: str, now: datetime | None = None, *, allow_past: bool = False,
               whole_day: bool = False) -> datetime:
    """Resolve a `when` string to an aware local datetime strictly after `now` (or any time at
    all with `allow_past`, for range queries that look backward).

    Accepted: ISO 8601 (`2026-09-06T09:00`, offset kept when given, else local), a relative
    offset (`in 20 minutes`, `in 2 hours`, `+3d`, `in 1 week`, `20m`),
    `today/tomorrow/[next] monday [at] HH:MM[am|pm]`, and a bare clock time (`16:00`, `3pm`)
    meaning its NEXT occurrence. A bare day (`today`, `tomorrow`, `next monday`) has no time to
    fire at: it is accepted only with `whole_day` (a calendar window), as that day's first
    minute. Anything in the past raises NotifyError with a hint to call current_time first,
    unless `allow_past`."""
    now = (now or datetime.now()).astimezone()
    raw = (text or "").strip()
    s = raw.lower()
    if not s:
        raise NotifyError("could not understand an empty time")

    when: datetime | None = None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        pass
    if when is not None:
        if when.tzinfo is None:
            when = when.astimezone()      # naive = local wall clock; an explicit offset is kept
    elif (m := _RELATIVE.match(s)) and m.group(2) in _UNITS:
        when = now + timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})
    elif (m := _CLOCK.match(s)) and (m.group(3) or m.group(4) or m.group(1)):
        day, hh, mm, ampm = m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4)
        if ampm == "pm" and hh < 12:
            hh += 12
        elif ampm == "am" and hh == 12:
            hh = 0
        if hh > 23 or mm > 59:
            raise NotifyError(f"could not understand the time {raw!r}")
        when = (now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                + timedelta(days=_day_offset(day, now)))
        if day is None and when <= now:
            when += timedelta(days=1)
    elif (m := _BARE_DAY.match(s)):
        if not whole_day:
            raise NotifyError(f"{raw!r} needs a clock time, e.g. '{raw} at 09:00'")
        when = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(days=_day_offset(m.group(1), now)))
    else:
        raise NotifyError(
            f"could not understand the time {raw!r}; use ISO 8601 (2026-09-06T09:00), "
            "'in 20 minutes', 'tomorrow at 09:00', or '16:00'"
        )

    if when <= now and not allow_past:
        raise NotifyError(
            f"{when.isoformat(timespec='minutes')} is in the past (now is "
            f"{now.isoformat(timespec='minutes')}); call current_time and pick a future time"
        )
    return when
