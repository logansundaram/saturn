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
                  Adding Linux or Windows is one new module plus one branch here.
  parse_when()  — the small deterministic time grammar the tool accepts (ISO 8601, relative
                  offsets, today/tomorrow at HH:MM, a bare clock time). Refuses the past.

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
}
_RELATIVE = re.compile(r"^(?:in\s+|\+)?(\d+)\s*([a-z]+)$")
_CLOCK = re.compile(
    r"^(?:(today|tomorrow)\s*)?(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$"
)


def parse_when(text: str, now: datetime | None = None) -> datetime:
    """Resolve a `when` string to an aware local datetime strictly after `now`.

    Accepted: ISO 8601 (`2026-09-06T09:00`, offset kept when given, else local), a relative
    offset (`in 20 minutes`, `in 2 hours`, `+3d`, `20m`), `today/tomorrow [at] HH:MM[am|pm]`,
    and a bare clock time (`16:00`, `3pm`) meaning its NEXT occurrence. Anything in the past
    raises NotifyError with a hint to call current_time first."""
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
        when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if day == "tomorrow" or (day is None and when <= now):
            when += timedelta(days=1)
    else:
        raise NotifyError(
            f"could not understand the time {raw!r}; use ISO 8601 (2026-09-06T09:00), "
            "'in 20 minutes', 'tomorrow at 09:00', or '16:00'"
        )

    if when <= now:
        raise NotifyError(
            f"{when.isoformat(timespec='minutes')} is in the past (now is "
            f"{now.isoformat(timespec='minutes')}); call current_time and pick a future time"
        )
    return when
