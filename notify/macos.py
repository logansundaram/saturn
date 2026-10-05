"""
macOS backend — one launchd LaunchAgent per notification, displayed by osascript.

Why launchd: it is the OS's own scheduler. A LaunchAgent with `StartCalendarInterval` fires at
the wall-clock minute whether or not Saturn is running, survives a reboot, and runs missed
jobs on wake from sleep. No daemon of our own, no dependency: `/usr/bin/osascript -e 'display
notification …'` is on every Mac (the alert shows under "Script Editor" until the user allows
it once in System Settings → Notifications).

One plist per notification at `~/Library/LaunchAgents/com.saturn.notify.<id>.plist`. Its
program is a `/bin/sh -c` line that (0) exits unless the scheduled moment has arrived, (1) shows
the notification, (2) deletes its own plist, and (3) boots its own label out of launchd — so a
one-shot can never re-fire on the same calendar date next year.

`StartCalendarInterval` has no year, which is what step 0 is for: a reminder more than twelve
months out is started by launchd on THIS year's date, finds its moment has not come, and stays
loaded for the next one. `RunAtLoad` covers the opposite miss — a job whose minute passed while
the Mac was off is started again at the next login, finds itself due, and shows then instead of
a year later. The listing metadata (`SaturnNotification`) rides inside the plist, so the
plists on disk ARE the pending set: no second registry to drift.

`_run`, `agents_dir`, and `_uid` are the three seams the tests capture; nothing else touches
the OS.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from notify import Notification, NotifyError
from tools.applescript import quote

LABEL_PREFIX = "com.saturn.notify."


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _uid() -> int:
    return os.getuid()


def _run(argv: list[str]) -> str:
    """Run one launchctl/osascript command and return its stdout; a non-zero exit becomes a
    NotifyError carrying the tool's own stderr (launchctl's messages are the useful
    diagnostic)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NotifyError(f"{argv[0]} failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise NotifyError(f"{' '.join(argv[:2])} failed: {detail}")
    return proc.stdout or ""


def _flat(text: str) -> str:
    """One line: a notification shows a single line, so whitespace runs (newlines too) collapse."""
    return quote(" ".join(str(text or "").split()))


def _display_script(title: str, body: str) -> str:
    return f"display notification {_flat(body)} with title {_flat(title)}"


def _guard(when: datetime) -> str:
    """The shell test that holds a job until its real date: exit quietly while the clock is
    before the scheduled MINUTE (launchd starts the job at the minute's first second, which is
    before a `when` that carries seconds)."""
    due = int(when.replace(second=0, microsecond=0).timestamp())
    return f'[ "$(date +%s)" -ge {due} ] || exit 0'


def _label(id: str) -> str:
    return LABEL_PREFIX + id


def _domain() -> str:
    return f"gui/{_uid()}"


class LaunchdBackend:
    name = "macos-launchd"

    def _path(self, id: str) -> Path:
        return agents_dir() / f"{_label(id)}.plist"

    def schedule(self, n: Notification) -> None:
        path = self._path(n.id)
        label = _label(n.id)
        when = n.when.astimezone()      # launchd calendar intervals are local wall-clock
        # Every user-controlled string is a single shell word (shlex.quote), so `$(…)`, backticks
        # and quotes in a title are inert; the AppleScript escaping happens inside that word.
        script = " ; ".join([
            _guard(when),
            f"/usr/bin/osascript -e {shlex.quote(_display_script(n.title, n.body))}",
            f"rm -f {shlex.quote(str(path))}",
            f"launchctl bootout {_domain()}/{label}",
        ])
        plist = {
            "Label": label,
            "ProgramArguments": ["/bin/sh", "-c", script],
            "StartCalendarInterval": {
                "Month": when.month, "Day": when.day, "Hour": when.hour, "Minute": when.minute,
            },
            "RunAtLoad": True,
            "SaturnNotification": {
                "id": n.id,
                "when": when.isoformat(timespec="minutes"),
                "title": n.title,
                "body": n.body or "",
            },
        }
        agents_dir().mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            plistlib.dump(plist, fh)
        try:
            _run(["launchctl", "bootstrap", _domain(), str(path)])
        except NotifyError:
            path.unlink(missing_ok=True)      # never leave a plist launchd doesn't know about
            raise

    def cancel(self, id: str) -> bool:
        path = self._path(id)
        if not path.exists():
            return False
        try:
            _run(["launchctl", "bootout", f"{_domain()}/{_label(id)}"])
        except NotifyError:
            pass      # already fired / never loaded (reboot): the file is what's left to remove
        path.unlink(missing_ok=True)
        return True

    def pending(self) -> list[Notification]:
        out: list[Notification] = []
        d = agents_dir()
        if not d.exists():
            return out
        for path in d.glob(f"{LABEL_PREFIX}*.plist"):
            try:
                with open(path, "rb") as fh:
                    meta = plistlib.load(fh).get("SaturnNotification") or {}
                out.append(Notification(
                    id=str(meta["id"]),
                    when=datetime.fromisoformat(meta["when"]),
                    title=str(meta.get("title", "")),
                    body=str(meta.get("body", "")),
                ))
            except Exception:
                continue      # a hand-edited or damaged plist is not ours to interpret
        out.sort(key=lambda n: n.when)
        return out

    def fire_now(self, title: str, body: str) -> None:
        _run(["osascript", "-e", _display_script(title, body)])
