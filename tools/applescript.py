"""
AppleScript runner — the one seam every native macOS app tool (Notes, Calendar) goes through.

Why AppleScript and not a framework binding: a terminal-launched Python only gets Calendar /
Contacts access through EventKit if the *terminal app* itself carries Apple's usage-description
keys (iTerm does, Terminal.app and VS Code don't), and the request fails silently rather than
prompting. Apple-event automation is uniform: one "Terminal wants to control Notes" dialog per
target app on first use, from any terminal, then it works. Measured 2026-09-06: Notes queries
sub-second warm, a two-week Calendar window across eight calendars ~6.5s — slow but bounded, and
it is one planner step.

  run(script)   — `osascript -e script`, stdout stripped; every failure is an AppleScriptError
                  whose message is written for the model to relay (not macOS, Automation denied,
                  app not running, timeout, or the raw osascript error).
  quote(text)   — an AppleScript string literal; newlines become `& linefeed &` so a note body
                  survives, quotes and backslashes are escaped.
  records(out)  — parse the RS/US-delimited output the tool scripts emit, so user text with
                  commas, pipes or newlines can never split a field.

Nothing here is egress: Apple events stay on this machine (tests/test_no_new_egress.py needs no
allowlist entry). This module imports nothing project-side, so the tool modules import it freely.
"""

from __future__ import annotations

import re
import subprocess
import sys

RS = "\x1e"   # record separator — between records
US = "\x1f"   # unit separator — between fields of one record

# `RS`/`US` as AppleScript expressions, for building output lines inside a script.
AS_RS = "(ASCII character 30)"
AS_US = "(ASCII character 31)"


# Handlers appended to a script that emits dates: an ISO-8601 minute-precision local timestamp
# from an AppleScript date (`date as string` is locale-dependent, so it is never used).
ISO_HANDLERS = """
on iso(d)
  return (year of d as string) & "-" & my pad((month of d) as integer) & "-" & my pad(day of d) ¬
    & "T" & my pad((time of d) div 3600) & ":" & my pad(((time of d) mod 3600) div 60)
end iso
on pad(n)
  if n < 10 then return "0" & (n as string)
  return n as string
end pad
"""


class AppleScriptError(Exception):
    """A script could not run or failed. The message is written for the model/user to read
    verbatim (it always starts with a lowercase clause the tool prefixes with 'Error: ')."""


def _platform() -> str:
    return sys.platform


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


_DENIED = re.compile(r"Not authorized to send Apple events to (\w+)")
_NOT_RUNNING = re.compile(r"(\w+) got an error: Application isn.t running")


def run(script: str, timeout: float = 30.0, *, app: str | None = None) -> str:
    """Run one AppleScript and return its stripped stdout; raises AppleScriptError otherwise.

    `app` names the application the script targets: it is opened hidden and in the background
    first (`open -gja`). osascript does NOT reliably launch a closed app itself — Calendar
    answers -600 "Application isn't running" even to `launch` (probed 2026-09-06) — and a
    LaunchServices open is ~50ms when the app is already up."""
    plat = _platform()
    if plat != "darwin":
        raise AppleScriptError(f"this tool is only available on macOS (this is {plat})")
    try:
        if app:
            _run(["open", "-gja", app], timeout)
        proc = _run(["osascript", "-e", script], timeout)
    except subprocess.TimeoutExpired as exc:
        # Killing osascript does NOT cancel the Apple event: the app keeps executing it on its
        # main thread and later events queue behind it. Callers must not retry on this error.
        raise AppleScriptError(
            f"osascript timed out after {int(timeout)}s; {app or 'the app'} may still be busy "
            f"with the request — wait before asking again") from exc
    except OSError as exc:
        raise AppleScriptError(f"osascript could not start: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        if m := _DENIED.search(detail):
            raise AppleScriptError(
                f"macOS denied automation access to {m.group(1)}; allow this terminal under "
                f"System Settings > Privacy & Security > Automation and retry"
            )
        if m := _NOT_RUNNING.search(detail):
            raise AppleScriptError(f"{m.group(1)} is not running and could not be launched")
        raise AppleScriptError(detail)
    # Only line endings: str.strip() would also eat RS/US (they count as whitespace), and a
    # record whose last field is empty ends in exactly that.
    return (proc.stdout or "").strip("\r\n")


def quote(text) -> str:
    """An AppleScript double-quoted string literal for `text`, newlines preserved."""
    s = str(text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\r\n", "\n")
    return '"' + s.replace("\n", '" & linefeed & "') + '"'


def records(output: str) -> list[list[str]]:
    """Split RS/US-delimited script output into a list of field lists."""
    out = (output or "").strip("\r\n")
    if not out:
        return []
    return [rec.split(US) for rec in out.split(RS) if rec]
