# Native scheduled notifications

**Date:** 2026-09-05
**Status:** approved design (chat), implemented in the same session

## Goal

Let the agent schedule a one-shot desktop notification ("at 3pm tomorrow, remind me to call
the dentist") that fires through the operating system's own notifier, survives Saturn being
quit, and is visible and cancellable by the human. macOS first, with the platform seam shaped so
Linux and Windows are one file each later.

This reverses PLAN.md's verification-pass item 4 ("no scheduling, by decision") for exactly one
shape: a one-shot notification handed to the OS scheduler. Saturn itself still never runs in
the background; launchd does the waiting.

## Decisions

| Question | Decision | Rationale |
|---|---|---|
| Fires after Saturn quits? | Yes. One LaunchAgent plist per notification, `StartCalendarInterval` | Native, no daemon of our own, survives quit and reboot; launchd runs missed jobs on wake |
| Display mechanism | `/usr/bin/osascript -e 'display notification …'` | Built in, no dependency. Shows under "Script Editor" until the user allows it once |
| Recurrence | One-shot only | Smallest surface; the plist self-deletes after firing so it never re-fires next year |
| Risk tier | `side_effecting` | It changes OS state, so it faces the gate; it is not egress (nothing leaves the machine) |
| Agent cancel | No. Humans cancel via `/notify cancel <id>` | YAGNI; the id is in the trace and the answer |
| Time grammar | ISO 8601 local, `in N minutes/hours/days`, `today/tomorrow [at] HH:MM`, bare `HH:MM` (next occurrence) | Small planners pass relative phrases; a deterministic parser is cheaper than a retry |
| Past times | Refused, with a hint to call `current_time` | Never silently schedule something that can't fire |

## Components

- `notify/__init__.py` — `Notification` dataclass (`id`, `when`, `title`, `body`), `NotifyError`,
  the `Backend` protocol (`schedule`, `cancel`, `pending`, `fire_now`), `backend()` chosen by
  `sys.platform`, `new_id()`, and `parse_when(text, now)`. Non-macOS resolves to `Unsupported`,
  whose every method raises `NotifyError` with an honest platform message.
- `notify/macos.py` — the LaunchAgent backend. Plist at
  `~/Library/LaunchAgents/com.saturn.notify.<id>.plist`, `ProgramArguments` is
  `/bin/sh -c "<osascript …>; rm -f <plist>; launchctl bootout gui/<uid>/<label>"`. Metadata for
  listing lives in a `SaturnNotification` dict inside the plist (`plistlib`). Load with
  `launchctl bootstrap gui/<uid> <plist>`; cancel with `bootout` plus unlink.
- `tools/notify.py` — `schedule_notification(when, title, body="")`, `side_effecting`. Returns the
  resolved local time and the id, or an `Error: …` string.
- `commands/notify.py` — `/notify` (list), `/notify cancel <id>`, `/notify test [message]`.

## Testing

`tests/test_notify.py`, fully offline: `launchctl` and `osascript` are captured through a
monkeypatched runner and the agents directory is a tmp path. Pins: plist content and the
self-cleanup script, AppleScript escaping, every `parse_when` form, past-time refusal, cancel,
pending order, the tool's tier and error path, the command's three verbs, and the unsupported
platform message.

## Out of scope

Recurrence, click actions, an in-process fallback, agent-side cancel, Linux/Windows backends.
