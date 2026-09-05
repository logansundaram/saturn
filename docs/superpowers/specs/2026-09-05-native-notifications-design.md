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

## Addendum (same day): the menu bar item

Approved in chat after the tool shipped. A ringed-planet icon in the macOS menu bar, derived
from the splash motif (drawn as an 18-pt template image, no asset file), that outlives the
terminal the way the notifications do.

| Question | Decision | Rationale |
|---|---|---|
| Toolkit | `pyobjc-framework-Cocoa`, a `sys_platform == 'darwin'` dependency | Installs only on Macs; NSStatusItem directly, no rumps |
| Lifecycle | Started by each interactive launch (`ensure_running`), registered as a login LaunchAgent (RunAtLoad, KeepAlive off) | Outlives the terminal and returns after reboot like the notifications; a crash never loops |
| What Quit means | Confirm sheet, then: stop a running agent (SIGTERM via `database/agent.pid`), cancel every pending notification, unregister the icon | "Fully quit" — the one full stop; closing the terminal does none of it |
| Menu | agent status · pending list (click cancels) · test alert · Quit Saturn… | Rebuilt from the plists on every open; nothing new stored |
| Off switch | `notify.menubar: false`; `/notify icon start|stop` on demand | The icon is a convenience, never a dependency of launch |

`notify/menubar.py` is the tested, Cocoa-free half (`tests/test_menubar.py`); `menubar_app.py`
only draws and was verified live (launchd registration, icon rendering, idempotent restart).
The LaunchAgent pins `WorkingDirectory` to the project root because launchd starts jobs at `/`
and `python -m` needs the package importable in a clone-mode install.

## Out of scope

Recurrence, click actions on the notification itself, an in-process fallback, agent-side
cancel, Linux/Windows backends, a Dock tile, tracking more than one concurrent agent process.
