"""
/notify — the human's view of scheduled notifications: list what is pending, cancel one, or
fire a test alert to check the OS permission. The agent side is `tools/notify.py`
(schedule_notification); both sit on the same `notify/` backend seam, so the list here IS what
the OS will fire.
"""

from __future__ import annotations

import notify
from commands._framework import command, _print


@command(
    "notify",
    "Scheduled desktop notifications: what is pending, cancel one, or send a test alert.",
    usage="/notify [cancel <id> | test [message] | icon [start|stop]]",
    details="""
Notifications are one-shot reminders the agent scheduled with the schedule_notification tool
(or that you ask for directly). They are handed to the operating system's own scheduler —
launchd on macOS — so they fire at the set time whether or not Saturn is still running, and
survive a reboot.

  /notify                 list pending notifications: id, local time, title
  /notify cancel <id>     remove one (the id is in the agent's answer and the trace)
  /notify test [message]  show a notification right now — use it once to grant the
                          permission macOS asks for on the first alert (it appears under
                          "Script Editor", the built-in notifier Saturn calls)
  /notify icon            is the menu bar icon up? It is a separate small process (a login
                          LaunchAgent) that outlives this terminal: it lists what is pending
                          and its "Quit Saturn…" stops the agent, cancels every notification,
                          and removes itself. Starts with each launch while `notify.menubar`
                          is on in config.yaml.
    /notify icon start|stop  bring it back / take it down without touching notifications

Nothing here is egress: the schedule lives in ~/Library/LaunchAgents and the alert is shown
by the OS. macOS only for now.
""",
)
def _notify(ctx, args):
    if not args:
        return _list()
    sub = args[0].lower()
    if sub in ("cancel", "rm", "remove", "delete"):
        return _cancel(args[1:])
    if sub == "test":
        return _test(" ".join(args[1:]))
    if sub in ("icon", "menubar"):
        return _icon(args[1:])
    _print(f"  unknown subcommand: {sub} — usage: /notify [cancel <id> | test [message] | icon [start|stop]]")


def _list():
    from tui import ui

    try:
        pending = notify.backend().pending()
    except notify.NotifyError as exc:
        ui.warn(str(exc))
        return
    ui.section("notifications", f"{len(pending)} pending")
    if not pending:
        ui.note("no pending notifications")
        return
    ui.table([
        (n.id, n.when.astimezone().strftime("%Y-%m-%d %H:%M"), n.title, n.body)
        for n in pending
    ], styles=["accent", None, None, "dim"])


def _cancel(args):
    if not args:
        _print("  usage: /notify cancel <id>")
        return
    from tui import ui

    id = args[0]
    try:
        removed = notify.backend().cancel(id)
    except notify.NotifyError as exc:
        ui.warn(str(exc))
        return
    if removed:
        _print(f"  cancelled {id}")
    else:
        _print(f"  no pending notification with id {id} (see /notify)")


def _test(message: str):
    from tui import ui

    try:
        notify.backend().fire_now("Saturn", message or "notifications are working")
    except notify.NotifyError as exc:
        ui.warn(str(exc))
        return
    _print("  sent — if nothing appeared, allow notifications for Script Editor in System Settings")


def _icon(args):
    from notify import menubar

    verb = (args[0].lower() if args else "")
    if verb in ("start", "on"):
        _print(f"  menu bar icon: {menubar.ensure_running()}")
    elif verb in ("stop", "off"):
        _print("  menu bar icon: stopped" if menubar.stop() else "  menu bar icon: not running")
    elif verb:
        _print("  usage: /notify icon [start|stop]")
    else:
        up = menubar.is_running()
        knob = "on" if menubar.enabled() else "off"
        _print(f"  menu bar icon: {'running' if up else 'not running'} "
               f"(launch autostart {knob}: notify.menubar in config.yaml)")
