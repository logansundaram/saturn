"""
The menu bar item — the Cocoa-free half (lifecycle, pidfile, menu model, quit).

A small separate process (`notify/menubar_app.py`, AppKit) puts a ringed-planet icon in the
macOS menu bar. It is registered as a login LaunchAgent (`com.saturn.menubar`, RunAtLoad), so
it outlives the terminal that launched Saturn and returns after a reboot — the same lifetime
as the notifications it watches over. Closing the terminal stops the agent and nothing else;
"Quit Saturn…" in the menu is the one full stop: it ends a running agent, cancels every
pending notification, and unregisters the icon (`quit_all`). The next `saturn` launch brings
the icon back (`ensure_running`, called from the REPL when `notify.menubar` is on).

Everything here is plain Python — plists via the notification backend's launchctl seam, a
pidfile for agent liveness, and `menu_model()` returning the rows the renderer draws — so it
is fully tested offline; only `menubar_app.py` touches AppKit. This module imports config (for
the pidfile location and the knob) and `notify`; nothing here is egress.
"""

from __future__ import annotations

import os
import plistlib
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import diag
import notify
from notify import NotifyError, macos

LABEL = "com.saturn.menubar"
APP_MODULE = "notify.menubar_app"
# The directory holding the `notify` package: the repo root in clone mode, site-packages for a
# wheel. launchd starts jobs at `/`, so the job's WorkingDirectory is pinned here — `python -m`
# puts the cwd first on sys.path, which is what makes the clone-mode import work.
PROJECT_ROOT = Path(notify.__file__).resolve().parent.parent


# ── config + paths ───────────────────────────────────────────────────────────────────────────

def enabled() -> bool:
    """`notify.menubar` in config.yaml (default on): whether an interactive launch starts the
    icon. `/notify icon start` works either way."""
    from config import get_config
    return bool(get_config().get("notify.menubar", True))


def _database_dir() -> Path:
    from config import get_config
    return get_config().path("database")


def plist_path() -> Path:
    return macos.agents_dir() / f"{LABEL}.plist"


def pid_path() -> Path:
    return _database_dir() / "agent.pid"


# ── the login LaunchAgent ────────────────────────────────────────────────────────────────────

def agent_plist(python: str) -> dict:
    """The LaunchAgent definition. The venv's Python is pinned (that is where pyobjc lives) and
    so is the project root (see PROJECT_ROOT). KeepAlive is OFF on purpose: a job that dies at
    import (pyobjc missing, a moved checkout) would otherwise be relaunched every 10 s forever;
    a crashed icon simply comes back on the next `saturn` launch."""
    return {
        "Label": LABEL,
        "ProgramArguments": [python, "-m", APP_MODULE],
        "WorkingDirectory": str(PROJECT_ROOT),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "StandardErrorPath": str(_database_dir() / "menubar.err"),
    }


def _print_job() -> str | None:
    """`launchctl print` for our label: its text when launchd knows the job, else None."""
    try:
        return macos._run(["launchctl", "print", f"{macos._domain()}/{LABEL}"]) or ""
    except NotifyError:
        return None


def is_loaded() -> bool:
    """launchd knows the job (it may still have exited — see is_running)."""
    return _print_job() is not None


def is_running() -> bool:
    """The icon process is alive right now. A loaded job whose process died at import stays
    registered with launchd, so `is_loaded` alone would call a dead icon "up"."""
    out = _print_job()
    return out is not None and "pid = " in out


def _current_python() -> str | None:
    path = plist_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return str(plistlib.load(fh)["ProgramArguments"][0])
    except Exception:
        return None


def ensure_running(python: str | None = None) -> str:
    """Register + start the icon if it is not up. Returns a one-word status for the caller to
    log or print: started · already running · restarted (the pinned Python moved) · unsupported
    · failed: <why>. Never raises — a launch must not depend on the menu bar."""
    if isinstance(notify.backend(), notify.Unsupported):
        return "unsupported"
    python = python or sys.executable
    path = plist_path()
    try:
        pinned = _current_python()
        out = _print_job()  # one launchctl round-trip on the startup path: running implies loaded
        loaded = out is not None
        if pinned == python and loaded and "pid = " in out:
            return "already running"

        if loaded:
            try:
                macos._run(["launchctl", "bootout", f"{macos._domain()}/{LABEL}"])
            except NotifyError:
                pass
        macos.agents_dir().mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            plistlib.dump(agent_plist(python), fh)
        macos._run(["launchctl", "bootstrap", macos._domain(), str(path)])
    except NotifyError as exc:
        diag.log(f"menubar: {exc}")
        return f"failed: {exc}"
    except OSError as exc:
        diag.log(f"menubar: {exc}")
        return f"failed: {exc}"
    return "restarted" if loaded else "started"


def stop() -> bool:
    """Unload the icon and remove its login registration. True when there was one."""
    path = plist_path()
    existed = path.exists()
    if is_loaded():
        existed = True
        try:
            macos._run(["launchctl", "bootout", f"{macos._domain()}/{LABEL}"])
        except NotifyError:
            pass
    path.unlink(missing_ok=True)
    return existed


# ── agent liveness (pidfile) ─────────────────────────────────────────────────────────────────

def write_pid(pid: int | None = None) -> None:
    try:
        pid_path().parent.mkdir(parents=True, exist_ok=True)
        pid_path().write_text(str(pid or os.getpid()))
    except OSError as exc:
        diag.log(f"menubar: pidfile write failed: {exc}")


def clear_pid(owner: int | None = None) -> None:
    """Drop the pidfile. With `owner`, only when the file still records that pid: two REPLs
    overwrite each other's entry, and the first to exit must not unlink the live session's
    (the icon then showed no agent and its Quit could stop nothing — review 2026-09-06)."""
    try:
        if owner is not None:
            try:
                if pid_path().read_text().strip() != str(owner):
                    return
            except OSError:
                return
        pid_path().unlink(missing_ok=True)
    except OSError:
        pass



def agent_pid() -> int | None:
    """The running agent's pid, or None. A stale file (dead pid) is removed on sight."""
    try:
        pid = int(pid_path().read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        clear_pid()
        return None
    except PermissionError:
        return pid      # alive, just not ours to signal
    return pid


# ── the menu model ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Row:
    kind: str       # status | sep | pending | empty | test | quit
    label: str = ""
    id: str = ""    # the notification id for `pending` rows


def _when_label(when: datetime) -> str:
    local = when.astimezone()
    today = datetime.now().astimezone().date()
    day = "today" if local.date() == today else local.strftime("%a %d %b")
    return f"{day} {local.strftime('%H:%M')}"


def menu_model() -> list[Row]:
    """The rows the renderer draws, top to bottom. Pure data so the menu is testable without
    AppKit; the app rebuilds it every time the menu opens."""
    pid = agent_pid()
    rows = [Row("status", f"Saturn agent running (pid {pid})" if pid else "Saturn agent not running"),
            Row("sep")]
    try:
        pending = notify.backend().pending()
    except NotifyError as exc:
        pending, problem = [], str(exc)
    else:
        problem = ""
    if problem:
        rows.append(Row("empty", problem))
    elif not pending:
        rows.append(Row("empty", "no pending notifications"))
    for n in pending:
        rows.append(Row("pending", f"{_when_label(n.when)} · {n.title}", n.id))
    rows += [Row("sep"), Row("test", "Send test notification"), Row("quit", "Quit Saturn…")]
    return rows


# ── the full stop ────────────────────────────────────────────────────────────────────────────

def quit_summary(n_pending: int, agent_running: bool) -> str:
    """The sentence the confirm dialog shows — exactly what Quit will do."""
    parts = []
    if agent_running:
        parts.append("stops the running Saturn agent")
    if n_pending:
        parts.append(f"cancels {n_pending} pending notification{'s' if n_pending != 1 else ''}")
    parts.append("removes the menu bar icon (it returns the next time you launch saturn)")
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"This {body}."


def quit_all() -> dict:
    """Everything "Quit Saturn" means: cancel every pending notification, stop a running agent,
    drop the pidfile, and unregister the icon — its own bootout LAST, because that ends the
    process calling this."""
    cancelled = 0
    try:
        be = notify.backend()
        for n in be.pending():
            if be.cancel(n.id):
                cancelled += 1
    except NotifyError as exc:
        diag.log(f"menubar quit: {exc}")
    stopped = False
    pid = agent_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except OSError as exc:
            diag.log(f"menubar quit: could not stop agent {pid}: {exc}")
    clear_pid()
    stop()
    return {"cancelled": cancelled, "agent_stopped": stopped}
