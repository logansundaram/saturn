"""
The macOS menu bar item: `notify/menubar.py` — the Cocoa-free half. Plist shape
and Python pinning, idempotent ensure, stop, the agent pidfile, the menu model the AppKit
renderer draws, `quit_all`, and the config knob. `launchctl` is captured through the same
runner seam the notification backend uses; the LaunchAgents dir is a tmp path; the platform is
irrelevant (nothing here imports AppKit).
"""

import os
import plistlib
import signal
from datetime import datetime, timedelta

import pytest

import notify
from notify import Notification, NotifyError, macos, menubar

NOW = datetime(2026, 9, 5, 14, 30, 0).astimezone()


@pytest.fixture
def agents(tmp_path, monkeypatch, isolated_paths):
    calls: list[list[str]] = []
    loaded: set[str] = set()
    dead: set[str] = set()     # loaded but the process exited (KeepAlive is off)

    def run(argv):
        """A fake launchd: bootstrap loads a plist's label, bootout unloads it, print fails
        for an unknown label (the real exit code semantics is_loaded relies on)."""
        calls.append(list(argv))
        if argv[:2] == ["launchctl", "bootstrap"]:
            loaded.add(argv[3].rsplit("/", 1)[-1].removesuffix(".plist"))
        elif argv[:2] == ["launchctl", "bootout"]:
            label = argv[2].rsplit("/", 1)[-1]
            if label not in loaded:
                raise NotifyError("launchctl bootout failed: No such process")
            loaded.discard(label)
        elif argv[:2] == ["launchctl", "print"]:
            label = argv[2].rsplit("/", 1)[-1]
            if label not in loaded:
                raise NotifyError("launchctl print failed: Could not find service")
            return "state = not running\n" if label in dead else "state = running\n\tpid = 555\n"
        return ""

    monkeypatch.setattr(macos, "agents_dir", lambda: tmp_path / "LaunchAgents")
    monkeypatch.setattr(macos, "_run", run)
    monkeypatch.setattr(macos, "_uid", lambda: 501)
    monkeypatch.setattr(notify, "backend", lambda: macos.LaunchdBackend())
    run.dead = dead
    return tmp_path / "LaunchAgents", calls


def _n(id, minutes, title="x"):
    return Notification(id=id, when=NOW + timedelta(minutes=minutes), title=title)


# ── the login LaunchAgent ────────────────────────────────────────────────────────────────────

def test_agent_plist_pins_python_and_project_root_and_never_loops(agents, isolated_paths):
    """`python -m` needs the project root on sys.path in a clone-mode install (launchd starts
    jobs at `/`), and a job that fails at import must not be relaunched every 10 s forever."""
    plist = menubar.agent_plist("/venv/bin/python")
    assert plist["Label"] == "com.saturn.menubar"
    assert plist["ProgramArguments"] == ["/venv/bin/python", "-m", "notify.menubar_app"]
    assert plist["WorkingDirectory"] == str(menubar.PROJECT_ROOT)
    assert (menubar.PROJECT_ROOT / "notify" / "menubar_app.py").exists()
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is False
    assert plist["ProcessType"] == "Interactive"
    assert plist["StandardErrorPath"] == str(isolated_paths / "database" / "menubar.err")


def test_ensure_running_writes_and_bootstraps_once(agents):
    d, calls = agents
    assert menubar.ensure_running("/venv/bin/python") == "started"
    path = d / "com.saturn.menubar.plist"
    assert path.exists()
    assert calls[-1] == ["launchctl", "bootstrap", "gui/501", str(path)]
    calls.clear()
    assert menubar.ensure_running("/venv/bin/python") == "already running"
    assert not any(c[1] == "bootstrap" for c in calls)
    # one launchctl round-trip on the startup path, not two (running implies loaded)
    assert sum(1 for c in calls if c[1] == "print") == 1



def test_ensure_running_reloads_when_the_python_moved(agents):
    d, calls = agents
    menubar.ensure_running("/old/bin/python")
    calls.clear()
    assert menubar.ensure_running("/new/bin/python") == "restarted"
    with open(d / "com.saturn.menubar.plist", "rb") as fh:
        assert plistlib.load(fh)["ProgramArguments"][0] == "/new/bin/python"
    verbs = [c[1] for c in calls if c[0] == "launchctl"]
    assert verbs.index("bootout") < verbs.index("bootstrap")


def test_ensure_running_restarts_a_loaded_job_whose_process_died(agents):
    d, calls = agents
    menubar.ensure_running("/venv/bin/python")
    macos._run.dead.add("com.saturn.menubar")
    assert menubar.is_loaded() and not menubar.is_running()
    calls.clear()
    assert menubar.ensure_running("/venv/bin/python") == "restarted"
    verbs = [c[1] for c in calls if c[0] == "launchctl"]
    assert verbs.index("bootout") < verbs.index("bootstrap")


def test_ensure_running_never_raises(agents, monkeypatch):
    def boom(argv):
        raise NotifyError("launchctl bootstrap failed: Input/output error")

    monkeypatch.setattr(macos, "_run", boom)
    out = menubar.ensure_running("/venv/bin/python")
    assert out.startswith("failed") and "Input/output error" in out


def test_ensure_running_is_unsupported_off_macos(agents, monkeypatch):
    monkeypatch.setattr(notify, "backend", lambda: notify.Unsupported("linux"))
    assert menubar.ensure_running("/venv/bin/python") == "unsupported"


def test_stop_boots_out_and_removes_the_plist(agents):
    d, calls = agents
    menubar.ensure_running("/venv/bin/python")
    calls.clear()
    assert menubar.stop() is True
    assert not (d / "com.saturn.menubar.plist").exists()
    assert ["launchctl", "bootout", "gui/501/com.saturn.menubar"] in calls
    assert menubar.stop() is False
    assert menubar.is_loaded() is False


def test_enabled_reads_the_config_knob(isolated_paths, monkeypatch):
    from config import get_config
    cfg = get_config()
    notify_cfg = {k: v for k, v in cfg._data.get("notify", {}).items() if k != "menubar"}
    monkeypatch.setitem(cfg._data, "notify", notify_cfg)
    assert menubar.enabled() is False  # off unless asked for
    monkeypatch.setitem(notify_cfg, "menubar", True)
    assert menubar.enabled() is True


# ── the agent pidfile ────────────────────────────────────────────────────────────────────────

def test_pidfile_lifecycle(isolated_paths):
    assert menubar.agent_pid() is None
    menubar.write_pid(os.getpid())
    assert menubar.agent_pid() == os.getpid()
    menubar.clear_pid()
    assert menubar.agent_pid() is None
    assert not menubar.pid_path().exists()


def test_clear_pid_with_an_owner_leaves_another_sessions_pidfile(isolated_paths):
    # Review 2026-09-06: two REPLs — the first to exit unlinked the live session's pidfile, so
    # the icon showed no agent and its Quit could stop nothing. A session clears only its own.
    menubar.write_pid(4242)
    menubar.clear_pid(owner=os.getpid())
    assert menubar.pid_path().read_text() == "4242"
    menubar.clear_pid(owner=4242)
    assert not menubar.pid_path().exists()
    menubar.clear_pid(owner=4242)  # already gone: no error


def test_stale_pidfile_reads_as_not_running_and_is_cleared(isolated_paths, monkeypatch):

    menubar.write_pid(999999)

    def dead(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(menubar.os, "kill", dead)
    assert menubar.agent_pid() is None
    assert not menubar.pid_path().exists()


# ── the menu model ───────────────────────────────────────────────────────────────────────────

def test_menu_model_with_pending_and_running_agent(agents, isolated_paths):
    be = notify.backend()
    be.schedule(_n("later", 90, "Later"))
    be.schedule(_n("soon", 5, "Call the dentist"))
    menubar.write_pid(os.getpid())

    rows = menubar.menu_model()
    kinds = [r.kind for r in rows]
    assert kinds == ["status", "sep", "pending", "pending", "sep", "test", "quit"]
    assert rows[0].label == f"Saturn agent running (pid {os.getpid()})"
    assert rows[2].id == "soon" and "Call the dentist" in rows[2].label
    assert (NOW + timedelta(minutes=5)).strftime("%H:%M") in rows[2].label
    assert rows[-1].label == "Quit Saturn…"


def test_menu_model_empty_and_idle(agents, isolated_paths):
    rows = menubar.menu_model()
    assert [r.kind for r in rows] == ["status", "sep", "empty", "sep", "test", "quit"]
    assert rows[0].label == "Saturn agent not running"
    assert rows[2].label == "no pending notifications"


def test_menu_model_survives_a_backend_error(agents, isolated_paths, monkeypatch):
    monkeypatch.setattr(notify, "backend", lambda: notify.Unsupported("linux"))
    rows = menubar.menu_model()
    assert any(r.kind == "empty" and "not supported" in r.label for r in rows)


# ── quit_all ─────────────────────────────────────────────────────────────────────────────────

def test_quit_all_cancels_kills_and_unregisters_in_order(agents, isolated_paths, monkeypatch):
    d, calls = agents
    be = notify.backend()
    be.schedule(_n("a", 5))
    be.schedule(_n("b", 10))
    menubar.ensure_running("/venv/bin/python")
    menubar.write_pid(4242)
    killed: list = []
    monkeypatch.setattr(menubar.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    calls.clear()

    summary = menubar.quit_all()

    assert summary == {"cancelled": 2, "agent_stopped": True}
    assert be.pending() == []
    assert [k for k in killed if k[1] != 0] == [(4242, signal.SIGTERM)]   # 0 = the liveness probe
    assert not (d / "com.saturn.menubar.plist").exists()
    assert not menubar.pid_path().exists()
    # the icon's own bootout is LAST — it ends this very process
    assert calls[-1] == ["launchctl", "bootout", "gui/501/com.saturn.menubar"]


def test_quit_all_with_nothing_to_do(agents, isolated_paths):
    assert menubar.quit_all() == {"cancelled": 0, "agent_stopped": False}


def test_quit_summary_text():
    assert menubar.quit_summary(2, True) == (
        "This stops the running Saturn agent, cancels 2 pending notifications, and removes "
        "the menu bar icon (it returns the next time you launch saturn)."
    )
    assert menubar.quit_summary(0, False) == (
        "This removes the menu bar icon (it returns the next time you launch saturn)."
    )
    assert "1 pending notification," in menubar.quit_summary(1, False)


# ── /notify icon ─────────────────────────────────────────────────────────────────────────────

def test_notify_icon_status_start_stop(agents, ctx, capsys):
    from commands._framework import dispatch
    dispatch("/notify icon", ctx)
    assert "not running" in capsys.readouterr().out
    dispatch("/notify icon start", ctx)
    assert "started" in capsys.readouterr().out
    dispatch("/notify icon", ctx)
    assert "icon: running" in capsys.readouterr().out
    dispatch("/notify icon stop", ctx)
    assert "stopped" in capsys.readouterr().out
    assert menubar.is_loaded() is False
