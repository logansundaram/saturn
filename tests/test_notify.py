"""
Native scheduled notifications (2026-09-05): the `notify/` backend seam, the macOS LaunchAgent
backend, the `schedule_notification` tool, and the `/notify` command.

Fully offline and platform-independent: `launchctl` / `osascript` never run (the backend's
process runner is captured), the LaunchAgents directory is a tmp path, and the platform selector
is pinned per test so the macOS backend is exercised on the Linux/Windows CI matrix too.
"""

import plistlib
from datetime import datetime, timedelta

import pytest

import notify
from notify import NotifyError, Notification, parse_when
from notify import macos


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────

NOW = datetime(2026, 9, 5, 14, 30, 0).astimezone()


@pytest.fixture
def agents(tmp_path, monkeypatch):
    """A tmp LaunchAgents dir + a captured process runner. Yields (dir, calls)."""
    calls: list[list[str]] = []
    monkeypatch.setattr(macos, "agents_dir", lambda: tmp_path)
    monkeypatch.setattr(macos, "_run", lambda argv: calls.append(list(argv)))
    monkeypatch.setattr(macos, "_uid", lambda: 501)
    return tmp_path, calls


@pytest.fixture
def mac_backend(agents, monkeypatch):
    """Pin the platform selector to the macOS backend regardless of the host OS."""
    be = macos.LaunchdBackend()
    monkeypatch.setattr(notify, "backend", lambda: be)
    return be


def _n(id="ab12cd34", when=NOW + timedelta(hours=1), title="Call the dentist", body="ask about Tuesday"):
    return Notification(id=id, when=when, title=title, body=body)


# ── parse_when ───────────────────────────────────────────────────────────────────────────────

def test_parse_when_iso_local():
    got = parse_when("2026-09-06T09:00", now=NOW)
    assert got == datetime(2026, 9, 6, 9, 0).astimezone()
    assert got.tzinfo is not None


def test_parse_when_iso_with_offset_is_kept():
    got = parse_when("2026-09-06T09:00:00+00:00", now=NOW)
    assert got.utcoffset() == timedelta(0)


@pytest.mark.parametrize("text, delta", [
    ("in 20 minutes", timedelta(minutes=20)),
    ("in 1 minute", timedelta(minutes=1)),
    ("in 2 hours", timedelta(hours=2)),
    ("in 3 days", timedelta(days=3)),
    ("20m", timedelta(minutes=20)),
    ("+2h", timedelta(hours=2)),
])
def test_parse_when_relative(text, delta):
    assert parse_when(text, now=NOW) == NOW + delta


def test_parse_when_tomorrow_at():
    assert parse_when("tomorrow at 09:00", now=NOW) == NOW.replace(day=6, hour=9, minute=0, second=0, microsecond=0)
    assert parse_when("tomorrow 9:05", now=NOW) == NOW.replace(day=6, hour=9, minute=5, second=0, microsecond=0)


def test_parse_when_today_at_and_bare_clock():
    assert parse_when("today at 16:00", now=NOW) == NOW.replace(hour=16, minute=0, second=0, microsecond=0)
    assert parse_when("16:00", now=NOW) == NOW.replace(hour=16, minute=0, second=0, microsecond=0)
    # A bare clock time already past today means the next occurrence — tomorrow.
    assert parse_when("at 09:00", now=NOW) == NOW.replace(day=6, hour=9, minute=0, second=0, microsecond=0)


def test_parse_when_accepts_pm_clock():
    assert parse_when("tomorrow at 3pm", now=NOW) == NOW.replace(day=6, hour=15, minute=0, second=0, microsecond=0)
    assert parse_when("3:30 pm", now=NOW) == NOW.replace(hour=15, minute=30, second=0, microsecond=0)


def test_parse_when_refuses_past_and_hints_current_time():
    with pytest.raises(NotifyError, match="current_time"):
        parse_when("2026-09-05T10:00", now=NOW)
    with pytest.raises(NotifyError, match="past"):
        parse_when("today at 10:00", now=NOW)


def test_parse_when_allow_past_accepts_earlier_times():
    assert parse_when("2020-01-01T00:00", now=NOW, allow_past=True) == datetime(2020, 1, 1).astimezone()
    assert parse_when("today at 09:00", now=NOW, allow_past=True) == NOW.replace(hour=9, minute=0)


def test_parse_when_refuses_garbage():
    with pytest.raises(NotifyError, match="could not understand"):
        parse_when("whenever", now=NOW)
    with pytest.raises(NotifyError):
        parse_when("", now=NOW)


# ── the macOS LaunchAgent backend ────────────────────────────────────────────────────────────

def test_schedule_writes_a_plist_and_bootstraps_it(agents):
    d, calls = agents
    n = _n()
    macos.LaunchdBackend().schedule(n)

    path = d / "com.saturn.notify.ab12cd34.plist"
    assert path.exists()
    with open(path, "rb") as fh:
        plist = plistlib.load(fh)

    assert plist["Label"] == "com.saturn.notify.ab12cd34"
    assert plist["RunAtLoad"] is False
    assert plist["StartCalendarInterval"] == {"Month": 9, "Day": 5, "Hour": 15, "Minute": 30}
    prog = plist["ProgramArguments"]
    assert prog[:2] == ["/bin/sh", "-c"]
    script = prog[2]
    assert "osascript" in script
    assert 'display notification' in script
    assert "Call the dentist" in script and "ask about Tuesday" in script
    # self-cleanup: a one-shot must never re-fire next year
    assert f"rm -f {path}" in script
    assert "launchctl bootout gui/501/com.saturn.notify.ab12cd34" in script
    assert script.index("osascript") < script.index("rm -f") < script.index("bootout")
    # listing metadata rides inside the plist
    meta = plist["SaturnNotification"]
    assert meta["id"] == "ab12cd34"
    assert meta["title"] == "Call the dentist"
    assert meta["when"] == n.when.isoformat(timespec="minutes")

    assert calls == [["launchctl", "bootstrap", "gui/501", str(path)]]


def test_schedule_escapes_applescript_and_shell_metacharacters(agents):
    d, calls = agents
    n = _n(title='Say "hi" $(rm -rf ~)', body="it's `done`\\ ok\nnext line")
    macos.LaunchdBackend().schedule(n)
    with open(d / "com.saturn.notify.ab12cd34.plist", "rb") as fh:
        script = plistlib.load(fh)["ProgramArguments"][2]

    # The osascript -e argument is ONE single-quoted shell word: no $(...) expansion, no
    # backticks, and the AppleScript string escapes its own quotes and backslashes.
    assert '\\"hi\\"' in script
    assert "'\"'\"'" in script or "'\\''" in script          # the shell-quoted apostrophe
    assert "\\\\ ok" in script
    assert "\n" not in script                                 # newlines flattened


def test_schedule_failure_removes_the_plist_and_raises(agents, monkeypatch):
    d, _ = agents

    def boom(argv):
        raise NotifyError("launchctl bootstrap failed: Input/output error")

    monkeypatch.setattr(macos, "_run", boom)
    with pytest.raises(NotifyError, match="bootstrap"):
        macos.LaunchdBackend().schedule(_n())
    assert not (d / "com.saturn.notify.ab12cd34.plist").exists()


def test_cancel_boots_out_and_unlinks(agents):
    d, calls = agents
    be = macos.LaunchdBackend()
    be.schedule(_n())
    calls.clear()

    assert be.cancel("ab12cd34") is True
    assert not (d / "com.saturn.notify.ab12cd34.plist").exists()
    assert calls == [["launchctl", "bootout", "gui/501/com.saturn.notify.ab12cd34"]]


def test_cancel_unknown_id_is_false_and_runs_nothing(agents):
    _, calls = agents
    assert macos.LaunchdBackend().cancel("nope") is False
    assert calls == []


def test_cancel_survives_an_already_unloaded_job(agents, monkeypatch):
    """The job fired (or the machine rebooted) but the plist lingers: bootout fails, the file
    still goes away and cancel reports success."""
    d, _ = agents
    be = macos.LaunchdBackend()
    be.schedule(_n())

    def boom(argv):
        raise NotifyError("launchctl bootout failed: No such process")

    monkeypatch.setattr(macos, "_run", boom)
    assert be.cancel("ab12cd34") is True
    assert not (d / "com.saturn.notify.ab12cd34.plist").exists()


def test_pending_lists_ours_sorted_by_time_and_ignores_strangers(agents):
    d, _ = agents
    be = macos.LaunchdBackend()
    be.schedule(_n(id="later", when=NOW + timedelta(days=2), title="Later"))
    be.schedule(_n(id="soon", when=NOW + timedelta(minutes=5), title="Soon"))
    (d / "com.example.other.plist").write_bytes(plistlib.dumps({"Label": "com.example.other"}))
    (d / "com.saturn.notify.broken.plist").write_bytes(b"not a plist")

    got = be.pending()
    assert [n.id for n in got] == ["soon", "later"]
    assert got[0].title == "Soon" and got[0].when == NOW + timedelta(minutes=5)
    assert isinstance(got[0], Notification)


def test_fire_now_runs_osascript_directly(agents):
    _, calls = agents
    macos.LaunchdBackend().fire_now("Saturn", "test ping")
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "osascript" and argv[1] == "-e"
    assert 'display notification "test ping" with title "Saturn"' == argv[2]


def test_run_wraps_a_failing_process_in_notify_error(monkeypatch):
    import subprocess

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 5, stdout="", stderr="Bootstrap failed: 5: Input/output error\n")

    monkeypatch.setattr(macos.subprocess, "run", fake_run)
    with pytest.raises(NotifyError, match="Input/output error"):
        macos._run(["launchctl", "bootstrap", "gui/501", "/x.plist"])


# ── the platform seam ────────────────────────────────────────────────────────────────────────

def test_backend_selects_launchd_on_darwin(monkeypatch):
    monkeypatch.setattr(notify.sys, "platform", "darwin")
    assert isinstance(notify.backend(), macos.LaunchdBackend)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_backend_is_honest_elsewhere(monkeypatch, platform):
    monkeypatch.setattr(notify.sys, "platform", platform)
    be = notify.backend()
    assert isinstance(be, notify.Unsupported)
    with pytest.raises(NotifyError, match="not supported on"):
        be.schedule(_n())
    with pytest.raises(NotifyError, match="not supported on"):
        be.pending()
    with pytest.raises(NotifyError, match="not supported on"):
        be.cancel("x")
    with pytest.raises(NotifyError, match="not supported on"):
        be.fire_now("a", "b")


def test_new_id_is_short_and_unique():
    ids = {notify.new_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 8 and i.isalnum() for i in ids)


# ── the agent tool ───────────────────────────────────────────────────────────────────────────

def test_tool_is_side_effecting():
    from tools.registry import risk_of, tools_by_name
    assert "schedule_notification" in tools_by_name
    assert risk_of("schedule_notification") == "side_effecting"


def test_tool_schedules_and_reports_the_resolved_time(mac_backend, monkeypatch):
    import tools.notify as tool_mod
    from tools.registry import tools_by_name

    monkeypatch.setattr(tool_mod, "_now", lambda: NOW)
    monkeypatch.setattr(notify, "new_id", lambda: "ab12cd34")
    out = tools_by_name["schedule_notification"].invoke(
        {"when": "in 20 minutes", "title": "Stand up", "body": "stretch"}
    )
    assert out["id"] == "ab12cd34"
    assert out["scheduled_for"] == (NOW + timedelta(minutes=20)).isoformat(timespec="minutes")
    assert out["title"] == "Stand up"
    assert [n.id for n in mac_backend.pending()] == ["ab12cd34"]


def test_tool_refuses_past_time_without_scheduling(mac_backend, monkeypatch):
    import tools.notify as tool_mod
    from tools.registry import tools_by_name

    monkeypatch.setattr(tool_mod, "_now", lambda: NOW)
    out = tools_by_name["schedule_notification"].invoke({"when": "2020-01-01T00:00", "title": "x"})
    assert isinstance(out, str) and out.startswith("Error:") and "current_time" in out
    assert mac_backend.pending() == []


def test_tool_refuses_empty_title(mac_backend):
    from tools.registry import tools_by_name
    out = tools_by_name["schedule_notification"].invoke({"when": "in 5 minutes", "title": "  "})
    assert out.startswith("Error:") and "title" in out
    assert mac_backend.pending() == []


def test_tool_reports_unsupported_platform_honestly(monkeypatch):
    from tools.registry import tools_by_name
    monkeypatch.setattr(notify, "backend", lambda: notify.Unsupported("linux"))
    out = tools_by_name["schedule_notification"].invoke({"when": "in 5 minutes", "title": "x"})
    assert out.startswith("Error:") and "not supported on linux" in out


# ── /notify ──────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ctx():
    import commands.notify  # noqa: F401  — registers the command
    from commands._framework import CommandContext
    return CommandContext(state={}, make_initial_state=dict, db_path="")


def test_notify_lists_pending(mac_backend, ctx, capsys):
    from commands._framework import dispatch
    mac_backend.schedule(_n(id="soon", when=NOW + timedelta(minutes=5), title="Soon"))
    dispatch("/notify", ctx)
    out = capsys.readouterr().out
    assert "soon" in out and "Soon" in out
    assert (NOW + timedelta(minutes=5)).strftime("%H:%M") in out


def test_notify_empty_is_a_note_not_a_crash(mac_backend, ctx, capsys):
    from commands._framework import dispatch
    dispatch("/notify", ctx)
    assert "no pending notifications" in capsys.readouterr().out


def test_notify_cancel(mac_backend, ctx, capsys):
    from commands._framework import dispatch
    mac_backend.schedule(_n(id="soon", when=NOW + timedelta(minutes=5)))
    dispatch("/notify cancel soon", ctx)
    assert "cancelled soon" in capsys.readouterr().out
    assert mac_backend.pending() == []
    dispatch("/notify cancel soon", ctx)
    assert "no pending notification" in capsys.readouterr().out
    dispatch("/notify cancel", ctx)
    assert "usage" in capsys.readouterr().out


def test_notify_test_fires_now(mac_backend, agents, ctx, capsys):
    from commands._framework import dispatch
    _, calls = agents
    dispatch("/notify test hello there", ctx)
    assert calls[-1][0] == "osascript"
    assert "hello there" in calls[-1][2]
    assert "sent" in capsys.readouterr().out


def test_notify_help_and_unknown_verb(ctx, capsys):
    from commands._framework import dispatch
    dispatch("/notify --help", ctx)
    assert "cancel" in capsys.readouterr().out
    dispatch("/notify frobnicate", ctx)
    assert "unknown subcommand" in capsys.readouterr().out


def test_notify_unsupported_platform_warns(ctx, capsys, monkeypatch):
    from commands._framework import dispatch
    monkeypatch.setattr(notify, "backend", lambda: notify.Unsupported("win32"))
    dispatch("/notify", ctx)
    assert "not supported on win32" in capsys.readouterr().out


# ── arg recovery ─────────────────────────────────────────────────────────────────────────────

def test_coerce_args_maps_aliases_onto_when_and_title():
    from core.tool_args import coerce_args, schema_hint
    assert coerce_args("schedule_notification", {"time": "in 5 minutes", "message": "Stretch"}) == {
        "when": "in 5 minutes", "title": "Stretch",
    }
    assert coerce_args("schedule_notification", {"when": "16:00", "title": "x", "body": "y"}) == {
        "when": "16:00", "title": "x", "body": "y",
    }
    assert coerce_args("schedule_notification", {"title": "x"}) is None   # `when` missing → retry
    assert "schedule_notification(when=" in schema_hint("schedule_notification", "missing arg")
