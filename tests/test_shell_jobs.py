"""tools/shell.py — foreground run_shell and the background-job lifecycle
(start → check → stop), exercised against real subprocesses kept deliberately tiny.
Backgrounding is opt-in (`shell.background`, default off) and pinned at IMPORT
(`_ENABLED_AT_IMPORT`): the lifecycle tests patch that snapshot; the off path must refuse to
detach (even after a live config flip) and register no job tools."""

import re
import sys

import pytest

from tools import shell as shell_mod
from tools.shell import check_shell_job, run_shell, stop_shell_job

_PY = sys.executable
# The platform shell differs (PowerShell vs /bin/sh): PowerShell needs the call operator `&` to
# run a quoted executable path; POSIX shells take it bare (where `&` would mean background!).
_CALL = f'& "{_PY}"' if sys.platform == "win32" else f'"{_PY}"'


def _job_id(result: str) -> int:
    m = re.search(r"\[job (\d+)\]", result)
    assert m, f"no job id in: {result!r}"
    return int(m.group(1))


@pytest.fixture
def background_on(monkeypatch):
    """Enable backgrounding for one test by patching the IMPORT-TIME snapshot — the one value
    run_shell's refusal and the job tools' registration both pin to (a live config flip alone
    is deliberately inert; see test_live_config_flip_alone_still_refuses)."""
    monkeypatch.setattr(shell_mod, "_ENABLED_AT_IMPORT", True)


@pytest.fixture
def background_off(monkeypatch):
    """Pin the snapshot off so the refusal test holds even on a config.yaml with it enabled."""
    monkeypatch.setattr(shell_mod, "_ENABLED_AT_IMPORT", False)


def test_foreground_returns_output_and_exit_code(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(\'fg-hello\')"'})
    assert out.startswith("[exit code 0]")
    assert "fg-hello" in out


def test_foreground_nonzero_exit_code_reported(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "import sys; sys.exit(3)"'})
    assert "[exit code" in out and "0]" not in out.splitlines()[0]


def test_background_job_runs_and_is_checkable(isolated_paths, background_on):
    out = run_shell.invoke(
        {"command": f'{_CALL} -c "print(\'bg-hello\')"', "background": True}
    )
    jid = _job_id(out)
    assert "started in the background" in out
    shell_mod._JOBS[jid]["proc"].wait(timeout=30)

    status = check_shell_job.invoke({"job_id": jid})
    assert "exited with code 0" in status
    assert "bg-hello" in status
    # The log file exists under the isolated jobs dir and holds the output.
    log = shell_mod._JOBS[jid]["log"]
    assert "bg-hello" in open(log, encoding="utf-8", errors="replace").read()


def test_background_listing_and_unknown_ids(isolated_paths, background_on):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(1)"', "background": True})
    jid = _job_id(out)
    shell_mod._JOBS[jid]["proc"].wait(timeout=30)
    listing = check_shell_job.invoke({"job_id": 0})
    assert f"[job {jid}]" in listing
    assert "No background job 99999" in check_shell_job.invoke({"job_id": 99999})
    assert "No background job 99999" in stop_shell_job.invoke({"job_id": 99999})


def test_stop_kills_running_job(isolated_paths, background_on):
    out = run_shell.invoke(
        {"command": f'{_CALL} -c "import time; time.sleep(120)"', "background": True}
    )
    jid = _job_id(out)
    assert shell_mod._JOBS[jid]["proc"].poll() is None  # actually running
    stopped = stop_shell_job.invoke({"job_id": jid})
    assert "stopped" in stopped
    assert shell_mod._JOBS[jid]["proc"].poll() is not None
    # Stopping again reports the already-finished state instead of erroring.
    again = stop_shell_job.invoke({"job_id": jid})
    assert "already" in again


def test_knob_off_refuses_to_detach(isolated_paths, background_off):
    before = set(shell_mod._JOBS)
    out = run_shell.invoke({"command": f'{_CALL} -c "print(1)"', "background": True})
    assert "background jobs are disabled" in out
    assert "shell.background" in out  # the refusal names the knob...
    assert "restart" in out           # ...and says enabling needs a restart
    assert set(shell_mod._JOBS) == before  # nothing detached, no job registered


def test_live_config_flip_alone_still_refuses(isolated_paths, monkeypatch):
    """A mid-session `/config shell.background true` must NOT half-enable backgrounding: the job
    tools were never registered at import, so detaching would leave a process no registered tool
    can check or stop (and the success message would coach the model into calling one). The
    refusal pins to the import-time snapshot, not the live knob."""
    from config import get_config

    monkeypatch.setattr(shell_mod, "_ENABLED_AT_IMPORT", False)
    monkeypatch.setitem(get_config()._data.setdefault("shell", {}), "background", True)
    assert shell_mod._background_enabled() is True  # the live knob really did flip
    before = set(shell_mod._JOBS)
    out = run_shell.invoke({"command": f'{_CALL} -c "print(1)"', "background": True})
    assert "background jobs are disabled" in out
    assert "restart" in out
    assert set(shell_mod._JOBS) == before  # still nothing detached


def test_job_tools_registration_follows_import_time_knob():
    # Registration is decided once at import from shell.background (shipped default: false) —
    # under the default the job tools never enter the registry the planner catalog/gate read,
    # while run_shell itself always does. Asserted against the import-time SNAPSHOT (no fixture
    # active here, so it holds the value registration actually ran under) rather than a
    # hard-coded false, so a locally-enabled config.yaml doesn't false-fail the suite.
    from tools import toolspec

    registered = shell_mod._ENABLED_AT_IMPORT
    assert ("check_shell_job" in toolspec._RISK) == registered
    assert ("stop_shell_job" in toolspec._RISK) == registered
    assert "run_shell" in toolspec._RISK
