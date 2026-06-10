"""tool_registry/shell.py — foreground run_shell and the background-job lifecycle
(start → check → stop), exercised against real subprocesses kept deliberately tiny."""

import re
import sys

import pytest

from tool_registry import shell as shell_mod
from tool_registry.shell import check_shell_job, run_shell, stop_shell_job

_PY = sys.executable
# The platform shell differs (PowerShell vs /bin/sh): PowerShell needs the call operator `&` to
# run a quoted executable path; POSIX shells take it bare (where `&` would mean background!).
_CALL = f'& "{_PY}"' if sys.platform == "win32" else f'"{_PY}"'


def _job_id(result: str) -> int:
    m = re.search(r"\[job (\d+)\]", result)
    assert m, f"no job id in: {result!r}"
    return int(m.group(1))


def test_foreground_returns_output_and_exit_code(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(\'fg-hello\')"'})
    assert out.startswith("[exit code 0]")
    assert "fg-hello" in out


def test_foreground_nonzero_exit_code_reported(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "import sys; sys.exit(3)"'})
    assert "[exit code" in out and "0]" not in out.splitlines()[0]


def test_background_job_runs_and_is_checkable(isolated_paths):
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


def test_background_listing_and_unknown_ids(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(1)"', "background": True})
    jid = _job_id(out)
    shell_mod._JOBS[jid]["proc"].wait(timeout=30)
    listing = check_shell_job.invoke({"job_id": 0})
    assert f"[job {jid}]" in listing
    assert "No background job 99999" in check_shell_job.invoke({"job_id": 99999})
    assert "No background job 99999" in stop_shell_job.invoke({"job_id": 99999})


def test_stop_kills_running_job(isolated_paths):
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
