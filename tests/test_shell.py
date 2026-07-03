"""tools/shell.py — foreground run_shell, exercised against real subprocesses kept
deliberately tiny. (The background-job lifecycle and its opt-in knob were DELETED 2026-07-03 —
run_shell is a bounded foreground run only; see shelf/2026-07-03-runtime-trim.)"""

import sys

from tools.shell import run_shell

_PY = sys.executable
# The platform shell differs (PowerShell vs /bin/sh): PowerShell needs the call operator `&` to
# run a quoted executable path; POSIX shells take it bare (where `&` would mean background!).
_CALL = f'& "{_PY}"' if sys.platform == "win32" else f'"{_PY}"'


def test_foreground_returns_output_and_exit_code(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(\'fg-hello\')"'})
    assert out.startswith("[exit code 0]")
    assert "fg-hello" in out


def test_foreground_nonzero_exit_code_reported(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "import sys; sys.exit(3)"'})
    assert "[exit code" in out and "0]" not in out.splitlines()[0]


def test_no_background_surface():
    # The detached-process surface is gone for good: run_shell's schema takes only `command`,
    # and the job tools are not in the registry.
    from tools import toolspec

    assert set(run_shell.args.keys()) == {"command"}
    assert "check_shell_job" not in toolspec._RISK
    assert "stop_shell_job" not in toolspec._RISK
    assert "run_shell" in toolspec._RISK
