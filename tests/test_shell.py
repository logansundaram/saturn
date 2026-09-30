"""tools/shell.py — foreground run_shell, exercised against real subprocesses kept
deliberately tiny. (The background-job lifecycle and its opt-in knob were DELETED 2026-07-03 —
run_shell is a bounded foreground run only; see shelf/2026-07-03-runtime-trim.)"""

import sys

import pytest

from tools.shell import run_shell
from tools.toolspec import ToolError

_PY = sys.executable
_CALL = f'"{_PY}"'


def test_foreground_returns_output_and_exit_code(isolated_paths):
    out = run_shell.invoke({"command": f'{_CALL} -c "print(\'fg-hello\')"'})
    assert out.startswith("[exit code 0]")
    assert "fg-hello" in out


def test_foreground_nonzero_exit_code_is_a_failure(isolated_paths):
    """A failed command is a failed step: raised, so the round is stamped error and disclosed."""
    with pytest.raises(ToolError) as info:
        run_shell.invoke({"command": f'{_CALL} -c "import sys; print(\'out\'); sys.exit(3)"'})
    assert str(info.value).startswith("[exit code 3]") and "out" in str(info.value)


def test_the_child_never_reads_saturns_terminal(isolated_paths, monkeypatch):
    """The type-ahead/Esc watcher reads the terminal during a turn; a command that inherited
    it (git's editor, a password prompt, input()) would split keys with it and hang to the
    timeout. The child's stdin is /dev/null: an immediate EOF."""
    import subprocess

    seen = {}
    real = subprocess.Popen

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    with pytest.raises(ToolError, match="EOFError"):
        run_shell.invoke({"command": f'{_CALL} -c "input()"'})
    assert seen.get("stdin") is subprocess.DEVNULL


def test_no_background_surface():
    # The detached-process surface is gone for good: run_shell's schema takes only `command`,
    # and the job tools are not in the registry.
    from tools import toolspec

    assert set(run_shell.args.keys()) == {"command"}
    assert "check_shell_job" not in toolspec._RISK
    assert "stop_shell_job" not in toolspec._RISK
    assert "run_shell" in toolspec._RISK


# ── env scrub (transplanted from the gating isolate's sandbox.scrubbed_env) ─────────────────


def test_scrubbed_env_removes_secret_shaped_variables(monkeypatch):
    """A shell command can read a secret straight out of its own environment; the workspace cwd
    does nothing about that. Secret-shaped variables (substring, case-insensitive fragments from
    `shell.env_scrub`) are removed from the child's env; everything else — PATH, SystemRoot, an
    innocuous EDITOR — survives untouched."""
    from tools import shell as shell_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-survive")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-should-not-survive")
    monkeypatch.setenv("my_secret_thing", "nope")
    monkeypatch.setenv("EDITOR", "vim")
    env = shell_mod.scrubbed_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "my_secret_thing" not in env
    assert env.get("EDITOR") == "vim"
    assert "PATH" in env or "Path" in env


def test_scrubbed_env_honors_the_config_list_and_an_empty_list(monkeypatch):
    from config import get_config
    from tools import shell as shell_mod

    monkeypatch.setenv("FOO_TOKEN", "x")
    monkeypatch.setenv("BARBAZ", "y")
    cfg = get_config()
    prev = cfg.get("shell.env_scrub", None)
    try:
        cfg.set("shell.env_scrub", ["BARBAZ"])
        env = shell_mod.scrubbed_env()
        assert "BARBAZ" not in env and env.get("FOO_TOKEN") == "x"
        cfg.set("shell.env_scrub", [])
        env = shell_mod.scrubbed_env()
        assert env.get("BARBAZ") == "y" and env.get("FOO_TOKEN") == "x"
        # garbage config → the built-in default list (fail toward scrubbing)
        cfg.set("shell.env_scrub", "not-a-list")
        assert "FOO_TOKEN" not in shell_mod.scrubbed_env()
    finally:
        cfg.set("shell.env_scrub", prev)


def test_run_shell_child_does_not_see_secrets(isolated_paths, monkeypatch):
    monkeypatch.setenv("SATURN_TEST_API_KEY", "leak-me")
    monkeypatch.setenv("SATURN_TEST_PLAIN", "keep-me")
    out = run_shell.invoke({"command": f'{_CALL} -c "import os; print(os.environ.get(\'SATURN_TEST_API_KEY\', \'<absent>\'), os.environ.get(\'SATURN_TEST_PLAIN\'))"'})
    assert "<absent> keep-me" in out


def test_env_scrub_is_a_trust_key():
    from commands.config import _TRUST_KEYS

    assert "shell.env_scrub" in _TRUST_KEYS
