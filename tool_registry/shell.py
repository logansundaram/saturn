"""
Shell / code-execution tool — `run_shell`.

The agent's general-purpose escape hatch (roadmap Tier 3 #10): the one tool that can do anything
the host shell can — run scripts, build/test commands, git, package managers, a quick one-off bit
of code. That reach is exactly why it is registered `destructive`, so it ALWAYS hits the approval
gate. The gate — the user seeing the exact command + working directory and approving it — is the
safety boundary here, NOT a path jail. This is the design the roadmap calls "a `destructive`
run_shell safe-by-default": the risk tier does the guarding, the same way write_file's overwrite
is made safe by the gate (gotcha #2) rather than by being forbidden.

Working directory — every call runs inside `config.path("workspace")` by default, resolved per
call so a live `/config paths.workspace` change is honored, matching the file tools' sandbox
(files.py). A shell can of course `cd` out of it; this is a sensible default, not a hard jail.

Cross-platform — the raw command line is handed to the host's own shell (PowerShell on Windows,
/bin/sh elsewhere) so the agent writes native syntax and chains with the shell's own operators.

Bounded — the call is killed after `shell.timeout` seconds (config.yaml `shell:`) so a hung or
interactive command can't wedge the turn, mirroring `runtime.llm_timeout` /
`web.deep_research_timeout`. stdout and stderr are combined and returned with the exit code; the
tool_node clamps the observation before it enters context (gotcha #5), so a runaway command can't
overflow the window.
"""

import subprocess
import sys
import time

import diag

from config import get_config
from toolspec import register_tool

# Fallback when config.yaml has no `shell.timeout` (or an invalid one). Mirrors the local-helper
# style web.py uses for its own knobs — no config.py property needed for a single tool-local value.
_DEFAULT_TIMEOUT = 60


def _timeout() -> "float | None":
    """Max seconds to let a command run before it is terminated (config `shell.timeout`). None
    disables the timeout (a value <= 0); an invalid value falls back to `_DEFAULT_TIMEOUT`."""
    v = get_config().get("shell.timeout", _DEFAULT_TIMEOUT)
    try:
        n = float(v)
    except (TypeError, ValueError):
        return float(_DEFAULT_TIMEOUT)
    return n if n > 0 else None


def _format(returncode: int, stdout: str, stderr: str) -> str:
    """Compose the observation: exit-code header + combined stdout/stderr. The exit code is kept
    explicit so the agent can tell success from failure without guessing from the text."""
    body = ((stdout or "") + (stderr or "")).strip()
    header = f"[exit code {returncode}]"
    return f"{header}\n{body}" if body else f"{header} (no output)"


@register_tool("destructive")
def run_shell(command: str):
    """Runs a shell command on the host machine and returns its combined stdout+stderr plus the exit code. Use this for anything no other tool covers: running scripts or quick one-off code, build/test commands, git, package managers, inspecting the system. `command` is a single command line interpreted by the host's default shell (PowerShell on Windows, /bin/sh on Unix) — chain steps with the shell's own operators (`;`, `&&`, `|`). It runs inside the workspace directory by default. This is a powerful, irreversible action and always requires user approval; do not assume it succeeded — check the returned exit code."""
    start = time.perf_counter()
    timeout = _timeout()
    try:
        workspace = get_config().path("workspace")
        workspace.mkdir(parents=True, exist_ok=True)

        # Hand the raw command line to the platform's own shell so native syntax works. On Windows
        # we explicitly invoke PowerShell (the project's shell) rather than rely on shell=True,
        # which would use cmd.exe; elsewhere shell=True is /bin/sh. -NonInteractive guards against a
        # command that would otherwise block forever waiting on a prompt the agent can't answer.
        if sys.platform == "win32":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            use_shell = False
        else:
            argv = command
            use_shell = True

        try:
            proc = subprocess.run(
                argv,
                shell=use_shell,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                # The workspace holds arbitrary user content and shells emit non-cp1252 bytes;
                # decode as UTF-8 and degrade undecodable bytes to a marker (matching files.py)
                # rather than letting a UnicodeDecodeError crash the turn.
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Surface whatever the command printed before it was killed — it's often the most
            # useful part (e.g. a test runner that hung mid-suite). exc.stdout/stderr are str here
            # (text=True); they may be None if nothing was captured.
            partial = ((exc.stdout or "") + (exc.stderr or "")).strip()
            msg = f"Command timed out after {timeout:g}s and was terminated."
            return f"{msg}\n{partial}" if partial else msg

        return _format(proc.returncode, proc.stdout, proc.stderr)
    except Exception as exc:  # never let a shell failure kill the turn — report it to the agent
        return f"Shell execution failed: {exc}"
    finally:
        diag.log(f"run_shell : {time.perf_counter() - start:.4f}s")
