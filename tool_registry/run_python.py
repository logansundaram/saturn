"""
run_python — gated, sandboxed Python execution (Phase 4, the capability "force multiplier").

The agent's general-purpose escape hatch: computation, data wrangling, parsing, format
conversion, file generation — anything better done by running code than by reasoning about it.
A single orthogonal primitive that subsumes the long tail, which is what keeps the tool suite
small (SATURDAY_MVP_PLAN.md §"Essential tool suite").

Sandboxing (MVP, honest scope): the snippet runs in a SEPARATE interpreter process (never
in-process, so it can't corrupt the agent), in Python's isolated mode (`-I`: ignores
env/user-site), with its working directory pinned to the workspace sandbox so file I/O lands
there with plain relative paths. A wall-clock timeout kills runaways; output is size-capped so a
chatty script can't blow the context window. This is NOT a security boundary — the snippet can
still reach the network and the wider filesystem by absolute path. The REAL control is the
approval gate: run_python is risk-tier `destructive`, so every call is shown to the user and
must be approved before it runs. True OS-level isolation (containers / seccomp) is post-MVP.

Self-correction: stdout, stderr, and any traceback are returned verbatim, so when code fails the
ReAct loop sees the error as a ToolMessage and can fix the snippet and retry — the Phase 4 exit
criterion ("agent can run and self-correct code").
"""

import sys
import subprocess

from langchain.tools import tool

from config import get_config

# Guardrails. The timeout is clamped into [_MIN, _MAX]; output is truncated to _MAX_OUTPUT
# characters per stream so a runaway print loop can't flood the trace / synthesis input.
_DEFAULT_TIMEOUT = 30
_MIN_TIMEOUT = 1
_MAX_TIMEOUT = 120
_MAX_OUTPUT = 4000


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    """Truncate a stream to `limit` chars, noting how much was dropped."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return text[:limit] + f"\n… [truncated {dropped} more characters]"


@tool
def run_python(code: str, timeout_seconds: int = _DEFAULT_TIMEOUT) -> str:
    """Run a short Python 3 script in the sandboxed workspace and return its output.

    Use this for computation, data wrangling, parsing, format conversion, file generation, and
    anything better solved by running code than by reasoning. The script runs in a separate
    process with its working directory set to the workspace sandbox (database/workspace), so
    reading and writing files there works with plain relative paths (e.g. open('out.csv','w')).

    PRINT what you want to see — only stdout and stderr are captured, not bare expression values
    (so `2+2` shows nothing; `print(2+2)` shows 4). The script is killed after `timeout_seconds`
    (default 30, max 120). If it raises, the full traceback is returned: read it, fix the code,
    and call run_python again. This tool requires user approval before each run.
    """
    timeout = max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, int(timeout_seconds)))

    workspace = get_config().path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        # `-I` isolated mode (ignore env vars + user site); read the script from stdin so
        # there's no command-line length/escaping limit and nothing is left on disk. cwd pins
        # file I/O to the workspace. utf-8 + errors=replace so non-cp1252 output never crashes.
        proc = subprocess.run(
            [sys.executable, "-I", "-"],
            input=code,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: execution timed out after {timeout}s and was killed. "
            "Reduce the work, add a smaller input, or avoid blocking calls (network, "
            "sleeps, infinite loops)."
        )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # Clean success path: just the output (or an explicit note when the script printed nothing,
    # so the model doesn't assume the run failed silently).
    if proc.returncode == 0 and not stderr:
        return _clip(stdout) if stdout else "(ran successfully, no output — remember to print() results)"

    # Error / non-empty stderr: hand back everything, labeled, for self-correction.
    sections = [f"exit code: {proc.returncode}"]
    sections.append("--- stdout ---\n" + (_clip(stdout) if stdout else "(empty)"))
    sections.append("--- stderr ---\n" + (_clip(stderr) if stderr else "(empty)"))
    return "\n".join(sections)
