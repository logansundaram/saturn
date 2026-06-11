"""
Shell / code-execution tools — `run_shell` (+ the background-job pair `check_shell_job` /
`stop_shell_job`).

The agent's escape hatch (roadmap Tier 3 #10): the one tool that can run anything the host shell
can — scripts, build/test commands, git, package managers, a quick one-off bit of code. That reach
is exactly why it is registered `destructive`, so it ALWAYS hits the approval
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
interactive command can't wedge the turn, mirroring `runtime.llm_timeout`. stdout and stderr
are combined and returned with the exit code; the
tool_node clamps the observation before it enters context (gotcha #5), so a runaway command can't
overflow the window.

Background — `run_shell(background=True)` is the escape hatch from that timeout for processes
MEANT to outlive the turn (dev servers, watchers, long builds): the command starts detached, its
output streams to a log under `paths.shell_jobs`, and the agent checks/stops it later via the
job tools (see the "background jobs" section below). Still `destructive` — the gate shows the
exact command either way; backgrounding changes the lifetime, not the trust boundary.
"""

import atexit
import itertools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import diag

from config import get_config
from toolspec import register_tool

# Fallback when config.yaml has no `shell.timeout` (or an invalid one). Mirrors the local-helper
# style web.py uses for its own knobs — no config.py property needed for a single tool-local value.
_DEFAULT_TIMEOUT = 60

# ── background jobs ────────────────────────────────────────────────────────────────────────────
# `run_shell(background=True)` starts a command DETACHED from the turn: the call returns
# immediately with a job id while the process keeps running (a dev server, a long build), its
# output captured to a log file under `paths.shell_jobs` (logging/shell/). `check_shell_job`
# (read_only) reports status + the log tail; `stop_shell_job` (side_effecting — still gated under
# the default policy) kills the job's whole process tree. Jobs are session-scoped on purpose: any
# still running when Saturn exits are terminated by the atexit hook below — a trust-focused agent
# must never leave invisible processes running after the user closes it. The log files remain for
# post-mortem reading.
_JOBS: "dict[int, dict]" = {}
_JOB_IDS = itertools.count(1)
_ATEXIT_ARMED = False

# How much of a job's log check_shell_job returns (the tail — the most recent output). The
# tool_node clamp would bound it anyway; reading only the tail also keeps file IO small.
_LOG_TAIL_BYTES = 8000


def _jobs_dir() -> Path:
    """Where job logs land (`paths.shell_jobs`, default logging/shell). Falls back beside the
    database dir for a user config.yaml written before the key existed — a missing path must
    degrade, not break backgrounding."""
    cfg = get_config()
    try:
        d = cfg.path("shell_jobs")
    except KeyError:
        d = cfg.path("database").parent / "logging" / "shell"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _arm_atexit() -> None:
    """Install the session-end cleanup once: kill any still-running background jobs so quitting
    Saturn never leaves orphaned processes behind. Best-effort — exit must never hang on it."""
    global _ATEXIT_ARMED
    if _ATEXIT_ARMED:
        return
    _ATEXIT_ARMED = True

    def _cleanup():
        for job in _JOBS.values():
            try:
                if job["proc"].poll() is None:
                    _kill_tree(job["proc"])
            except Exception:
                pass
            _close_log(job)

    atexit.register(_cleanup)


def _close_log(job: dict) -> None:
    fh = job.get("fh")
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass
        job["fh"] = None


def _job_status(job: dict) -> str:
    """One-word state + exit code when finished. Also closes the log handle the first time a
    finished job is observed, so the file is fully flushed for reading."""
    code = job["proc"].poll()
    if code is None:
        return "running"
    _close_log(job)
    return f"exited with code {code}"


def _log_tail(job: dict) -> str:
    """The most recent output from the job's log file, decoded tolerantly (the process writes raw
    bytes; the workspace and shells emit non-UTF-8 sequences)."""
    try:
        raw = Path(job["log"]).read_bytes()
    except OSError:
        return "(no output captured yet)"
    if not raw:
        return "(no output yet)"
    tail = raw[-_LOG_TAIL_BYTES:]
    text = tail.decode("utf-8", errors="replace").strip()
    if len(raw) > _LOG_TAIL_BYTES:
        return f"... [showing the last {_LOG_TAIL_BYTES} bytes of {len(raw)}] ...\n{text}"
    return text


def _start_background(command: str, argv, popen_kwargs: dict) -> str:
    """Launch `command` detached and register it in the session job table. The log file receives
    the process's combined stdout+stderr directly (byte-level — no pipe to drain, so nothing
    blocks and nothing is lost if Saturn is busy)."""
    job_id = next(_JOB_IDS)
    log_path = _jobs_dir() / f"job_{job_id}.log"
    fh = open(log_path, "wb")
    header = f"$ {command}\n[started {time.strftime('%Y-%m-%d %H:%M:%S')}]\n\n"
    fh.write(header.encode("utf-8"))
    fh.flush()

    popen_kwargs = dict(popen_kwargs)
    # Replace the pipes with the log file; drop the text decoding (the file takes raw bytes).
    popen_kwargs.update(stdout=fh, stderr=subprocess.STDOUT)
    for k in ("text", "encoding", "errors"):
        popen_kwargs.pop(k, None)

    proc = subprocess.Popen(argv, **popen_kwargs)
    _JOBS[job_id] = {
        "proc": proc,
        "command": command,
        "log": str(log_path),
        "started": time.time(),
        "fh": fh,
    }
    _arm_atexit()
    return (
        f"[job {job_id}] started in the background (pid {proc.pid}).\n"
        f"Output is being captured to {log_path}.\n"
        f"Check on it with check_shell_job(job_id={job_id}); stop it with "
        f"stop_shell_job(job_id={job_id}). Background jobs still running when the session ends "
        "are terminated."
    )


def _timeout() -> "float | None":
    """Max seconds to let a command run before it is terminated (config `shell.timeout`). None
    disables the timeout (a value <= 0); an invalid value falls back to `_DEFAULT_TIMEOUT`."""
    v = get_config().get("shell.timeout", _DEFAULT_TIMEOUT)
    try:
        n = float(v)
    except (TypeError, ValueError):
        return float(_DEFAULT_TIMEOUT)
    return n if n > 0 else None


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Terminate the command's whole process TREE, not just the shell. A bare proc.kill() (what
    subprocess.run's own timeout does) only takes out the direct child — a grandchild the command
    started (a server, a hung build step) would survive the 'timeout' and keep running unattended.
    Windows: taskkill /T walks the tree. POSIX: the child runs in its own session (start_new_session
    below), so killing its process group gets everything. Best-effort, falls back to a plain kill."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _format(returncode: int, stdout: str, stderr: str) -> str:
    """Compose the observation: exit-code header + combined stdout/stderr. The exit code is kept
    explicit so the agent can tell success from failure without guessing from the text."""
    body = ((stdout or "") + (stderr or "")).strip()
    header = f"[exit code {returncode}]"
    return f"{header}\n{body}" if body else f"{header} (no output)"


@register_tool("destructive")
def run_shell(command: str, background: bool = False):
    """Runs a shell command on the host machine and returns its combined stdout+stderr plus the exit code. Use this for anything no other tool covers: running scripts or quick one-off code, build/test commands, git, package managers, inspecting the system. `command` is a single command line interpreted by the host's default shell (PowerShell on Windows, /bin/sh on Unix) — chain steps with the shell's own operators (`;`, `&&`, `|`). It runs inside the workspace directory by default. Set `background=true` for a long-running process (a server, a watcher, a long build): the call returns immediately with a job id, output is captured to a log file, and you check on or stop it later with check_shell_job / stop_shell_job — never run a server in the foreground, it will just time out. Background jobs are terminated when the session ends. This is a powerful, irreversible action and always requires user approval; do not assume it succeeded — check the returned exit code (or the job's status)."""
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

        popen_kwargs = dict(
            shell=use_shell,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # The workspace holds arbitrary user content and shells emit non-cp1252 bytes;
            # decode as UTF-8 and degrade undecodable bytes to a marker (matching files.py)
            # rather than letting a UnicodeDecodeError crash the turn.
            encoding="utf-8",
            errors="replace",
        )
        if sys.platform != "win32":
            # Own session = own process group, so a timeout can kill the whole tree (_kill_tree).
            popen_kwargs["start_new_session"] = True

        # Detached mode: start the job, return its handle immediately — no timeout applies (the
        # whole point is outliving the turn). Same gate, same argv, same workspace cwd.
        if background:
            return _start_background(command, argv, popen_kwargs)

        # Popen + communicate (not subprocess.run): on timeout we need the child's pid to kill
        # its whole process tree — run() kills only the direct child, leaving grandchildren alive.
        proc = subprocess.Popen(argv, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            # Collect whatever the command printed before it was killed — it's often the most
            # useful part (e.g. a test runner that hung mid-suite).
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            partial = ((stdout or "") + (stderr or "")).strip()
            msg = f"Command timed out after {timeout:g}s and was terminated."
            return f"{msg}\n{partial}" if partial else msg

        return _format(proc.returncode, stdout, stderr)
    except Exception as exc:  # never let a shell failure kill the turn — report it to the agent
        return f"Shell execution failed: {exc}"
    finally:
        diag.log(f"run_shell : {time.perf_counter() - start:.4f}s")


def _fmt_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


@register_tool("read_only")
def check_shell_job(job_id: int = 0):
    """Checks on background shell jobs started with run_shell(background=true). With a job_id, returns that job's status (running, or its exit code) plus the most recent output from its log. With job_id=0 (the default), lists every background job this session with its status. Read-only — looking never affects the job."""
    start = time.perf_counter()
    try:
        if not _JOBS:
            return "No background jobs have been started this session."
        if not job_id:
            lines = ["Background jobs this session:"]
            for jid, job in sorted(_JOBS.items()):
                runtime = _fmt_runtime(time.time() - job["started"])
                cmd = job["command"]
                if len(cmd) > 80:
                    cmd = cmd[:79] + "…"
                lines.append(f"  [job {jid}] {_job_status(job)} · {runtime} · {cmd}")
            lines.append("Pass a job_id for that job's output.")
            return "\n".join(lines)
        job = _JOBS.get(int(job_id))
        if job is None:
            known = ", ".join(str(j) for j in sorted(_JOBS)) or "none"
            return f"No background job {job_id} (known jobs: {known})."
        status = _job_status(job)
        runtime = _fmt_runtime(time.time() - job["started"])
        return (
            f"[job {job_id}] {status} ({runtime} since start)\n"
            f"command: {job['command']}\n"
            f"log: {job['log']}\n\n"
            f"{_log_tail(job)}"
        )
    except Exception as exc:
        return f"check_shell_job failed: {exc}"
    finally:
        diag.log(f"check_shell_job : {time.perf_counter() - start:.4f}s")


@register_tool("side_effecting")
def stop_shell_job(job_id: int):
    """Stops a background shell job started with run_shell(background=true), killing its whole process tree (the job and anything it spawned). Use check_shell_job first to confirm which job to stop. The job's log file is kept for reading after the stop."""
    start = time.perf_counter()
    try:
        job = _JOBS.get(int(job_id))
        if job is None:
            known = ", ".join(str(j) for j in sorted(_JOBS)) or "none"
            return f"No background job {job_id} (known jobs: {known})."
        if job["proc"].poll() is not None:
            return f"[job {job_id}] already {_job_status(job)} — nothing to stop."
        _kill_tree(job["proc"])
        try:
            job["proc"].wait(timeout=10)
        except Exception:
            pass
        _close_log(job)
        status = _job_status(job)
        return f"[job {job_id}] stopped ({status}). Log kept at {job['log']}."
    except Exception as exc:
        return f"stop_shell_job failed: {exc}"
    finally:
        diag.log(f"stop_shell_job : {time.perf_counter() - start:.4f}s")
