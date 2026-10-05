"""Hooks — the user's own shell commands on four moments of a turn.

`~/.saturn/hooks.yaml` ($SATURN_HOME overrides the folder), hand-written:

    turn-start:
      - say "working"
    turn-end:
      - command: ~/bin/log-answer.sh
        timeout: 5
    before-write:
      - ~/bin/refuse-outside-drafts.sh     # a non-zero exit blocks the write
    after-write:
      - git -C ~/notes add -A
      - 'echo "wrote: $SATURN_FILE" >> ~/saturn-writes.log'   # quote a command holding ": "

Each command runs through the shell in the working folder with the moment's details in the
environment (SATURN_HOOK_EVENT, SATURN_QUERY, SATURN_ANSWER, SATURN_FILE, SATURN_TOOL,
SATURN_WORKSPACE) and the same details as one JSON object on stdin. Output goes to the diag log.
`before-write` is the one that can say no: a non-zero exit refuses the write, and the hook's
stderr (or stdout) becomes the refusal the model reads.

The user's hooks are the user's commands: they run without the gate and are not the agent's
egress. That is exactly why the file tools refuse to write this file (tools/files
`_control_files`) — the model must never be able to plant a command that later runs ungated.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import diag

EVENTS = ("turn-start", "turn-end", "before-write", "after-write")
_DEFAULT_TIMEOUT = 10.0
_ENV_VALUE_CAP = 8000  # an answer rides stdin whole; the env copy is clipped


@dataclass
class HookResult:
    command: str
    code: int          # the exit code; -1 when it timed out or could not start
    output: str        # stderr, else stdout, stripped — what a refusal quotes


def hooks_path() -> Path:
    from config import saturn_home

    return saturn_home() / "hooks.yaml"


def load() -> "dict[str, list[dict]]":
    """{event: [{command, timeout}]} from the hooks file. A missing file is no hooks; a broken
    one is logged and treated as no hooks — a typo must not fail every turn. `problems()` says
    what was skipped, for the startup warning."""
    return _parse()[0]


def problems() -> "list[str]":
    """What the hooks file got wrong (each skipped entry, an unreadable file) — the REPL warns
    once at startup, so a hook that silently never runs can't happen."""
    return _parse()[1]


def _parse() -> "tuple[dict, list[str]]":
    path = hooks_path()
    if not path.is_file():
        return {}, []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("the top level must be a mapping of event: [commands]")
    except Exception as exc:
        return {}, _logged([f"unreadable, no hooks run: {' '.join(str(exc).split())}"])
    out: dict = {}
    issues: list = []
    for event, entries in data.items():
        if event not in EVENTS:
            issues.append(f"unknown event {event!r} skipped (known: {', '.join(EVENTS)})")
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            if isinstance(entry, str):
                entry = {"command": entry}
            if not isinstance(entry, dict) or not str(entry.get("command") or "").strip():
                hint = ""
                if isinstance(entry, dict) and "command" not in entry:
                    hint = " — a command containing ': ' must be quoted"
                issues.append(f"{event} entry skipped, no command: {entry!r}{hint}")
                continue
            try:
                timeout = float(entry.get("timeout", _DEFAULT_TIMEOUT))
            except (TypeError, ValueError):
                timeout = _DEFAULT_TIMEOUT
            out.setdefault(event, []).append({"command": str(entry["command"]), "timeout": timeout})
    return out, _logged(issues)


def _logged(issues: list) -> list:
    for issue in issues:
        diag.log(f"hooks: {hooks_path()}: {issue}")
    return issues


def run(event: str, **details) -> "list[HookResult]":
    """Run every hook for `event`, in file order, and return what each did. `details` (query,
    answer, file, tool) ride the environment as SATURN_<NAME> and stdin as JSON. Never raises."""
    hooks = load().get(event, [])
    if not hooks:
        return []
    from core import workspace

    cwd = workspace.root()
    payload = {"event": event, "workspace": str(cwd), **{k: v for k, v in details.items() if v is not None}}
    env = dict(os.environ)
    for key, value in payload.items():
        env[f"SATURN_{key.upper()}"] = str(value)[:_ENV_VALUE_CAP]
    env["SATURN_HOOK_EVENT"] = event
    results = []
    for hook in hooks:
        cmd = hook["command"]
        try:
            proc = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd.is_dir() else None, env=env,
                                  input=json.dumps(payload), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=hook["timeout"])
            code, output = proc.returncode, (proc.stderr or proc.stdout or "").strip()
        except subprocess.TimeoutExpired:
            code, output = -1, f"timed out after {hook['timeout']:g}s"
        except OSError as exc:
            code, output = -1, str(exc)
        diag.log(f"hooks: {event} `{cmd}` → exit {code}" + (f": {output[:200]}" if output else ""))
        results.append(HookResult(cmd, code, output))
    return results


def before_write(path, tool: str) -> "str | None":
    """The refusal text when a before-write hook says no (a non-zero exit), else None."""
    for r in run("before-write", file=str(path), tool=tool):
        if r.code != 0:
            return (f"Blocked by your before-write hook (`{r.command}`)"
                    + (f": {r.output}" if r.output else "") + " — the file was not changed.")
    return None
