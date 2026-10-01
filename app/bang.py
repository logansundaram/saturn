"""
`!<command>` at the prompt: the user runs a shell command THEMSELVES, sees its
output, and the output rides into their next message as an attachment — so `!git diff` then
"summarize that" works the way `git diff | saturn -q "summarize"` does headless.

This is the user's own action, not the agent's: no gate, no trace row, no scrubbed env — it
is exactly the command they would have typed in the terminal next door, run in the directory
Saturn was launched from. What DOES apply is the attachment admission warning (the output is
their data, not their words) and the same length clamp @file mentions get.
"""

from __future__ import annotations

import os
import subprocess

from core.mentions import _MAX_FILE_CHARS

TIMEOUT = 120.0
_HEADER = ("### Shell output attached to this message (the user ran `!{command}` themselves"
           " before sending it; exit status {code})")


def is_bang(line: str) -> bool:
    """A line whose first non-space character is `!` followed by a command. A bare `!` (or
    `!!`, `!?` — punctuation, not a command) is an ordinary message."""
    s = line.lstrip()
    return s.startswith("!") and bool(s[1:].strip()) and s[1] not in "!?"


def command_of(line: str) -> str:
    return line.lstrip()[1:].strip()


def run(command: str, cwd: str | None = None) -> tuple[str, int]:
    """Run `command` in the user's shell and return (combined output, exit status). A timeout
    is reported as output, not raised — the prompt must come back."""
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                              errors="replace", timeout=TIMEOUT, cwd=cwd or os.getcwd())
    except subprocess.TimeoutExpired:
        return f"[timed out after {TIMEOUT:.0f}s]", 124
    except OSError as exc:
        return f"[could not run: {exc}]", 126
    out = proc.stdout
    if proc.stderr:
        out = (out + "\n" if out else "") + proc.stderr
    return out.rstrip(), proc.returncode


def attachment(command: str, output: str, code: int) -> str:
    """The context block for the next turn: clamped like an @file, fenced as data."""
    body = output
    if len(body) > _MAX_FILE_CHARS:
        body = body[:_MAX_FILE_CHARS] + f"\n… [truncated — output exceeds {_MAX_FILE_CHARS} chars]"
    return _HEADER.format(command=command, code=code) + "\n```\n" + body + "\n```"
