"""`@file` mentions — pull a local file's contents into a turn inline.

A user message can reference local files as `@path/to/file` (Tab-completed at the `»` prompt; see
`tui.ui`'s completer). Before the turn runs, `agent.py` calls `expand()` to resolve each `@token`
to a readable file under the current working directory, read it (clamped), and format it as a
context block. The block is appended to `state["attachments"]`, which the grounding node folds into
`state["context"]` — so every node that reads context (planner, agent, synthesize) sees the file
without a tool round-trip or an approval prompt.

Design choices:
  - Resolution is cwd-relative (the user launches from the repo root) plus `~` expansion and
    absolute paths. The user is typing the path themselves in their own REPL, so this is not the
    sandboxed, model-driven path the file *tools* take — it's a convenience, like `@`-mentions in
    other agent CLIs.
  - Only existing, readable FILES attach; a token that doesn't resolve (a stray "@handle", a
    directory, a typo) is left as literal text — no error, no attachment.
  - Each file is clamped to mirror the tool-output clamp, so one large `@file` can't silently blow
    the context window.
  - The user's message text is untouched: the `@mention` stays visible in `messages`/the trace; the
    contents ride `context` for that turn only (reset each turn, never persisted into history).
"""

from __future__ import annotations

import os
import re

# A mention is `@` at a word boundary followed by a run of non-space, non-`@` characters.
_MENTION_RE = re.compile(r"(?<!\S)@([^\s@]+)")

# Mirror tool_node._clamp_observation's budget: one @file can't overflow the window.
_MAX_FILE_CHARS = 12_000

# Trailing punctuation to try stripping so `@notes.md.` / `@file)` / `@a.py,` resolve to the file
# (path extensions keep their dots — we only strip from the END, and only if the full token misses).
_TRAILING_PUNCT = ".,;:!?)]}>\"'`"


def _resolve(token: str) -> str | None:
    """Resolve a mention token to an existing, readable file path, trying the token as-typed and
    then with trailing sentence punctuation stripped. Returns None for a miss or a directory."""
    for cand in (token, token.rstrip(_TRAILING_PUNCT)):
        if not cand:
            continue
        path = os.path.expanduser(cand)
        if os.path.isfile(path):
            return path
    return None


def find_mentions(text: str) -> list[str]:
    """The resolvable file paths referenced by `@token`s in `text`, deduped in first-seen order."""
    seen: dict[str, None] = {}
    for m in _MENTION_RE.finditer(text or ""):
        path = _resolve(m.group(1))
        if path and path not in seen:
            seen[path] = None
    return list(seen)


def display(path: str) -> str:
    """A compact label for the attached file: cwd-relative when possible, else the path as given."""
    try:
        rel = os.path.relpath(path)
        # relpath can climb out of cwd with ../../… — that's noisier than the absolute path.
        return rel if not rel.startswith(".." + os.sep) and rel != ".." else path
    except ValueError:  # different drive on Windows
        return path


def _read_clamped(path: str) -> str:
    """Read a file as UTF-8 (replacing undecodable bytes), clamped to _MAX_FILE_CHARS with a marker.
    Never raises — an unreadable file becomes an inline note so the turn still runs."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(_MAX_FILE_CHARS + 1)
    except OSError as exc:
        return f"[could not read {display(path)}: {exc}]"
    if len(data) > _MAX_FILE_CHARS:
        return data[:_MAX_FILE_CHARS] + f"\n… [truncated — {display(path)} exceeds {_MAX_FILE_CHARS} chars]"
    return data


def expand(text: str) -> tuple[str, list[str]]:
    """Resolve the `@file` mentions in `text` to an attachments context block.

    Returns `(block, paths)`: `block` is a markdown section embedding each resolved file's contents
    (empty string when nothing resolved), suitable to append to the grounding context; `paths` is
    the list of attached files (for the UI's "attached N file(s)" note). The input `text` itself is
    never modified — the `@mention` stays in the user's message verbatim."""
    paths = find_mentions(text)
    if not paths:
        return "", []
    parts = ["### Files attached to this message (referenced inline with @)"]
    for path in paths:
        label = display(path)
        parts.append(f"\n#### {label}\n```\n{_read_clamped(path)}\n```")
    return "\n".join(parts), paths
