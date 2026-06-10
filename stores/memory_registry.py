"""
Persistent working-memory store (Phase 3) — the durable layer of the three-layer memory
(SATURDAY_MVP_PLAN.md §3).

Session memory rides in the checkpointed message thread; the knowledge base lives in the RAG
store. This module is the third layer: a small, append-only markdown file of durable facts the
user asked the agent to remember ("I prefer terse answers"), persisted across sessions and
reloaded into every turn by the grounding node.

It is deliberately a flat markdown file, not a database: it is human-readable, hand-editable,
and shows up in the workspace like everything else. The `remember` / `recall` tools are thin
wrappers over `add_memory` / `search_memory`; the grounding node calls `read_memory_block`.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

from config import get_config


def _atomic_write(path: Path, text: str) -> None:
    """Write durable memory crash-safely: write a sibling temp file, then os.replace (atomic on
    Windows + POSIX). A crash mid-write leaves the original intact rather than truncating the
    user's irreplaceable facts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

_HEADER = (
    "# Saturday — persistent memory\n\n"
    "Durable facts the user asked me to remember. Loaded into context every turn. "
    "One fact per bullet; safe to hand-edit.\n\n"
)


def _memory_path() -> Path:
    return get_config().path("memory")


def _read_raw() -> str:
    path = _memory_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _facts(text: str) -> list[str]:
    """Extract the fact bullets (lines starting with '- ') from the store text."""
    return [
        line[2:].strip()
        for line in text.splitlines()
        if line.startswith("- ") and line[2:].strip()
    ]


# The "(YYYY-MM-DD) [category] " prefix add_memory writes; stripped so dedup compares bare facts.
_PREFIX_RE = re.compile(r"^\(\d{4}-\d{2}-\d{2}\)\s*(?:\[[^\]]*\]\s*)?")


def _fact_text(stored: str) -> str:
    """Strip the date/category prefix from a stored fact line, leaving the bare fact text."""
    return _PREFIX_RE.sub("", stored).strip()


def add_memory(fact: str, category: str = "general") -> str:
    """Append a durable fact. No-op (reported) if an identical fact is already stored."""
    fact = (fact or "").strip()
    if not fact:
        return "Nothing to remember — the fact was empty."

    existing = _facts(_read_raw())
    # Dedup on the bare fact text (date/category prefix stripped), case-insensitive — and by
    # equality, not substring: a short new fact must not be swallowed just because it appears
    # inside a longer stored line (e.g. remembering "Python" when "I use Python at work" exists).
    if any(fact.lower() == _fact_text(line).lower() for line in existing):
        return f"Already remembered: {fact!r}"

    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tag = f"[{category}] " if category and category != "general" else ""
    entry = f"- ({date.today()}) {tag}{fact}\n"

    if not path.exists():
        _atomic_write(path, _HEADER + entry)
    else:
        current = path.read_text(encoding="utf-8").rstrip("\n")
        _atomic_write(path, current + "\n" + entry)
    return f"Remembered: {fact!r}"


def search_memory(query: str = "") -> list[str]:
    """Return stored facts matching `query` (case-insensitive substring). Empty query returns
    everything — durable memory is small, so a full dump is reasonable."""
    facts = _facts(_read_raw())
    query = (query or "").strip().lower()
    if not query:
        return facts
    return [f for f in facts if query in f.lower()]


def list_memory() -> list[str]:
    """The stored fact lines (with their date/category prefix), in file order. Backs the
    /memory command's numbered listing — the user-facing view of what the agent permanently
    believes, so it must show exactly what the grounding node will load."""
    return _facts(_read_raw())


def remove_memory(index: int) -> "str | None":
    """Delete the 1-based Nth stored fact (the numbering /memory shows). Returns the removed
    fact line, or None if the index is out of range. Rewrites the file atomically with the
    standard header — hand-written non-bullet lines outside the header are not preserved (the
    file's contract is one fact per bullet; see _HEADER)."""
    facts = _facts(_read_raw())
    if not (1 <= index <= len(facts)):
        return None
    removed = facts.pop(index - 1)
    body = "".join(f"- {f}\n" for f in facts)
    _atomic_write(_memory_path(), _HEADER + body)
    return removed


def read_memory_block() -> str:
    """The stored facts formatted for injection into the grounding context. Empty string if
    nothing has been remembered yet (the grounding node then omits the section)."""
    facts = _facts(_read_raw())
    if not facts:
        return ""
    return "\n".join(f"- {f}" for f in facts)
