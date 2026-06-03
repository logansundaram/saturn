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

from datetime import date
from pathlib import Path

from config import get_config

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


def add_memory(fact: str, category: str = "general") -> str:
    """Append a durable fact. No-op (reported) if an identical fact is already stored."""
    fact = (fact or "").strip()
    if not fact:
        return "Nothing to remember — the fact was empty."

    existing = _facts(_read_raw())
    # Dedup on the fact text, ignoring the date/category prefix we add.
    if any(fact.lower() in line.lower() for line in existing):
        return f"Already remembered: {fact!r}"

    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tag = f"[{category}] " if category and category != "general" else ""
    entry = f"- ({date.today()}) {tag}{fact}\n"

    if not path.exists():
        path.write_text(_HEADER + entry, encoding="utf-8")
    else:
        current = path.read_text(encoding="utf-8").rstrip("\n")
        path.write_text(current + "\n" + entry, encoding="utf-8")
    return f"Remembered: {fact!r}"


def search_memory(query: str = "") -> list[str]:
    """Return stored facts matching `query` (case-insensitive substring). Empty query returns
    everything — durable memory is small, so a full dump is reasonable."""
    facts = _facts(_read_raw())
    query = (query or "").strip().lower()
    if not query:
        return facts
    return [f for f in facts if query in f.lower()]


def read_memory_block() -> str:
    """The stored facts formatted for injection into the grounding context. Empty string if
    nothing has been remembered yet (the grounding node then omits the section)."""
    facts = _facts(_read_raw())
    if not facts:
        return ""
    return "\n".join(f"- {f}" for f in facts)
