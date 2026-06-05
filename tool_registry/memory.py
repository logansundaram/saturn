"""
Persistent-memory tools — thin wrappers over `memory_registry` (the durable markdown store).

  remember — save a durable fact/preference about the user (persists across sessions).
  recall   — look up facts previously saved.

Remembered facts are also injected into the grounding context each turn, so `recall` is mainly
for searching a large memory or confirming a specific detail.
"""

import time

from langchain.tools import tool

from stores.memory_registry import add_memory, search_memory


@tool
def remember(fact: str, category: str = "general"):
    """Save a durable fact about the user or their preferences to persistent memory so it is
    remembered in future sessions. Use this when the user shares a lasting preference, a fact
    about themselves, or explicitly asks you to remember something (e.g. "I prefer terse
    answers", "my timezone is PST"). `fact` is a single concise statement. `category` is an
    optional label such as preference, identity, or project. Do NOT use this for one-off,
    conversation-specific details."""
    start = time.perf_counter()
    try:
        return add_memory(fact, category)
    finally:
        print(f"remember : {time.perf_counter() - start:.4f}s")


@tool
def recall(query: str = ""):
    """Retrieve durable facts previously saved to persistent memory about the user or their
    preferences. `query` filters to matching facts (case-insensitive); an empty query returns
    everything remembered. Use this to check what you already know about the user before asking
    them to repeat something. Note: remembered facts are also loaded into your context each
    turn, so use this mainly to search a large memory or confirm a specific detail."""
    start = time.perf_counter()
    try:
        facts = search_memory(query)
        if not facts:
            return "No matching facts in persistent memory."
        return "\n".join(f"- {f}" for f in facts)
    finally:
        print(f"recall : {time.perf_counter() - start:.4f}s")
