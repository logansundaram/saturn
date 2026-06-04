from langchain.tools import tool

from memory_registry import search_memory


@tool
def recall(query: str = ""):
    """Retrieve durable facts previously saved to persistent memory about the user or their
    preferences. `query` filters to matching facts (case-insensitive); an empty query returns
    everything remembered. Use this to check what you already know about the user before asking
    them to repeat something. Note: remembered facts are also loaded into your context each
    turn, so use this mainly to search a large memory or confirm a specific detail."""
    facts = search_memory(query)
    if not facts:
        return "No matching facts in persistent memory."
    return "\n".join(f"- {f}" for f in facts)
