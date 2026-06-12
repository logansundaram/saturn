"""
Local-knowledge tools — what the agent already knows, on this machine.

  search_knowledge_base — semantic search over the local RAG store (the ingested corpus).
  remember / recall     — durable facts via `memory_registry` (the markdown memory store).

Kept separate from the live-web tools (`web.py`): these search the user's OWN data, not the
internet. (remember/recall lived in tools/memory.py until the 2026-06-11 leaf consolidation.)
Remembered facts are also injected into the grounding context each turn, so `recall` is mainly
for searching a large memory or confirming a specific detail.
"""

from tools.toolspec import register_tool

from stores.memory_registry import add_memory, search_memory


@register_tool("read_only", retrieval=True)
def search_knowledge_base(query: str):
    """Search the local document knowledge base for passages relevant to the query. Use this to answer questions about ingested documents, handbooks, notes, or reference material. Returns the most relevant chunks with their source. Does not search the live web."""
    # Lazy import so merely importing the registry doesn't load the embedding model.
    from stores.rag import get_vector_store, retrieval_k

    # k defaults to 6 (rag.k in config.yaml): at k=3 recall was too low and the agent
    # compensated by re-searching and falling back to read_file (see benchmark thrashing
    # on RAG queries).
    docs = get_vector_store().similarity_search(query, k=retrieval_k())
    if not docs:
        return "No relevant documents found in the knowledge base."
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}"
        + (f", page {d.metadata['page']}" if d.metadata.get("page") else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


@register_tool("side_effecting")
def remember(fact: str, category: str = "general"):
    """Save a durable fact about the user or their preferences to persistent memory so it is
    remembered in future sessions. Use this when the user shares a lasting preference, a fact
    about themselves, or explicitly asks you to remember something (e.g. "I prefer terse
    answers", "my timezone is PST"). `fact` is a single concise statement. `category` is an
    optional label such as preference, identity, or project. Do NOT use this for one-off,
    conversation-specific details."""
    return add_memory(fact, category)


@register_tool("read_only")
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
