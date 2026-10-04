"""
Local-knowledge tools — what the agent already knows, on this machine.

  search_knowledge_base — semantic search over the local RAG store (the ingested corpus).
  remember / recall     — durable facts via `memory_registry` (the layered markdown store).

Kept separate from the live-web tools (`web.py`): these search the user's OWN data, not the
internet. Remembered facts are also injected into the grounding context each turn, so `recall` is mainly
for searching a large memory or confirming a specific detail.
"""

from textutil import doc_source_label
from tools.toolspec import human_approved, register_tool, user_stated

from stores.memory_registry import add_memory, search_memory


# untrusted=True: the corpus may hold DOWNLOADED documents — retrieved chunks are external
# content the quarantine scans (admission screening reduces, but does not remove, the risk).
@register_tool("read_only", retrieval=True, untrusted=True)
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
    # textutil.doc_source_label — the one builder of the `[source: …]` marker the answer's
    # Sources labels (core/sources.py) parse back (parse_doc_sources); construction and parsing can't drift.
    return "\n\n".join(
        doc_source_label(d.metadata.get("source", "unknown"), d.metadata.get("page"))
        + f"\n{d.page_content}"
        for d in docs
    )


@register_tool("side_effecting")
def remember(fact: str, category: str = "general", layer: str = "user", replaces: "str | int" = "",
             sensitivity: str = ""):
    """Save a durable fact to persistent memory so it is remembered in future sessions. Use this
    when the user shares a lasting preference, a fact about themselves or the people in their
    life, a standing rule, or explicitly asks you to remember something (e.g. "I prefer terse
    answers", "my timezone is PST"). `fact` is a single concise statement in the user's own
    words — reuse the words they typed. `category` is an optional label such as preference,
    identity, or project.
    `layer` is where it belongs: "user" (preferences, identity, constraints and standing rules
    such as "always…", "never…", "from now on…" — the default),
    "entities" (a person, project, place, document, or the user's shorthand for one),
    "commitments" (a to-do, reminder, or deadline), "negative" (an approach or suggestion not
    to bring up again), "agent" (operating knowledge about this machine or its tools),
    "memo" (a dated note about what happened). `replaces` is the #id of an earlier fact this one
    supersedes (the ids are shown in your memory context, e.g. "#3") — use it when the user
    corrects a fact, so the old one is retired instead of contradicting the new one.
    `sensitivity` marks a private fact ("health", "money", "private"): it is then withheld from
    any prompt bound for a remote inference host. Do NOT use this for one-off,
    conversation-specific details."""
    from stores.trace import current_run_id

    from core.auto_memory import rule_layer

    # by=user is a person's yes to THIS fact: the gate, or the user having typed every word of
    # it (auto-learn — core/auto_memory, stamped src=said). A call that ran because the tier was
    # raised or the gate was open is the model's inference, and is recorded as one.
    approved, said = human_approved(), user_stated()
    by = "user" if approved or said else "inferred"
    return add_memory(fact, category, layer=rule_layer(fact, layer), replaces=replaces or None,
                      by=by, run_id=current_run_id(),
                      sensitivity=(sensitivity or "").strip() or None,
                      src="said" if said and not approved else None)


@register_tool("read_only")
def recall(query: str = ""):
    """Retrieve durable facts previously saved to persistent memory (every layer: user
    preferences, entities, commitments, notes, operating knowledge, things not to do). `query`
    filters to matching facts (case-insensitive); an empty query returns everything remembered.
    Use this to check what you already know before asking the user to repeat something. Note:
    the relevant facts are also loaded into your context each turn, so use this mainly to search
    for a fact that did not load or to confirm a specific detail."""
    facts = search_memory(query)
    if not facts:
        return "No matching facts in persistent memory."
    return "\n".join(f"- {f}" for f in facts)
