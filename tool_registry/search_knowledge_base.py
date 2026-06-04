from langchain.tools import tool


@tool
def search_knowledge_base(query: str):
    """Search the local document knowledge base for passages relevant to the query. Use this to answer questions about ingested documents, handbooks, notes, or reference material. Returns the most relevant chunks with their source. Does not search the live web."""
    # Lazy import so merely importing the registry doesn't load the embedding model.
    from rag import get_vector_store

    # k=6: at k=3 recall was too low and the agent compensated by re-searching and
    # falling back to read_file (see benchmark thrashing on RAG queries).
    docs = get_vector_store().similarity_search(query, k=6)
    if not docs:
        return "No relevant documents found in the knowledge base."
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}"
        + (f", page {d.metadata['page']}" if d.metadata.get("page") else "")
        + f"]\n{d.page_content}"
        for d in docs
    )
