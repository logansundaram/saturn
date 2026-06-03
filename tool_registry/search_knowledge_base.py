import time
from langchain.tools import tool


@tool
def search_knowledge_base(query: str):
    """Search the local document knowledge base for passages relevant to the query. Use this to answer questions about ingested documents, handbooks, notes, or reference material. Returns the most relevant chunks with their source. Does not search the live web."""
    start = time.perf_counter()
    try:
        # Lazy import so merely importing the registry doesn't load the embedding model.
        from rag import vector_store

        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant documents found in the knowledge base."
        return "\n\n".join(
            f"[source: {d.metadata.get('source', 'unknown')}"
            + (f", page {d.metadata['page']}" if d.metadata.get("page") else "")
            + f"]\n{d.page_content}"
            for d in docs
        )
    finally:
        print(f"search_knowledge_base : {time.perf_counter() - start:.4f}s")
