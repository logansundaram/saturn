from langgraph.graph import StateGraph, START, END
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import TypedDict, List, Any
import pypdf

from config import get_config
from llms import get_embeddings
from state import AgentState
from document_registry import register_rag_document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def documents_dir():
    """The corpus directory, resolved from config at call time so a live `/config paths.documents`
    change is honored without a restart."""
    return get_config().path("documents")


def iter_documents():
    """Yield the document files RAG will ingest — supported extensions under the corpus dir,
    recursively. The single source of truth for 'what counts as a document', shared by the
    ingest pipeline and the startup banner so their counts can't drift."""
    root = documents_dir()
    if not root.exists():
        return
    for file_path in root.glob("**/*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            yield file_path


# ── vector store (lazy, embedder-aware) ──────────────────────────────────────────────────────
# Built on first use from the active tier's `embedder` (config.yaml) and rebuilt when that
# embedder changes, so switching to a tier with a different embedder actually takes effect —
# reset_models() only clears the chat-model caches, not this store. A rebuilt store is empty
# until re-ingested; sync_to_config() does both. (Caching embeddings to disk is a future TODO.)
_embeddings = None
_vector_store = None
_store_embedder = None  # the embedder id the live store was built with


def _build_store() -> None:
    global _embeddings, _vector_store, _store_embedder
    _embeddings = get_embeddings()
    _vector_store = InMemoryVectorStore(_embeddings)
    _store_embedder = get_config().embedder_model


def get_vector_store():
    """The active in-memory vector store, built lazily on first use."""
    if _vector_store is None:
        _build_store()
    return _vector_store


def sync_to_config() -> bool:
    """Re-embed the corpus if the configured embedder changed since the store was built. Returns
    True if it re-ingested. Call after a live model/tier change (e.g. /model, /config) so an
    embedder swap takes effect; a no-op when the embedder is unchanged."""
    if _store_embedder == get_config().embedder_model:
        return False
    _build_store()
    build_ingest().invoke({"documents": []})
    return True


class IngestState(TypedDict):
    documents: List[Any]


def build_ingest():
    def load_documents(state: IngestState):
        docs = []
        root = documents_dir()
        for file_path in iter_documents():
            source = str(file_path.relative_to(root))
            if file_path.suffix == ".pdf":
                reader = pypdf.PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": source, "page": page_num + 1},
                            )
                        )
                full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                register_rag_document(source, full_text)
            else:
                text = file_path.read_text(encoding="utf-8")
                docs.append(Document(page_content=text, metadata={"source": source}))
                register_rag_document(source, text)
        print(f"Loaded {len(docs)} documents from {root}")
        return {"documents": docs}

    def split_documents(state: IngestState):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(state["documents"])
        return {"documents": chunks}

    def store_documents(state: IngestState):
        get_vector_store().add_documents(state["documents"])
        print(f"Stored {len(state['documents'])} chunks in vector store")
        return {}

    ingest_builder = StateGraph(IngestState)
    ingest_builder.add_node("load_documents", load_documents)
    ingest_builder.add_node("split_documents", split_documents)
    ingest_builder.add_node("store_documents", store_documents)
    ingest_builder.add_edge(START, "load_documents")
    ingest_builder.add_edge("load_documents", "split_documents")
    ingest_builder.add_edge("split_documents", "store_documents")
    ingest_builder.add_edge("store_documents", END)
    return ingest_builder.compile()


def build_retrieval():
    def retrieve_docs(state: AgentState):
        query = state["messages"][-1].content
        docs = get_vector_store().similarity_search(query, k=4)
        return {"documents_retrieved": docs}

    retrieval_builder = StateGraph(AgentState)
    retrieval_builder.add_node("retrieve_docs", retrieve_docs)
    retrieval_builder.add_edge(START, "retrieve_docs")
    retrieval_builder.add_edge("retrieve_docs", END)
    return retrieval_builder.compile()
