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


DOCUMENTS_DIR = get_config().path("documents")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def iter_documents():
    """Yield the document files RAG will ingest — supported extensions under DOCUMENTS_DIR,
    recursively. The single source of truth for 'what counts as a document', shared by the
    ingest pipeline and the startup banner so their counts can't drift."""
    if not DOCUMENTS_DIR.exists():
        return
    for file_path in DOCUMENTS_DIR.glob("**/*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            yield file_path


# Embedding model comes from the active tier's `embedder` slot (config.yaml).
embeddings = get_embeddings()
vector_store = InMemoryVectorStore(embeddings)

# consider caching as a local file so no need for embedding every time the program is run


class IngestState(TypedDict):
    documents: List[Any]


def build_ingest():
    def load_documents(state: IngestState):
        docs = []
        for file_path in iter_documents():
            source = str(file_path.relative_to(DOCUMENTS_DIR))
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
        print(f"Loaded {len(docs)} documents from {DOCUMENTS_DIR}")
        return {"documents": docs}

    def split_documents(state: IngestState):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(state["documents"])
        return {"documents": chunks}

    def store_documents(state: IngestState):
        vector_store.add_documents(state["documents"])
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
        docs = vector_store.similarity_search(query, k=4)
        return {"documents_retrieved": docs}

    retrieval_builder = StateGraph(AgentState)
    retrieval_builder.add_node("retrieve_docs", retrieve_docs)
    retrieval_builder.add_edge(START, "retrieve_docs")
    retrieval_builder.add_edge("retrieve_docs", END)
    return retrieval_builder.compile()
