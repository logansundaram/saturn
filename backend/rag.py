from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import TypedDict, List, Any

from state import AgentState
from document_registry import register_rag_document


DOCUMENTS_DIR = Path("database/documents").resolve()
SUPPORTED_EXTENSIONS = {".txt", ".md"}


# need a new embedding model
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vector_store = InMemoryVectorStore(embeddings)


class IngestState(TypedDict):
    documents: List[Any]


def build_ingest():
    def load_documents(state: IngestState):
        docs = []
        for file_path in DOCUMENTS_DIR.glob("**/*"):
            if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
                text = file_path.read_text(encoding="utf-8")
                source = str(file_path.relative_to(DOCUMENTS_DIR))
                docs.append(Document(page_content=text, metadata={"source": source}))
                register_rag_document(source, text)
        print(f"Loaded {len(docs)} documents from {DOCUMENTS_DIR}")
        return {"documents": docs}

    def split_documents(state: IngestState):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
        docs = vector_store.similarity_search(query, k=3)
        return {"documents_retrieved": docs}

    retrieval_builder = StateGraph(AgentState)
    retrieval_builder.add_node("retrieve_docs", retrieve_docs)
    retrieval_builder.add_edge(START, "retrieve_docs")
    retrieval_builder.add_edge("retrieve_docs", END)
    return retrieval_builder.compile()
