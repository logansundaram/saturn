from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.messages import SystemMessage
from typing import TypedDict, List

from state import AgentState


embeddings = OllamaEmbeddings(model="gemma3:4b")
vector_store = InMemoryVectorStore(embeddings)


class IngestState(TypedDict):
    documents: List[str]


def build_ingest():
    def split_documents(state: IngestState):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=text) for text in state["documents"]]
        chunks = splitter.split_documents(docs)
        return {"documents": [chunk.page_content for chunk in chunks]}

    def store_documents(state: IngestState):
        docs = [Document(page_content=text) for text in state["documents"]]
        vector_store.add_documents(docs)
        print(f"Stored {len(docs)} chunks in vector store")
        return {}

    ingest_builder = StateGraph(IngestState)
    ingest_builder.add_node("split_documents", split_documents)
    ingest_builder.add_node("store_documents", store_documents)
    ingest_builder.add_edge(START, "split_documents")
    ingest_builder.add_edge("split_documents", "store_documents")
    ingest_builder.add_edge("store_documents", END)
    return ingest_builder.compile()


def build_retrieval():
    def retrieve_docs(state: AgentState):
        query = state["messages"][-1].content
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"messages": SystemMessage(content=f"Relevant context:\n{context}")}

    retrieval_builder = StateGraph(AgentState)
    retrieval_builder.add_node("retrieve_docs", retrieve_docs)
    retrieval_builder.add_edge(START, "retrieve_docs")
    retrieval_builder.add_edge("retrieve_docs", END)
    return retrieval_builder.compile()
