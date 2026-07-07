"""The data + persistence layer. One module per store:

    rag.py                the RAG corpus: startup sync + the persisted vector store behind
                          search_knowledge_base (loaders, chunking, admission screening)
    document_registry.py  markdown manifests of the workspace + ingested docs (LLM summaries,
                          hash-cached) — read by the grounding node every turn
    memory_registry.py    durable memory (database/memory/memory.md) behind remember/recall
                          and /memory
    snapshots.py          per-turn pre-write file snapshots — the /undo layer
    trace.py              the Tracer: runs/events/llm_calls into db.sqlite (what /trace,
                          /trace answer #id, and /trace export read)

Data lives under database/ (paths configurable via config.yaml `paths:`); presentation stays
in tui/, never here.
"""
