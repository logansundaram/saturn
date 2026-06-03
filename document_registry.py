"""
Maintains per-directory markdown manifests that record metadata and LLM-generated
summaries for every document the agent can see.

Two manifests:
  database/workspace/.manifest.md  — files the agent can read/write via tools
  database/documents/.manifest.md  — files ingested into the RAG vector store

The context_builder node reads both at the start of each turn so the planning LLM
knows exactly what documents exist and what they contain before deciding whether to
call read_file or trigger retrieval.
"""

"""
vibe coded, need to review
"""

import time
from datetime import date
from pathlib import Path

from langchain.messages import HumanMessage

_DB_ROOT = Path(__file__).parent / "database"
DOCUMENTS_DIR = _DB_ROOT / "documents"
WORKSPACE_DIR = _DB_ROOT / "workspace"

WORKSPACE_MANIFEST = WORKSPACE_DIR / ".manifest.md"
DOCUMENTS_MANIFEST = DOCUMENTS_DIR / ".manifest.md"

_MANIFEST_HEADER = "# Document manifest\n\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_workspace_file(file_path: str, content: str) -> None:
    """Called by write_file after a successful write."""
    path = Path(file_path)
    _upsert(WORKSPACE_MANIFEST, path.name, content, path.suffix)


def register_rag_document(source: str, content: str) -> None:
    """Called by rag.build_ingest for each loaded document."""
    path = Path(source)
    _upsert(DOCUMENTS_MANIFEST, path.name, content, path.suffix)


def read_workspace_manifest() -> str:
    return (
        WORKSPACE_MANIFEST.read_text(encoding="utf-8")
        if WORKSPACE_MANIFEST.exists()
        else ""
    )


def read_documents_manifest() -> str:
    return (
        DOCUMENTS_MANIFEST.read_text(encoding="utf-8")
        if DOCUMENTS_MANIFEST.exists()
        else ""
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summarize(content: str, filename: str) -> str:
    """Ask the LLM for a 1-2 sentence summary. Falls back gracefully on error."""
    try:
        # Import here to avoid circular import at module load time.
        from llms import llm

        start = time.perf_counter()
        msg = HumanMessage(
            content=(
                f"Summarize the following document in 1-2 sentences. "
                f"Be specific: name what information it contains, not just its topic. "
                f"Document name: {filename}\n\n{content[:4000]}"
            )
        )
        response = llm.invoke([msg])
        print(
            f"document_registry summary ({filename}) : {time.perf_counter() - start:.4f}s"
        )
        return response.content.strip()
    except Exception as exc:
        print(f"document_registry: summary failed for {filename}: {exc}")
        return "No summary available."


def _upsert(manifest_path: Path, filename: str, content: str, suffix: str) -> None:
    summary = _summarize(content, filename)
    size_kb = len(content.encode()) / 1024

    entry_body = (
        f"- **Type**: {suffix or 'unknown'} | "
        f"**Added**: {date.today()} | "
        f"**Size**: {size_kb:.1f} KB\n"
        f"{summary}"
    )
    full_entry = f"### {filename}\n{entry_body}\n"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        manifest_path.write_text(_MANIFEST_HEADER + full_entry, encoding="utf-8")
        return

    content_text = manifest_path.read_text(encoding="utf-8")
    marker = f"### {filename}\n"

    if marker in content_text:
        start_idx = content_text.index(marker)
        rest = content_text[start_idx + len(marker) :]
        next_entry = rest.find("\n### ")
        if next_entry == -1:
            content_text = content_text[:start_idx] + full_entry
        else:
            content_text = (
                content_text[:start_idx] + full_entry + rest[next_entry + 1 :]
            )
    else:
        content_text = content_text.rstrip("\n") + "\n\n" + full_entry + "\n"

    manifest_path.write_text(content_text, encoding="utf-8")
