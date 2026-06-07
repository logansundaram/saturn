from commands._framework import command, _print
from stores.document_registry import read_documents_manifest, read_workspace_manifest


@command(
    "docs",
    "View ingested RAG documents and workspace files.",
    aliases=("documents",),
    details="""
Prints two manifests: the ingested RAG corpus (documents the agent can search through the
search_knowledge_base tool) and the workspace sandbox files (where the file tools read/write).

Manage the corpus with /ingest (add), /forget (remove), and /reingest (full rebuild).

Example:
  /docs
""",
)
def _docs(ctx, args):
    docs = read_documents_manifest().strip()
    ws = read_workspace_manifest().strip()
    _print("  === ingested documents (RAG corpus) ===")
    _print(docs if docs else "  (none ingested)")
    _print("")
    _print("  === workspace files ===")
    _print(ws if ws else "  (empty)")
