from pathlib import Path

from commands._framework import command, _print


@command(
    "ingest",
    "Add a document to the RAG corpus and embed it.",
    usage="/ingest <path>",
    details="""
Copies a file into the RAG corpus and embeds it so the search_knowledge_base tool can retrieve
from it. Supported types: pdf, txt, md, json, jsonl. Paths with spaces don't need quoting.

A no-op if the file is already present and unchanged (matched by content hash). See the corpus
with /docs; remove an entry with /forget.

Examples:
  /ingest C:\\notes\\spec.pdf
  /ingest database/documents/handbook.md
""",
)
def _ingest(ctx, args):
    from stores.rag import ingest_file

    if not args:
        _print("  usage: /ingest <path-to-file>")
        return
    path = " ".join(args)
    s = ingest_file(path)
    if s["added"] or s["updated"]:
        _print(f"  ingested {Path(path).name} — +{s['added']} ~{s['updated']} (cache updated).")
    else:
        _print(f"  {Path(path).name} already up to date in the corpus.")
