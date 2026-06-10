from pathlib import Path

from commands._framework import command, _print


@command(
    "ingest",
    "Add a document to the RAG corpus and embed it.",
    usage="/ingest <path>",
    details="""
Copies a file into the RAG corpus and embeds it so the search_knowledge_base tool can retrieve
from it. Supported types: pdf, txt, md. Paths with spaces don't need quoting — and a dragged
file's quoted path works as-is (tip: type `/ingest ` then drag the file onto the terminal).

A no-op if the file is already present and unchanged (matched by content hash). See the corpus
with /docs; remove an entry with /forget.

Examples:
  /ingest C:\\notes\\spec.pdf
  /ingest "C:\\my notes\\spec.pdf"
  /ingest database/documents/handbook.md
""",
)
def _ingest(ctx, args):
    from stores.rag import ingest_file, SUPPORTED_EXTENSIONS

    if not args:
        _print("  usage: /ingest <path-to-file>")
        _print("  tip: drag the file onto the terminal to paste its path.")
        return
    # Dragged paths arrive quoted; strip the quotes (and expand ~) before resolving.
    path = Path(" ".join(args).strip().strip("\"'")).expanduser()
    if not path.is_file():
        _print(f"  not a file: {path}")
        return
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _print(f"  unsupported type '{path.suffix}' — supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return
    s = ingest_file(str(path))
    if s["added"] or s["updated"]:
        _print(f"  ingested {Path(path).name} — +{s['added']} ~{s['updated']} (cache updated).")
    else:
        _print(f"  {Path(path).name} already up to date in the corpus.")
