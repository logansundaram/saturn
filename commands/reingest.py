from commands._framework import command, _print


@command(
    "reingest",
    "Rebuild the RAG vector store + cache from database/documents/.",
    usage="/reingest",
    details="""
Forces a full rebuild of the RAG vector store from database/documents/: re-embeds every
document and refreshes the on-disk cache (vectors.json + index.json).

Slower than the startup sync, which only embeds new/changed files. Reach for this after
editing a document in place (same filename, new content the hash already covers) or to recover
from a corrupted/stale cache. To add or drop a single file, prefer /ingest or /forget.

Example:
  /reingest
""",
)
def _reingest(ctx, args):
    from stores.rag import sync

    s = sync(force=True, verbose=False)
    n = s["added"] + s["updated"]
    _print(f"  reingested {n} document(s) — full rebuild, disk cache refreshed.")
