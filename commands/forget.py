from commands._framework import command, _print


@command(
    "forget",
    "Remove a document from the RAG corpus and drop its vectors.",
    aliases=("remove",),
    usage="/forget <name>",
    details="""
Removes a document from the RAG corpus: drops its vectors from the store and its entry from the
manifest, then re-syncs. Use the document name as shown by /docs.

This affects only the RAG corpus (database/documents/), not the workspace sandbox. To remove
everything and start clean, delete database/documents/ and run /reingest.

Example:
  /forget spec.pdf
""",
)
def _forget(ctx, args):
    from stores.rag import forget_document

    if not args:
        _print("  usage: /forget <document-name>")
        return
    name = " ".join(args)
    if forget_document(name):
        _print(f"  removed {name} from the corpus — vectors + manifest entry dropped.")
    else:
        _print(f"  no document named {name} in the corpus (see /docs).")
