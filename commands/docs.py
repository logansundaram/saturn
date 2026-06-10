import time
from pathlib import Path

from commands._framework import command, _print


@command(
    "docs",
    "View and manage the RAG corpus (and see workspace files): /docs add | remove | sync.",
    aliases=("documents",),
    usage="/docs [add <path> | remove <name> | sync [--force]]",
    details="""
The one front door to the document knowledge base (what search_knowledge_base retrieves from),
plus a view of the workspace sandbox files (where the file tools read/write).

  /docs                 list the ingested corpus + the workspace files
  /docs add <path>      copy a file into the corpus and embed it (pdf/txt/md). Paths with
                        spaces don't need quoting; a dragged file's quoted path works as-is
                        (tip: type `/docs add ` then drag the file onto the terminal).
                        A no-op if the file is already present and unchanged (content hash).
  /docs remove <name>   remove a document: drops its vectors and manifest entry (rm/drop work)
  /docs sync            re-scan the corpus directory and embed anything new/changed
  /docs sync --force    full rebuild: re-embed every document (recovers a stale/corrupt cache;
                        also how an edited rag.chunk_size/embedder change is applied on demand)

Durable memory is separate — see /memory. (This command replaces the old /ingest, /forget,
and /reingest.)

Examples:
  /docs add "C:\\my notes\\spec.pdf"
  /docs remove spec.pdf
  /docs sync --force
""",
)
def _docs(ctx, args):
    if not args:
        _list_docs()
        return
    sub = args[0].lower()
    rest = args[1:]
    if sub == "add":
        _add(rest)
    elif sub in ("remove", "rm", "drop", "forget"):
        _remove(rest)
    elif sub == "sync":
        _sync(force=any(a in ("--force", "-f", "force") for a in rest))
    else:
        _print(f"  unknown subcommand '{args[0]}' — usage: /docs [add <path> | remove <name> | sync [--force]]")


def _list_docs() -> None:
    from config import get_config
    from stores.document_registry import (
        manifest_entries, read_documents_manifest, read_workspace_manifest,
    )
    from tui import ui

    corpus = manifest_entries(read_documents_manifest())
    ws = manifest_entries(read_workspace_manifest())

    ui.section(
        "documents",
        f"{len(corpus)} in the RAG corpus  ·  embedder {get_config().embedder_model}",
    )
    if corpus:
        ui.table(
            [(e["name"], (e["type"] or "·", "dim"), (e["size"] or "·", "dim"),
              (e["added"] or "·", "dim"), (e["summary"], "dim"))
             for e in corpus]
        )
    else:
        ui.note("none ingested — add one with /docs add <path>")

    _print("")
    ui.section("workspace", f"{len(ws)} file(s) the file tools can read/write")
    if ws:
        ui.table(
            [(e["name"], (e["type"] or "·", "dim"), (e["size"] or "·", "dim"),
              (e["added"] or "·", "dim"), (e["summary"], "dim"))
             for e in ws]
        )
    else:
        ui.note("empty — the agent writes here via write_file/edit_file")


def _add(rest: list) -> None:
    from stores.rag import ingest_file, SUPPORTED_EXTENSIONS

    if not rest:
        _print("  usage: /docs add <path-to-file>")
        _print("  tip: drag the file onto the terminal to paste its path.")
        return
    # Dragged paths arrive quoted; strip the quotes (and expand ~) before resolving.
    path = Path(" ".join(rest).strip().strip("\"'")).expanduser()
    if not path.is_file():
        _print(f"  not a file: {path}")
        return
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _print(f"  unsupported type '{path.suffix}' — supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return
    s = ingest_file(str(path))
    failed = dict(s.get("failed") or [])
    if any(src == path.name or src.endswith(path.name) for src in failed):
        err = next(e for src, e in failed.items() if src == path.name or src.endswith(path.name))
        _print(f"  could not load {path.name}: {err}")
    elif s["added"] or s["updated"]:
        _print(f"  added {path.name} — +{s['added']} ~{s['updated']} (cache updated).")
    else:
        _print(f"  {path.name} already up to date in the corpus.")


def _remove(rest: list) -> None:
    from stores.rag import forget_document

    if not rest:
        _print("  usage: /docs remove <document-name>   (names as shown by /docs)")
        return
    # Accept a full (possibly quoted/dragged) path too — the corpus is keyed by basename.
    name = Path(" ".join(rest).strip().strip("\"'")).name
    if forget_document(name):
        _print(f"  removed {name} from the corpus — vectors + manifest entry dropped.")
    else:
        _print(f"  no document named {name} in the corpus (see /docs).")


def _sync(*, force: bool) -> None:
    from config import get_config
    from stores.rag import iter_documents, sync
    from tui import ui

    n = sum(1 for _ in iter_documents())
    if force:
        ui.note(
            f"full rebuild — re-embedding {n} document(s) with "
            f"{get_config().embedder_model}; this can take a while…"
        )
    start = time.perf_counter()
    s = sync(
        force=force,
        verbose=False,
        on_file=lambda src, i, total: ui.note(f"embedding {src}  ({i}/{total})"),
    )
    took = time.perf_counter() - start
    _print(
        f"  synced in {took:.1f}s — +{s['added']} added  ~{s['updated']} updated  "
        f"-{s['removed']} removed  ={s['unchanged']} unchanged"
        + ("  [full rebuild]" if s["rebuilt"] else "")
    )
    for src, err in s.get("failed") or []:
        _print(f"  failed to load {src}: {err}")
