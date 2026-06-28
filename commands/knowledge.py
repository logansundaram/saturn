"""
Knowledge & workspace commands — what the agent knows and where it works, in one module (the
/help "knowledge & workspace" theme; consolidated from one-file-per-command 2026-06-11):

  /docs    the RAG corpus + workspace file listing (add/remove/sync)
  /memory  the durable remember/recall facts
  /init    survey the workspace and draft SATURDAY.md
  /undo    revert the last turn's file changes (pre-write snapshots)
"""

import time
from pathlib import Path

from commands._framework import command, _print
from commands._utils import is_list_verb, is_remove_verb


# ── /docs ────────────────────────────────────────────────────────────────────────────────────
@command(
    "docs",
    "View and manage the RAG corpus (and see workspace files): /docs add | remove | sync.",
    aliases=("documents",),
    usage="/docs [list | add <path> | remove <name> | sync [--force]]",
    details="""
The one front door to the document knowledge base (what search_knowledge_base retrieves from),
plus a view of the workspace sandbox files (where the file tools read/write).

  /docs                 list the ingested corpus + the workspace files (also: list, ls)
  /docs add <path>      copy a file into the corpus and embed it (txt/md/pdf/html/csv/docx).
                        Paths with spaces don't need quoting; a dragged file's quoted path
                        works as-is (tip: type `/docs add ` then drag the file onto the
                        terminal).
                        A no-op if the file is already present and unchanged (content hash).
  /docs remove <name>   remove a document: drops its vectors and manifest entry (any removal
                        verb works: remove/rm/delete/del/forget/drop)
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
    if is_list_verb(sub):
        _list_docs()
    elif sub == "add":
        _add(rest)
    elif is_remove_verb(sub):
        _remove(rest)
    elif sub == "sync":
        _sync(force=any(a in ("--force", "-f", "force") for a in rest))
    else:
        _print(f"  unknown subcommand '{args[0]}' — usage: /docs [list | add <path> | remove <name> | sync [--force]]")


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
    from stores.rag import ingest_file, screen_file, SUPPORTED_EXTENSIONS
    from tui import ui

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
    # Corpus admission screen: an ingested document re-presents its content on EVERY future
    # search, so instruction-shaped content gets one human look before it's let in. Default no
    # (bare Enter / Ctrl-C refuses) — the same fail-closed default as the approval gate.
    findings = screen_file(str(path))
    if findings:
        kinds = ", ".join(sorted({f.kind for f in findings}))
        ui.warn(f"{path.name} contains instruction-shaped content ({kinds}):")
        for f in findings[:3]:
            _print(f"      · {f.preview}")
        if len(findings) > 3:
            _print(f"      · … +{len(findings) - 3} more")
        _print("    retrieved chunks are quarantined as untrusted, but the text will reach the")
        _print("    model on every search that matches it.")
        if ui.ask("ingest anyway? [y/N] ").lower() not in ("y", "yes"):
            _print("  not ingested.")
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
    for src, kinds in s.get("flagged") or []:
        ui.warn(f"{src}: instruction-shaped content ({', '.join(kinds)}) — "
                "its search results are quarantined as untrusted")


# ── /memory ──────────────────────────────────────────────────────────────────────────────────
@command(
    "memory",
    "List, add, or delete the agent's persistent memory (the remember/recall facts).",
    aliases=("mem",),
    usage="/memory [list | add <fact> | forget <n>]",
    details="""
The transparency surface for durable memory: the facts saved via the `remember` tool (or your
own "remember that ..." requests) are loaded into the agent's context EVERY turn, so what is
stored here quietly shapes every answer. This command lets you see and manage that store
without hand-editing database/memory/memory.md.

  /memory               numbered list of every stored fact (also: list, ls)
  /memory add <fact>    save a fact directly (same dedup as the remember tool)
  /memory forget <n>    delete fact n (the number shown by /memory; any removal verb works:
                        forget/remove/rm/delete/del/drop)

The store is a plain markdown file (paths.memory in config.yaml) — still safe to hand-edit;
this is just the in-app view of it.

Examples:
  /memory
  /memory add I prefer answers in metric units
  /memory forget 3
""",
)
def _memory(ctx, args):
    from stores.memory_registry import add_memory, list_memory, remove_memory
    from tui import ui

    if not args or is_list_verb(args[0]):
        facts = list_memory()
        if not facts:
            ui.note("no persistent memory yet — say `remember that ...` or use /memory add.")
            return
        ui.section(
            "memory",
            f"{len(facts)} fact(s) · loaded into context every turn · /memory forget <n> deletes",
        )
        ui.table([((f"{i}", "accent"), fact) for i, fact in enumerate(facts, start=1)])
        return

    sub = args[0].lower()
    if sub == "add":
        fact = " ".join(args[1:]).strip()
        if not fact:
            _print("  usage: /memory add <fact>")
            return
        _print(f"  {add_memory(fact)}")
        return

    if is_remove_verb(sub):
        if len(args) < 2 or not args[1].isdigit():
            _print("  usage: /memory forget <n>   (the number shown by /memory)")
            return
        removed = remove_memory(int(args[1]))
        if removed is None:
            n = len(list_memory())
            _print(f"  no fact #{args[1]} — /memory lists {n} fact(s).")
        else:
            _print(f"  forgot: {removed}")
        return

    _print(f"  unknown subcommand '{args[0]}' — usage: /memory [list | add <fact> | forget <n>]")


# ── /init ────────────────────────────────────────────────────────────────────────────────────
# How much workspace evidence to show the drafting model.
_MAX_LISTING = 100

# Written when the workspace is empty or the LLM draft fails — still useful: the file's existence
# (and its section headings) teaches the user what to put there.
_TEMPLATE = """# SATURDAY.md

Standing instructions for this workspace. Saturday loads this file into context at the start of
every turn — keep it short and current.

## What this workspace is for

(Describe the project/notes/files that live here and what you're trying to do with them.)

## Conventions

- (e.g. "drafts live in drafts/, finished pieces in posts/")
- (e.g. "always write dates as YYYY-MM-DD")

## Things to remember

- (standing guidance: tone, formats, what to never touch, who this work is for)
"""

_DRAFT_PROMPT = """You are initializing SATURDAY.md — a standing-instructions file that a local
AI agent loads into context at the start of every turn it works in this workspace.

Below are the workspace's file listing and (when available) one-line summaries of its files.
Write a concise SATURDAY.md (under 60 lines) in markdown with exactly these sections:

# SATURDAY.md
## What this workspace is for      (1-3 sentences inferred from the files)
## Layout                          (the notable files/folders and what each holds — only what you
                                    can actually infer; skip boilerplate)
## Conventions                     (any naming/format patterns visible in the files; if none are
                                    evident, give 1-2 sensible placeholders the user can edit)

Be factual about what you can see and explicit about what you're guessing. Do not invent files.
Output ONLY the markdown file content, no preamble.

## File listing
{listing}

## File summaries
{summaries}
"""


def _workspace_listing(workspace: Path) -> list[str]:
    """Workspace-relative paths, capped. Best-effort — unreadable entries are skipped."""
    out = []
    try:
        for p in sorted(workspace.rglob("*")):
            if len(out) >= _MAX_LISTING:
                out.append("… (listing capped)")
                break
            try:
                rel = p.relative_to(workspace).as_posix()
            except ValueError:
                continue
            out.append(rel + ("/" if p.is_dir() else ""))
    except OSError:
        pass
    return out


@command(
    "init",
    "Survey the workspace and draft SATURDAY.md (standing per-workspace instructions).",
    usage="/init [--force]",
    details="""
The workspace is Saturn's sandboxed working area — the directory the file tools read and write,
at paths.workspace in config.yaml (database/workspace under the install by default). It is NOT
the directory you launched Saturn from, and /init never touches your current directory. To get
real files into Saturn's view: ingest them into the knowledge base with /docs add <path>, drop a
file onto the prompt (drag-and-drop offers ingest/attach), or copy them into the workspace.

/init creates SATURDAY.md at that workspace root — the per-workspace instructions file (the
CLAUDE.md equivalent). The grounding node loads it into context EVERY turn, so whatever it says
is standing guidance for the agent: what this workspace is for, its layout, your conventions.

/init surveys the workspace (file listing + the manifest's file summaries) and drafts the file
with the utility model; if the workspace is empty or the model is unavailable, it writes a
sensible template instead. Either way: open it and edit — it's your file, the draft is a start.

Refuses to overwrite an existing SATURDAY.md unless --force is passed.
""",
)
def _init(ctx, args):
    from config import get_config

    force = any(a in ("--force", "-f") for a in args)
    workspace = get_config().path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "SATURDAY.md"
    if target.exists() and not force:
        _print(f"  SATURDAY.md already exists at {target} — edit it directly, or re-draft "
               "with /init --force.")
        return

    listing = _workspace_listing(workspace)
    content = None
    # Only worth an LLM call when there is something to look at; an empty workspace gets the
    # template, which explains itself better than a model guessing at nothing.
    if [e for e in listing if e != "SATURDAY.md"]:
        try:
            from langchain.messages import HumanMessage
            from core.llms import get_model
            from stores.document_registry import read_workspace_manifest

            _print("  surveying the workspace and drafting SATURDAY.md…")
            prompt = _DRAFT_PROMPT.format(
                listing="\n".join(listing) or "(empty)",
                summaries=read_workspace_manifest().strip() or "(none)",
            )
            draft = str(get_model("utility").invoke([HumanMessage(content=prompt)]).content).strip()
            # Models love to wrap file output in a code fence — unwrap it.
            if draft.startswith("```"):
                lines = draft.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                draft = "\n".join(lines).strip()
            if draft.startswith("#"):
                content = draft + "\n"
        except Exception as exc:
            _print(f"  draft failed ({exc}) — writing the template instead.")

    target.write_text(content or _TEMPLATE, encoding="utf-8")
    kind = "drafted from the workspace contents" if content else "template"
    # Full absolute path on purpose: the workspace is Saturn's sandboxed area, not the cwd a
    # terminal user expects — a bare basename here left people unable to find the file they
    # were just told to edit.
    _print(f"  wrote {target} ({kind}).")
    _print("  this is Saturn's sandboxed workspace, not your current directory.")
    _print("  it now loads into context every turn — open it and make it yours.")


# ── /undo ────────────────────────────────────────────────────────────────────────────────────
@command(
    "undo",
    "Revert the file changes the last turn made to the workspace.",
    usage="/undo [list]",
    details="""
Restores the workspace files touched by the most recent turn that wrote anything, using the
pre-write snapshots taken automatically by write_file / edit_file (stores/snapshots.py). A file
the turn created is deleted; a file it overwrote or edited is restored to its turn-start bytes.
Each /undo pops one batch, so repeating it walks further back (up to the retained history).

  /undo         revert the most recent batch of file changes
  /undo list    show the stored snapshot batches (newest first) without restoring

Scope: only the file tools snapshot. run_shell can touch anything, so its effects are NOT
undoable — the approval gate showing the exact command is its safety boundary. The conversation
itself is not rewound, only the files.
""",
)
def _undo(ctx, args):
    from stores import snapshots

    if args and is_list_verb(args[0]):
        batches = snapshots.list_batches()
        if not batches:
            _print("  no snapshots stored — no turn has written to the workspace yet.")
            return
        _print(f"  {len(batches)} snapshot batch(es), newest first:")
        for i, b in enumerate(batches, 1):
            query = f' — "{b["query"]}"' if b["query"] else ""
            _print(f"    {i}. {b['created'] or b['id']}{query}")
            for path in b["files"]:
                _print(f"         {path}")
        _print("  /undo restores #1 (each /undo pops one batch).")
        return

    # Anything unrecognized REFUSES — never falls through to the revert. /undo is destructive
    # with no redo, so a typo'd listing attempt ('/undo lst', '/undo 2', '/undo show') must
    # error, not silently restore files (the /mcp typo'd-'relod' rule, with higher stakes).
    if args:
        _print(f"  unknown argument {args[0]!r} — usage: /undo [list]  "
               "(bare /undo reverts the last writing turn)")
        return

    try:
        summary, actions = snapshots.undo_last()
    except RuntimeError as exc:
        _print(f"  {exc}")
        return
    _print(f"  undid file changes from turn {summary}:")
    for line in actions:
        _print(f"    {line}")
    if not actions:
        _print("    (the batch recorded no file changes)")
