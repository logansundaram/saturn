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
    # Compare BASENAMES, never an unanchored suffix: "my-notes.md".endswith("notes.md") is True,
    # so a pre-existing corrupt file in the corpus reported its loader error against the file
    # just added — and suppressed the success line for a document that embedded fine.
    err = next((e for src, e in failed.items() if Path(str(src)).name == path.name), None)
    if err is not None:
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
    "See, add, edit, and review the agent's persistent memory (the layered remember/recall store).",
    aliases=("mem",),
    usage="/memory [list [layer] | add [--layer L] [--replaces n] [--sens mark] <fact> | "
          "edit <n> <text> | forget <n> | why <n> | review [--no-llm] | pending | stale]",
    details="""
The transparency surface for durable memory. What is stored here quietly shapes every answer:
the user layer and open commitments load into the agent's context EVERY turn, the recent memo
notes do too, and agent / entities / negative facts load whenever they match the request
(/trace context shows the exact block a run got). This command shows and manages the store
without hand-editing database/memory/memory.md (still safe to hand-edit).

Layers:  user (identity, preferences, constraints) · commitments (open items, with a due date)
         · memo (dated notes) · agent (operating knowledge about this machine) · entities
         (people, projects, places, documents, your shorthand) · negative (what not to do)

  /memory                    every fact, grouped by layer, with its #id (also: list, ls)
  /memory list <layer>       one layer
  /memory add <fact>         save a fact (user layer; --layer <name> for another; --replaces <n>
                             retires fact n so a correction never sits beside the old fact;
                             --sens <mark> marks it sensitive: withheld from any prompt bound
                             for a remote inference host, e.g. --sens health)
  /memory edit <n> <text>    rewrite fact n in place (keeps its id and provenance)
  /memory forget <n>         delete fact n (any removal verb: forget/remove/rm/delete/del/drop;
                             `done <n>` reads better for a finished commitment)
  /memory why <n>            provenance: when it was learned, who said it (you, or inferred
                             at a review), the run it came from (→ /trace why #run), last use,
                             confirmations
  /memory review             the learning step: candidates this session queued — your mid-task
                             corrections, plan-review vetoes, gate denials, unfinished steps,
                             the compaction summary — plus the model's own proposals from the
                             transcript, each shown as a diff line and kept only on your y.
                             Also runs at /quit. --no-llm skips the model's proposals.
  /memory pending            what the review would show, without deciding
  /memory stale              by-match facts that have not matched a request in
                             memory.stale_days (flagged, never auto-deleted)

Examples:
  /memory add I prefer answers in metric units
  /memory add --layer entities "the deck" means Q3_investor_update.pptx
  /memory add --replaces 4 I live in Berlin now
  /memory add --sens health I take blood-pressure medication
  /memory review
""",
)
def _memory(ctx, args):
    from stores import memory_registry as mr
    from tui import ui

    usage = ("  usage: /memory [list [layer] | add [--layer L] [--replaces n] [--sens mark] <fact> "
             "| edit <n> <text> | forget <n> | why <n> | review [--no-llm] | pending | stale]")

    if not args or is_list_verb(args[0]):
        _list_memory(mr, ui, args[1] if len(args) > 1 else None)
        return

    sub = args[0].lower()
    if sub == "add":
        layer, replaces, sens, words = "user", None, None, []
        it = iter(args[1:])
        for a in it:
            low = a.lower()
            if low in ("--layer", "-l", "--in"):
                layer = next(it, "user")
            elif low in ("--replaces", "--replace", "-r"):
                replaces = next(it, None)
            elif low in ("--sens", "--sensitive", "--sensitivity", "-s"):
                sens = next(it, None)
            else:
                words.append(a)
        fact = " ".join(words).strip()
        if not fact:
            _print("  usage: /memory add [--layer <layer>] [--replaces <n>] [--sens <mark>] <fact>")
            return
        _print(f"  {mr.add_memory(fact, layer=layer, replaces=replaces, sensitivity=sens)}")
        return

    if sub == "edit":
        if len(args) < 3 or not _fact_id(args[1]):
            _print("  usage: /memory edit <n> <new text>   (the #id shown by /memory)")
            return
        old = mr.edit_memory(_fact_id(args[1]), " ".join(args[2:]))
        if old is None:
            _print(f"  no fact #{args[1]} (or empty text) — /memory lists the ids.")
        else:
            _print(f"  #{_fact_id(args[1])}: {old!r} → {' '.join(args[2:])!r}")
        return

    if is_remove_verb(sub) or sub == "done":
        if len(args) < 2 or not _fact_id(args[1]):
            _print("  usage: /memory forget <n>   (the #id shown by /memory)")
            return
        removed = mr.remove_memory(_fact_id(args[1]))
        if removed is None:
            _print(f"  no fact #{args[1]} — /memory lists {len(mr.entries())} fact(s).")
        else:
            _print(f"  {'done' if sub == 'done' else 'forgot'}: {removed}")
        return

    if sub in ("why", "show", "info"):
        if len(args) < 2 or not _fact_id(args[1]):
            _print("  usage: /memory why <n>")
            return
        _why(mr, ui, _fact_id(args[1]))
        return

    if sub == "review":
        review_pending(ctx, use_llm=not any(a.lower() in ("--no-llm", "--mechanical") for a in args))
        return

    if sub in ("pending", "queue", "candidates"):
        from core.memory_review import load_pending, render_line

        pending = load_pending()
        if not pending:
            ui.note("no memory candidates pending — they queue from your corrections, vetoes, "
                    "gate denials, unfinished steps and compaction summaries.")
            return
        ui.section("memory · pending review", f"{len(pending)} candidate(s) · /memory review decides")
        for c in pending:
            _print(f"  {render_line(c)}")
        return

    if sub == "stale":
        stale = [e for e in mr.entries() if mr.is_stale(e)]
        if not stale:
            ui.note(f"nothing stale — no by-match fact has gone {mr.stale_days()} days unmatched.")
            return
        ui.section("memory · stale", f"{len(stale)} fact(s) unmatched for {mr.stale_days()}+ days "
                   "· /memory forget <n> drops one (nothing is deleted on its own)")
        ui.table([((f"#{e['id']}", "accent"), e["layer"], _display_entry(e)) for e in stale])
        return

    _print(f"  unknown subcommand '{args[0]}'\n{usage}")


def _fact_id(token) -> int:
    t = str(token or "").strip().lstrip("#")
    return int(t) if t.isdigit() else 0


def _display_entry(e: dict) -> str:
    from stores.memory_registry import display

    return display(e)


def _list_memory(mr, ui, layer_filter=None):
    items = mr.entries(layer_filter) if layer_filter else mr.entries()
    if not items:
        if layer_filter:
            ui.note(f"nothing in the {mr.normalize_layer(layer_filter)} layer yet.")
        else:
            ui.note("no persistent memory yet — say `remember that ...` or use /memory add.")
        return
    from core.memory_review import load_pending

    n_pending = len(load_pending())
    pending_note = f" · {n_pending} candidate(s) pending review" if n_pending else ""
    ui.section(
        "memory",
        f"{len(items)} fact(s) · user + commitments + recent memo load every turn, the rest by "
        f"match · /memory why <n> for provenance{pending_note}",
    )
    if layer_filter:
        layers = [mr.normalize_layer(layer_filter)]
    else:  # the six standard layers first, then any section a hand edit / layer= introduced
        extra = sorted({e["layer"] for e in items} - set(mr.LAYERS))
        layers = list(mr.LAYERS) + extra
    for layer in layers:
        rows = [e for e in items if e["layer"] == layer]
        if not rows:
            continue
        _print(f"  {layer}")
        ui.table([
            (
                (f"#{e['id']}", "accent"),
                ("inferred" if e.get("by") == "inferred" else "", "dim"),
                ("stale" if mr.is_stale(e) else "", "dim"),
                _display_entry(e),
            )
            for e in rows
        ])


def _why(mr, ui, fact_id: int):
    e = mr.entry(fact_id)
    if e is None:
        _print(f"  no fact #{fact_id} — /memory lists the ids.")
        return
    ui.section(f"memory · #{fact_id}", _display_entry(e))
    who = ("you said it" if e.get("by") == "user"
           else "inferred (proposed at a review, accepted by you)")
    rows = [
        ("layer", e["layer"]),
        ("learned", e["date"]),
        ("said by", who),
        ("source run", f"#{e['run']}  → /trace why #{e['run']}" if e.get("run")
         else "none recorded (added by /memory add, or before runs were stamped)"),
        ("last used", e.get("used") or ("always loaded" if e["layer"] in ("user", "commitments")
                                        else "never matched a request yet")),
        ("confirmed", f"×{e.get('n', 1)}"),
    ]
    if e.get("sens"):
        rows.append(("sensitivity", f"{e['sens']} — withheld when inference is not local"))
    if e.get("due"):
        rows.append(("due", e["due"]))
    ui.table([((k, "dim"), v) for k, v in rows])


# The last transcript the model was asked to propose from (per command context): /memory review
# followed by /quit must not send the same session to the model twice.
_LAST_MODEL_PASS: dict = {}


def review_pending(ctx, *, use_llm: bool = True, on_quit: bool = False) -> None:
    """The review screen (core/memory_review.run_review) over the pending queue plus, when
    enabled, the model's proposals from the live transcript. Shared by `/memory review` and
    /quit. Nothing is written without a y; q leaves the remainder pending."""
    import sys

    from core import memory_review as rv
    from tui import ui

    pending = rv.load_pending()
    if not sys.stdin.isatty():
        # No screen, no review — and no model call spent proposing for one. The queue waits.
        if pending and not on_quit:
            _print(f"  {len(pending)} candidate(s) pending — the review needs an interactive "
                   "terminal; they stay queued.")
        return
    use_llm = use_llm and rv.llm_enabled()
    messages = (ctx.state or {}).get("messages") or []
    if use_llm and messages and (not on_quit or pending):
        # On /quit, the model pass only runs when something mechanical is already queued — a
        # quiet session must not pay a model call on its way out. And the same transcript is
        # never sent twice: a pass is recorded per (session, transcript length).
        mark = (id(ctx), len(messages))
        if mark != _LAST_MODEL_PASS.get("mark"):
            try:
                proposals = rv.llm_candidates(messages)
            except KeyboardInterrupt:
                proposals = []
                _print("  (model proposals skipped)")
            except Exception as exc:
                proposals = []
                _print(f"  (model proposals unavailable: {exc})")
            _LAST_MODEL_PASS["mark"] = mark
            if proposals:
                rv.add_pending(proposals)
                pending = rv.load_pending()
    if not pending:
        if not on_quit:
            ui.note("nothing to review — no memory candidates are pending.")
        return
    ui.section(
        "memory review",
        f"{len(pending)} candidate(s) · each is a proposed line for the memory file · "
        "y keep · n drop · e edit · a keep all · q stop (rest stay pending)",
    )
    # Ctrl-C / Ctrl-D at a prompt resolve to the review's own "stop" (the rest stays pending),
    # never to the empty reply a y/N prompt would read as "drop".
    result = rv.run_review(pending, ask=lambda prompt: ui.ask(prompt, on_interrupt=rv.INTERRUPT),
                           emit=_print)
    rv.save_pending(result["remaining"])
    kept, dropped, left = len(result["accepted"]), len(result["rejected"]), len(result["remaining"])
    summary = f"  kept {kept} · dropped {dropped}"
    if left:
        summary += f" · {left} still pending (/memory review)"
    _print(summary)


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

# The draft prompt lives in core/messages.py (INIT_DRAFT_PROMPT — the one-prompt-home rule),
# imported lazily at the call site below.


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
            from core.messages import INIT_DRAFT_PROMPT
            from stores.document_registry import read_workspace_manifest

            _print("  surveying the workspace and drafting SATURDAY.md…")
            prompt = INIT_DRAFT_PROMPT.format(
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
