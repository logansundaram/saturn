"""Shared startup work for both entry paths (headless -p and the interactive REPL).

`startup_load` is the slow part of launch — the knowledge-base sync + graph build — run under
the splash animation interactively, or directly headless. The two warning shapers here exist so
startup problems surface as one readable line instead of a raw exception repr.
"""

from app.graph import build_agent

# RAG ingest (reconciles the disk-cached vector store the search_knowledge_base tool reads).
from stores.rag import sync


def startup_load(interactive: bool = True):
    """The slow startup loading (knowledge-base ingest + graph build). Returns
    `(graph, warning_or_None)`. Runs while the ring art animates in interactive mode, or
    directly (no TUI) in headless mode."""
    warn = None
    # Reconcile the knowledge base against the disk cache at startup: only new/changed
    # documents are embedded, the rest load from the persisted store. Non-fatal if it fails
    # (e.g. embedding model not pulled) — search_knowledge_base just returns "no documents";
    # the warning is shaped by _ingest_warning (one line, daemon-down stated plainly).
    try:
        sync(verbose=False)
    except Exception as exc:
        warn = _ingest_warning(exc, interactive=interactive)
    return build_agent(), warn


def _ingest_warning(exc: Exception, *, reachable: "bool | None" = None,
                    interactive: bool = True) -> str:
    """One readable line for a failed startup knowledge-base ingest (non-fatal: the agent runs on
    without RAG). The common first-launch cause is the Ollama daemon being down — the embedder
    can't run — and the model health check that prints moments later already explains exactly
    that, so this line says it plainly and defers to it instead of dumping a multi-line httpx
    ConnectError repr right above the clean explanation of the same root cause. Headless (-p)
    prints no health check, so the deferral clause is dropped there. Any other failure keeps its
    exception, collapsed to one line. `reachable` overrides the live llms.ollama_reachable()
    probe (offline tests)."""
    if reachable is None:
        from core.llms import ollama_reachable

        reachable = ollama_reachable()
    if not reachable:
        return "knowledge-base ingest skipped (Ollama not reachable" + (
            " — the model check below explains)" if interactive else ")"
        )
    from textutil import clip

    detail = clip(exc, 300) or exc.__class__.__name__
    return f"knowledge-base ingest failed, continuing without RAG: {detail}"


def _warn_flagged_attachments(block: str, emit) -> None:
    """Attachment admission warning — @file mentions and piped stdin attach the user's OWN files,
    but their CONTENT often isn't the user's words (a downloaded PDF, a vendored README, a piped
    log). Instruction-shaped content gets one warning naming the patterns, never a block: the
    human chose to attach it; the point is that they KNOW what rode in with it. `emit` is the
    output channel (ui.warn interactively, stderr headless)."""
    try:
        from trust import quarantine

        if not block or not quarantine.active():
            return
        kinds = sorted({f.kind for f in quarantine.scan(block)})
        if kinds:
            emit(f"attachment contains instruction-shaped content ({', '.join(kinds)}) — "
                 f"the model sees it as data; watch the plan and gate for actions you didn't ask for")
    except Exception:
        pass  # a warning helper must never cost the turn
