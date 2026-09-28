"""Shared startup work for both entry paths (headless -p and the interactive REPL).

`startup_load` is the slow part of launch — the knowledge-base sync + graph build — run under
the splash animation interactively, or directly headless. The two warning shapers here exist so
startup problems surface as one readable line instead of a raw exception repr.
"""

import threading

import diag
from app.graph import build_agent

# RAG ingest (reconciles the disk-cached vector store the search_knowledge_base tool reads).
from stores.rag import iter_documents, sync


def startup_load(interactive: bool = True):
    """The slow startup loading (knowledge-base ingest + graph build). Returns
    `(graph, warning_or_None)`. Runs while the ring art animates in interactive mode, or
    directly (no TUI) in headless mode."""
    warn = None
    # Read the hardware once per launch (chip / RAM / VRAM) so /models — and the first-launch
    # tier pick — price the ladder against a cached profile instead of re-probing. Interactive
    # only: headless (-p) never renders /models, and the probe spawns sysctl / nvidia-smi (up to
    # its 3 s timeout on a waking driver). Wrapped: probe() never raises, but a launch must not
    # depend on that.
    if interactive:
        try:
            from core.hardware import profile

            profile()
        except Exception:
            pass
    # Reconcile the knowledge base against the disk cache at startup: only new/changed
    # documents are embedded, the rest load from the persisted store. An EMPTY corpus (the
    # fresh-install state — the embedder is pulled lazily on the first /docs add) skips the
    # sync entirely, so a launch never touches the embedder it may not have. Non-fatal if it
    # fails — search_knowledge_base just returns "no documents"; the warning is shaped by
    # _ingest_warning (one line, daemon-down and embedder-missing stated plainly).
    try:
        if any(True for _ in iter_documents()):
            missing = embedder_missing()
            if missing:
                warn = (f"knowledge base not synced: the embedder {missing} is not pulled — "
                        "`/docs sync` pulls it on consent")
            else:
                sync(verbose=False)
    except Exception as exc:
        warn = _ingest_warning(exc, interactive=interactive)
    return build_agent(), warn


def embedder_missing() -> "str | None":
    """The active tier's embedder tag when the daemon is reachable and has NOT pulled it; None
    when it is pulled, the daemon is down (the sync's own error says so), or the check fails."""
    try:
        from config import get_config
        from core.llms import _model_present, list_local_models, ollama_reachable

        tag = get_config().embedder_model
        local = list_local_models()
        if not local and not ollama_reachable():
            return None
        return None if _model_present(tag, {m.name for m in local}) else tag
    except Exception:
        return None


def warm_model(role: str = "tool_caller") -> bool:
    """Load `role`'s model into the daemon with ONE minimal request, so the session's first turn
    does not pay the weight load inside its first agent call (measured 2026-09-02: 50 s and 37 s
    for a cold "hello" against 15 s warm — the difference between an agent that looks hung and
    one that answers). The request rides the same `num_ctx` every turn uses: Ollama keys the
    loaded runner on the context size, so warming at another window would load a runner the
    first turn then evicts. Every role on a tier binds the same model, so one role suffices.
    Never raises — a down daemon is the health check's report, not this one's."""
    from langchain.messages import HumanMessage

    from core.llms import generate, get_model
    from core.structured import _invoke_kwargs, _model_tag

    try:
        kwargs = _invoke_kwargs(role, None, 0.0, task="agent")
        kwargs.setdefault("options", {})["num_predict"] = 1
        generate(get_model(role), [HumanMessage(content="ok")], tag=_model_tag(role), **kwargs)
        return True
    except Exception as exc:
        diag.log(f"startup: model warm-up skipped ({exc})")
        return False


def _warm_and_prime(role: str) -> None:
    """The warm-up thread's body: load the weights, then plant the agent's prefix checkpoint
    (core/prime.py) so the first turn's first call prefills only its request."""
    warm_model(role)
    from core import prime

    prime.prime_now(only=("agent",))


def start_warm_up(role: str = "tool_caller") -> threading.Thread:
    """`warm_model` + the agent prime on a daemon thread: the REPL keeps starting while the
    weights load, and a first query typed early simply queues behind the load at the daemon —
    as it did before, minus the second load it used to pay."""
    t = threading.Thread(target=_warm_and_prime, args=(role,), name="model-warm-up", daemon=True)
    t.start()
    return t


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
