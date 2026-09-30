"""
Model factory — `get_model()` instead of hard-coded globals.

Each hardware tier in `config.yaml` binds ONE chat model (the agent's pass and the background
calls — compaction, the memory review, /init — share it; the `utility` role collapsed into it
2026-09-30). This module resolves it against the active tier and builds the LangChain chat
model. Swapping hardware is a config edit; graph code never names a model.

Ollama is the only backend: nothing leaves the machine to compute the words (cloud providers
were shelved 2026-07-03 and cut 2026-09-27). The one network boundary that remains is a REMOTE
`OLLAMA_HOST`: `_NetworkBoundaryModel` wraps that daemon so every call is recorded
to the egress ledger and refused under air-gap. Built models are cached per model id;
`reset_models()` clears the cache after a live model change (the `/models` command).

Capability descriptors come from config; the loop requires native tool-calling, and we warn
(not crash) if the bound model does not advertise it.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings

import diag

from trust import egress
from config import get_config


def _approx_bytes(messages) -> int:
    """Approximate the size of what a model call sends off-machine — the char count of every
    message's string content. Best-effort: a non-list / odd shape just reads as 0."""
    if not isinstance(messages, list):
        return 0
    total = 0
    for m in messages:
        c = getattr(m, "content", None)
        if isinstance(c, str):
            total += len(c)
    return total


class _NetworkBoundaryModel:
    """Thin proxy around an off-machine chat model — an Ollama daemon behind a remote
    OLLAMA_HOST, via `_wrap_ollama` — that makes the network boundary observable + safe. Every
    call through it records the egress to the ledger (`egress.record`) — what left, where to,
    how big. Everything else (bind_tools, with_structured_output, attribute access) delegates to the inner
    model and re-wraps any derived runnable so the boundary survives `.bind_tools(...)` /
    `.with_structured_output(...)`. LOOPBACK Ollama models are never wrapped — there is no
    boundary."""

    def __init__(self, inner, model: str, host: str):
        self._inner = inner
        self._model = model
        self._host = host

    def _outgoing(self, messages):
        """Record the egress; return the messages to send (unchanged)."""
        egress.record("llm", self._host, self._model, provider="ollama",
                      n_bytes=_approx_bytes(messages))
        return messages

    def invoke(self, input, *args, **kwargs):
        return self._inner.invoke(self._outgoing(input), *args, **kwargs)

    def stream(self, input, *args, **kwargs):
        return self._inner.stream(self._outgoing(input), *args, **kwargs)

    async def ainvoke(self, input, *args, **kwargs):
        return await self._inner.ainvoke(self._outgoing(input), *args, **kwargs)

    async def astream(self, input, *args, **kwargs):
        async for chunk in self._inner.astream(self._outgoing(input), *args, **kwargs):
            yield chunk

    def batch(self, inputs, *args, **kwargs):
        # Through invoke one input at a time so EVERY input is recorded — the inner
        # model's batch would take the whole list past the boundary in one unobserved call.
        return [self.invoke(i, *args, **kwargs) for i in inputs]

    async def abatch(self, inputs, *args, **kwargs):
        return [await self.ainvoke(i, *args, **kwargs) for i in inputs]

    def bind_tools(self, *args, **kwargs):
        return _NetworkBoundaryModel(
            self._inner.bind_tools(*args, **kwargs), self._model, self._host
        )

    def with_structured_output(self, *args, **kwargs):
        return _NetworkBoundaryModel(
            self._inner.with_structured_output(*args, **kwargs), self._model, self._host
        )

    # Network entry points this proxy does NOT cover fail CLOSED: __getattr__ used to hand them
    # back bound to the INNER model, so a future caller (or a LangChain runnable composition)
    # would send unrecorded content — the exact leak the boundary exists to prevent.
    # Nothing in the repo calls these today; a new caller gets a loud pointer, never a bypass.
    _UNGUARDED = frozenset({
        "generate", "agenerate", "generate_prompt", "agenerate_prompt",
        "transform", "atransform", "batch_as_completed", "abatch_as_completed",
    })

    def __getattr__(self, name):
        if name in _NetworkBoundaryModel._UNGUARDED:
            raise AttributeError(
                f"_NetworkBoundaryModel does not expose {name!r}: it would bypass the "
                "egress boundary — use invoke/stream/astream/batch instead"
            )
        # Anything else we don't override (get_name, config_specs, etc.) defers to the inner model.
        return getattr(self._inner, name)


def _ollama_client_kwargs() -> dict:
    """client_kwargs for ChatOllama carrying the request timeout (forwarded to the underlying
    httpx client). A short connect timeout fails fast when the daemon is DOWN; a generous read
    timeout (runtime.llm_timeout) bounds a WEDGED daemon without false-tripping slow-but-healthy
    generation. Empty dict when the timeout is disabled — no behavioural change from before."""
    t = get_config().llm_timeout
    if not t:
        return {}
    return {"client_kwargs": {"timeout": httpx.Timeout(t, connect=min(10.0, t))}}

# model id -> BaseChatModel.  Cleared by reset_models().
_MODEL_CACHE: dict[str, object] = {}


def _wrap_ollama(m, model: str):
    """Loopback Ollama is handed back bare — there is no boundary to guard. A REMOTE Ollama
    (OLLAMA_HOST pointing off-machine) IS one: wrap it in the network boundary proxy so every
    call is checked against the air-gap and recorded to the egress ledger with the real
    endpoint as the host — 'local model' must never silently mean 'someone else's machine'."""
    if egress.ollama_is_local():
        return m
    return _NetworkBoundaryModel(m, model, host=f"ollama @ {egress.ollama_endpoint()}")


def _build(model: str):
    # Bind num_ctx to the effective window (runtime.num_ctx override, else the model's declared
    # window) so it actually runs at the size the UI gauges against — Ollama otherwise silently
    # caps at 2048, making the context-fill % lie. Setting runtime.num_ctx drops the cache to rebind live.
    # client_kwargs carries the request timeout (guards a wedged daemon; see _ollama_client_kwargs).
    # keep_alive rides every request (runtime.keep_alive): the daemon's default unloads the
    # weights after five idle minutes and the next turn pays the whole load again.
    cfg = get_config()
    keep = cfg.keep_alive
    return _wrap_ollama(
        ChatOllama(
            model=model,
            num_ctx=cfg.num_ctx_for(model),
            **({"keep_alive": keep} if keep is not None else {}),
            **_ollama_client_kwargs(),
        ),
        model,
    )


def get_model():
    """Return the active tier's chat model (cached).

    Air-gap enforcement for a remote OLLAMA_HOST lives here (not in a wrapper) because a cached
    remote handle would otherwise sneak a call through after the gate engaged. `/policy airgap`
    drops the cache so this re-checks."""
    model = get_config().chat_model
    if not egress.ollama_is_local():
        # An off-machine OLLAMA_HOST makes the "local" model network egress — through the one
        # gate so the blocked attempt always reaches the ledger.
        egress.check_or_raise("llm", f"ollama @ {egress.ollama_endpoint()}", model,
                              provider="ollama", subject=f"the model ({model})")
    if model not in _MODEL_CACHE:
        _MODEL_CACHE[model] = _build(model)
    return _MODEL_CACHE[model]


def model_id() -> str:
    """The active tier's chat model id (for display: banners, /model)."""
    return get_config().chat_model




class _EmbeddingsBoundary:
    """OllamaEmbeddings against a REMOTE daemon — the embedding twin of _NetworkBoundaryModel.
    Every batch checks the air-gap first (raising, since an embedder can't hand back a refusal
    string) and records the egress: corpus text leaving for another machine must show in the
    ledger like any other send. Loopback embeddings are never wrapped."""

    def __init__(self, inner, model: str, host: str):
        self._inner, self._model, self._host = inner, model, host

    def _gate(self, texts) -> None:
        egress.check_or_raise("embedding", self._host, self._model, provider="ollama",
                              subject="embedding document text")
        egress.record("embedding", self._host, self._model, provider="ollama",
                      n_bytes=sum(len(t) for t in texts if isinstance(t, str)))

    def embed_documents(self, texts):
        self._gate(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        self._gate([text])
        return self._inner.embed_query(text)

    async def aembed_documents(self, texts):
        # Without this override the Embeddings base-class async default runs against the INNER
        # object (self=inner), skipping the air-gap raise and the ledger entirely.
        self._gate(texts)
        return await self._inner.aembed_documents(texts)

    async def aembed_query(self, text):
        self._gate([text])
        return await self._inner.aembed_query(text)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_embeddings():
    """Embedding model for the RAG store (the `embedder` slot of the active tier). Behind a
    remote OLLAMA_HOST it comes back wrapped in the egress/air-gap boundary — document text
    crossing the network is egress, exactly like a remote chat call."""
    inner = OllamaEmbeddings(model=get_config().embedder_model)
    if egress.ollama_is_local():
        return inner
    return _EmbeddingsBoundary(inner, get_config().embedder_model,
                               f"ollama @ {egress.ollama_endpoint()}")


def reset_models() -> None:
    """Drop all cached models so the next get_* call rebuilds from current config. Called
    after a live model/tier change (e.g. the /models slash command)."""
    _MODEL_CACHE.clear()


# ── local (Ollama) model discovery ────────────────────────────────────────────
# `/models` pings the Ollama daemon for what's actually pulled on this machine so the picker
# lists real, runnable tags (not just whatever config.yaml names). Kept here in the factory
# module — it's the one place that already owns "which models exist / can we build them".


@dataclass(frozen=True)
class LocalModel:
    """A model pulled into the local Ollama daemon, as surfaced by `ollama list`."""

    name: str            # the tag you bind (e.g. "qwen3.5:4b")
    size_bytes: int      # on-disk size
    parameter_size: str  # e.g. "4B", "29.9B" ("" if Ollama didn't report it)
    quantization: str    # e.g. "Q4_K_M" ("" if absent)
    family: str          # e.g. "gemma", "glm4moelite" ("" if absent)
    is_embedding: bool   # heuristic: an embed-only model (can't serve a chat role)


def _looks_like_embedder(name: str, family: str, families) -> bool:
    """Best-effort: Ollama's tag list doesn't flag embed-only models, so sniff the name/family.
    Used only to group the picker (embedders bind the `embedder` slot, not a chat role)."""
    hay = " ".join([name, family or "", " ".join(families or [])]).lower()
    return any(tok in hay for tok in ("embed", "bert", "e5", "bge", "gte"))


def _field(obj, *names, default=None):
    """First present field of `obj` among `names`, tolerating both attribute and mapping shapes
    (the `ollama.list()` response has shipped as either across versions)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(n)
        if v is not None:
            return v
    return default


def list_local_models() -> list[LocalModel]:
    """Return the models pulled into the local Ollama daemon (sorted by name).

    Best-effort: returns [] if the `ollama` package is missing or the daemon is unreachable —
    callers degrade to config-only behaviour rather than crashing. Reads the typed
    `ollama.list()` response, tolerating both attribute and mapping shapes across versions."""
    try:
        import ollama

        resp = ollama.list()
    except Exception:
        return []

    raw = getattr(resp, "models", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("models", [])
    out: list[LocalModel] = []
    for m in raw or []:
        name = _field(m, "model", "name", default="") or ""
        if not name:
            continue
        details = _field(m, "details", default=None)
        family = _field(details, "family", default="") or "" if details is not None else ""
        families = _field(details, "families", default=[]) if details is not None else []
        out.append(
            LocalModel(
                name=name,
                size_bytes=int(_field(m, "size", default=0) or 0),
                parameter_size=(_field(details, "parameter_size", default="") or "")
                if details is not None else "",
                quantization=(_field(details, "quantization_level", default="") or "")
                if details is not None else "",
                family=family,
                is_embedding=_looks_like_embedder(name, family, families),
            )
        )
    return sorted(out, key=lambda lm: lm.name.lower())


# ── startup health check ──────────────────────────────────────────────────────
# Surfaces a missing daemon / un-pulled model at STARTUP with an actionable
# message, instead of letting it surface as a generic turn failure on the first real query.


def ollama_reachable() -> bool:
    """True if the local Ollama daemon answers. Distinguishes 'daemon down' from 'no models
    pulled' (both make list_local_models return [])."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def _model_present(required: str, have: set[str]) -> bool:
    """Whether a required model tag is among the pulled ones, tolerating the implicit ':latest'
    tag Ollama adds (so 'qwen3.5:9b' and a bare 'mymodel' both match correctly)."""
    def _norm(n: str) -> str:
        return n if ":" in n else f"{n}:latest"

    return _norm(required) in {_norm(h) for h in have}


def check_models() -> list[str]:
    """Startup health report for the active tier. Returns a list of human-readable PROBLEM strings
    (empty when all is well): the Ollama daemon being down or model tags not pulled. Non-fatal — `agent.main` prints these as
    warnings and continues (a degraded tier still runs the commands/REPL; the first affected turn
    fails cleanly rather than the app refusing to start)."""
    cfg = get_config()
    problems: list[str] = []

    need_ollama: list[str] = []
    try:
        need_ollama.append(cfg.chat_model)
    except KeyError as exc:
        # A {provider, model} mapping from a pre-cut config, or a tier without a model.
        problems.append(exc.args[0] if exc.args else str(exc))

    try:
        # The embedder is only required once the knowledge base holds a document: it is pulled
        # lazily by the first /docs add, so an empty corpus must not report it missing.
        from stores.rag import iter_documents

        if any(True for _ in iter_documents()):
            need_ollama.append(cfg.embedder_model)
    except KeyError as exc:
        # A tier without an `embedder:` (no hard-coded fallback id — config.yaml is the one
        # home for model ids) is a health-report problem, not a startup crash. args[0], not
        # str(exc): str() of a KeyError is the repr of its message (spurious quotes).
        problems.append(exc.args[0] if exc.args else str(exc))
    need_ollama = sorted(set(need_ollama))

    if need_ollama:
        local = list_local_models()
        have = {m.name for m in local}
        if not local and not ollama_reachable():
            problems.append(
                "Ollama daemon not reachable — start it with `ollama serve`, then pull: "
                + ", ".join(need_ollama)
            )
        else:
            for m in need_ollama:
                if not _model_present(m, have):
                    problems.append(f"model not pulled: `{m}`  →  run `ollama pull {m}`")

    # The capability advisory. It used to print lazily on a model's first use (mid-turn,
    # colliding with the live TUI); surfacing it here puts it next to the other startup
    # warnings with the rest of the health report.
    try:
        model = cfg.chat_model
    except KeyError:
        model = ""  # already reported above
    if model and not cfg.capability_of(model).supports_tools:
        problems.append(f"model `{model}` does not advertise native tool-calling — the agent "
                        "loop may misbehave")

    return problems


# ── the think-rejection fallback (from the engine isolate, 2026-08-15) ─────────────────────────
#
# Model tags whose daemon rejected a `think` parameter. A model without a thinking template 400s
# on `think` in EITHER direction, so the engine cannot express "no rationale please" to it — it
# can only stop asking. Learned once per tag per process, never guessed from the name;
# `invoke_kwargs` consults it and omits the flag for such tags.
_NO_THINK_SUPPORT: set = set()

_THINK_REJECTION_MARKERS = ("does not support thinking", "thinking is not supported", '"think"')


def _is_think_rejection(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return any(m in text for m in _THINK_REJECTION_MARKERS)


# ── the per-call decoding options (moved from core/structured.py, 2026-09-29) ───────────────────
# The output-token bound per task — a circuit breaker well above a healthy generation, so a
# repetition loop lands as a truncated draw instead of a full window. `agent` is the loop's one
# call (nodes/agent.py): prose OR a tool call, so it must fit a write_file payload (4096 tokens
# is ~12-16 KB of text).
NUM_PREDICT: dict = {"agent": 4096}


def model_tag() -> str:
    """The active chat model id, '' when the binding can't be read."""
    try:
        return get_config().chat_model
    except Exception:
        return ""


def invoke_kwargs(fmt: "dict | None", temp: float, task: "str | None" = None, *,
                  think: bool = False) -> dict:
    """THE builder of the options every model call sends: constrained decoding (`fmt`), the
    temperature and the per-TASK decisions ride the invoke kwargs (ChatOllama forwards
    `format`/`options`/`reasoning` to the daemon).

    The options dict must carry `num_ctx` too: langchain_ollama treats an invoke-time `options`
    as a FULL REPLACEMENT for the constructor-built options (which is the only place the
    configured context window lives), so temperature alone would silently revert the daemon to
    its ~2048 default and front-truncate long prompts. A task also carries its `num_predict`
    bound, and `reasoning` (think) is set EXPLICITLY OFF — never the model's default — unless
    the daemon already rejected the flag for this tag (`_NO_THINK_SUPPORT`). `think=True` is
    the agent's adaptive thinking pass (nodes/agent.py): the flag goes ON and the task's
    `num_predict` widens by `runtime.think_budget`, since thinking tokens count against it."""
    options: dict = {"temperature": temp}
    tag = model_tag()
    try:
        cfg = get_config()
        options["num_ctx"] = cfg.num_ctx_for(cfg.chat_model)
    except Exception:  # a broken binding must not fail the call that would surface it
        pass
    if task is not None:
        options["num_predict"] = NUM_PREDICT.get(task, 512)
        if think:
            options["num_predict"] += max(0, int(get_config().get("runtime.think_budget", 4096) or 0))
    kwargs: dict = {"options": options}
    if task is not None and tag not in _NO_THINK_SUPPORT:
        kwargs["reasoning"] = bool(think)
    if fmt is not None:
        kwargs["format"] = fmt
    return kwargs


def generate(runnable, messages, *, tag: str = "", **kwargs):
    """`runnable.invoke(messages, **kwargs)` with ONE structural fallback: a daemon that rejects
    the `reasoning`(think) flag gets the call retried without it and `tag` is remembered in
    `_NO_THINK_SUPPORT`. That must not be the difference between an engine that works against
    a thinking model and one that raises against a plain one — and it must not be papered over
    by dropping the flag globally (every thinking model would fall back to its default: ON)."""
    try:
        return runnable.invoke(messages, **kwargs)
    except Exception as exc:
        if "reasoning" not in kwargs or not _is_think_rejection(exc):
            raise
        _NO_THINK_SUPPORT.add(tag)
        diag.log(f"llms: {tag or 'model'} rejects the think flag — retrying without it")
        return runnable.invoke(messages, **{k: v for k, v in kwargs.items() if k != "reasoning"})


def stream(runnable, messages, *, tag: str = "", **kwargs):
    """`runnable.stream(...)` under the same one-shot think fallback as `generate`. The generator
    is materialized far enough to surface a parameter rejection HERE (the daemon rejects on the
    first chunk), because a caller iterating the stream cannot retry it."""
    try:
        gen = runnable.stream(messages, **kwargs)
        first = next(gen, None)
    except Exception as exc:
        if "reasoning" not in kwargs or not _is_think_rejection(exc):
            raise
        _NO_THINK_SUPPORT.add(tag)
        diag.log(f"llms: {tag or 'model'} rejects the think flag — re-streaming without it")
        kwargs = {k: v for k, v in kwargs.items() if k != "reasoning"}
        gen = runnable.stream(messages, **kwargs)
        first = next(gen, None)

    def _chunks():
        # try/finally so a consumer that closes this generator between the first chunk and the
        # rest (a cancelled turn) still closes the underlying stream.
        try:
            if first is not None:
                yield first
            yield from gen
        finally:
            close = getattr(gen, "close", None)
            if close:
                close()

    return _chunks()


def extract_tok_per_sec(response) -> float:
    """Return tokens/second from an AIMessage's response_metadata, or 0.0 if unavailable.
    Ollama populates eval_count (tokens generated) and eval_duration (nanoseconds); other
    providers leave these absent so we gracefully return 0."""
    meta = getattr(response, "response_metadata", None) or {}
    eval_count = meta.get("eval_count", 0) or 0
    eval_duration = meta.get("eval_duration", 0) or 0
    if eval_duration > 0:
        return eval_count / (eval_duration / 1e9)
    return 0.0


def extract_prompt_tokens(response) -> int:
    """Tokens the model just ingested — i.e. how full the context window is right now. Prefers
    the standard usage_metadata.input_tokens, falling back to Ollama's
    response_metadata.prompt_eval_count; 0 if neither is present. Feeds the UI context gauge."""
    usage = getattr(response, "usage_metadata", None) or {}
    n = usage.get("input_tokens")
    if n:
        return int(n)
    meta = getattr(response, "response_metadata", None) or {}
    return int(meta.get("prompt_eval_count", 0) or 0)


def active_context_window() -> int:
    """Effective context window (`num_ctx`) of the chat model — the denominator of the UI's
    fill gauge."""
    return get_config().num_ctx_for(model_id())
