"""
Model factory (Phase 3) — `get_model(role)` instead of hard-coded globals.

The agent references model ROLES (planner, tool_caller, synthesizer, utility, judge); this
module resolves each role to a concrete model against the active hardware tier in
`config.yaml` and builds the LangChain chat model. Swapping hardware is a config edit; graph
code never names a model.

Providers: Ollama only. **Cloud model support (Anthropic/OpenAI via `init_chat_model`) is
SHELVED (2026-07-03)** — the edge is local-first, and carrying a cloud path that nothing on the
tier presets exercises cost audit surface for no product. A role bound to a non-ollama provider
(an old config, or a hand edit) refuses to build with a pointer at `/models`; `check_models`
surfaces the same at startup. The network-boundary machinery is NOT shelved — `_CloudBoundaryModel`
still wraps a remote-OLLAMA_HOST daemon (redaction + egress + air-gap), and reintroducing cloud
later is: restore `_build`'s `init_chat_model` branch + the provider key/package checks in
`check_models` + the managed-key layer (env_keys.py holds only the .env read path since the
2026-07-16 /config key cut; the pre-cut ManagedKey registry is in git history — see the Roadmap
note).
Built models are cached per (provider, model); `reset_models()` clears the cache after a live
model change (the `/model` command).

Capability descriptors come from config; the MVP requires native tool-calling + structured
output for the loop-driving roles, and we warn (not crash) if a bound model lacks them.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings

from trust import egress
from trust import redaction
from config import MODEL_ROLES, get_config


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


class _CloudBoundaryModel:
    """Thin proxy around an off-machine chat model that makes the network boundary observable +
    safe. Its one live user today is a REMOTE Ollama (OLLAMA_HOST off-machine) via `_wrap_ollama`
    — cloud providers are SHELVED (2026-07-03), and when they return, `_build` wraps them here
    again (this class is the reintroduction seam; do not delete it with the shelve). Every call
    through it:
      - records the egress to the ledger (`egress.record`) — what left, where to, how big; and
      - runs the outgoing messages through `redaction.process_messages` first, stripping secrets
        when `runtime.redaction` is on.
    Everything else (bind_tools, with_structured_output, attribute access) delegates to the inner
    model and re-wraps any derived runnable so the boundary survives `.bind_tools(...)` /
    `.with_structured_output(...)`. LOOPBACK Ollama models are never wrapped — there is no
    boundary."""

    def __init__(self, inner, provider: str, model: str, host: str = ""):
        self._inner = inner
        self._provider = provider
        self._model = model
        self._host = host or f"{provider} API"

    def _outgoing(self, messages):
        """Redact (per the mode) then record the egress; return the messages to actually send.
        n_bytes measures `to_send` — what actually crosses the boundary — not the pre-redaction
        original: in redact mode the two differ by exactly the secrets that were stripped."""
        to_send, redactions = redaction.process_messages(messages) if isinstance(messages, list) else (messages, 0)
        egress.record(
            "llm", self._host, self._model,
            provider=self._provider, n_bytes=_approx_bytes(to_send), redactions=redactions,
        )
        return to_send

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
        # Through invoke one input at a time so EVERY input is redacted + recorded — the inner
        # model's batch would take the whole list past the boundary in one unobserved call.
        return [self.invoke(i, *args, **kwargs) for i in inputs]

    async def abatch(self, inputs, *args, **kwargs):
        return [await self.ainvoke(i, *args, **kwargs) for i in inputs]

    def bind_tools(self, *args, **kwargs):
        return _CloudBoundaryModel(
            self._inner.bind_tools(*args, **kwargs), self._provider, self._model, self._host
        )

    def with_structured_output(self, *args, **kwargs):
        return _CloudBoundaryModel(
            self._inner.with_structured_output(*args, **kwargs), self._provider, self._model, self._host
        )

    # Network entry points this proxy does NOT cover fail CLOSED: __getattr__ used to hand them
    # back bound to the INNER model, so a future caller (or a LangChain runnable composition)
    # would send unredacted, unrecorded content — the exact leak the boundary exists to prevent.
    # Nothing in the repo calls these today; a new caller gets a loud pointer, never a bypass.
    _UNGUARDED = frozenset({
        "generate", "agenerate", "generate_prompt", "agenerate_prompt",
        "transform", "atransform", "batch_as_completed", "abatch_as_completed",
    })

    def __getattr__(self, name):
        if name in _CloudBoundaryModel._UNGUARDED:
            raise AttributeError(
                f"_CloudBoundaryModel does not expose {name!r}: it would bypass the "
                "redaction/egress boundary — use invoke/stream/astream/batch instead"
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

# (provider, model) -> BaseChatModel.  Cleared by reset_models().
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _wrap_ollama(m, model: str):
    """Loopback Ollama is handed back bare — there is no boundary to guard. A REMOTE Ollama
    (OLLAMA_HOST pointing off-machine) IS one: wrap it in the same cloud boundary proxy so every
    call is redacted (per runtime.redaction) and recorded to the egress ledger with the real
    endpoint as the host — 'local model' must never silently mean 'someone else's machine'."""
    if egress.ollama_is_local():
        return m
    return _CloudBoundaryModel(m, "ollama", model, host=f"ollama @ {egress.ollama_endpoint()}")


def _cloud_shelved_error(role: str, provider: str, model: str) -> RuntimeError:
    """The one refusal a cloud-bound role gets (cloud support SHELVED 2026-07-03 — local-first
    is the edge; see the module docstring for the reintroduction seam)."""
    return RuntimeError(
        f"Cloud model support is shelved — role '{role}' is bound to {provider}:{model}, which "
        f"cannot run. Bind it to a local Ollama model (`/models {role} <id>`, or switch tiers "
        f"with `/models tier`)."
    )


def _build(provider: str, model: str):
    if provider != "ollama":
        # Unreachable through get_model (it refuses first); kept fail-closed so no future caller
        # can build a cloud client past the shelve.
        raise _cloud_shelved_error("?", provider, model)
    # Bind num_ctx to the effective window (runtime.num_ctx override, else the model's declared
    # window) so it actually runs at the size the UI gauges against — Ollama otherwise silently
    # caps at 2048, making the context-fill % lie. /config context drops the cache to rebind live.
    # client_kwargs carries the request timeout (guards a wedged daemon; see _ollama_client_kwargs).
    return _wrap_ollama(
        ChatOllama(
            model=model,
            num_ctx=get_config().num_ctx_for(model),
            **_ollama_client_kwargs(),
        ),
        model,
    )


def get_model(role: str):
    """Return the chat model bound to `role` under the active tier (cached).

    A role bound to a non-ollama provider refuses here — cloud model support is SHELVED
    (2026-07-03; an old config.yaml carrying a cloud-hybrid binding still loads, it just can't
    run). Air-gap enforcement for a remote OLLAMA_HOST also lives here (not in a wrapper)
    because a cached remote handle would otherwise sneak a call through after the gate engaged.
    `/privacy airgap` drops the cache so this re-checks."""
    spec = get_config().model_for_role(role)
    if spec.provider != "ollama":
        raise _cloud_shelved_error(role, spec.provider, spec.model)
    if not egress.ollama_is_local() and egress.airgap_on():
        # An off-machine OLLAMA_HOST makes the "local" model network egress — same refusal as a
        # cloud role, with the endpoint named so the fix is obvious.
        egress.record("llm", f"ollama @ {egress.ollama_endpoint()}", f"{role} → {spec.model}",
                      provider="ollama", status=egress.BLOCKED)
        raise RuntimeError(
            f"Air-gap is ON — OLLAMA_HOST points off this machine ({egress.ollama_endpoint()}), "
            f"so role '{role}' ({spec.model}) would cross the network. Unset OLLAMA_HOST to use "
            f"the local daemon, or turn the air-gap off with `/privacy airgap off`."
        )
    key = (spec.provider, spec.model)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build(spec.provider, spec.model)
    return _MODEL_CACHE[key]


def model_id(role: str) -> str:
    """The concrete model id serving `role` (for display: banners, /model)."""
    return get_config().model_for_role(role).model


# (get_tool_model / get_plan_model / get_judge_model were removed with the 2026-07-03 engine
# transplant: structured judgments now go through core/structured.py — flat schemas + shape hints
# + salvage parsing over get_model(role), with per-attempt temperature riding the invoke kwargs —
# and the execute node binds ONE tool per call (nodes/execute._generate_tool_call), never the
# whole registry.)


class _EmbeddingsBoundary:
    """OllamaEmbeddings against a REMOTE daemon — the embedding twin of _CloudBoundaryModel.
    Every batch checks the air-gap first (raising, since an embedder can't hand back a refusal
    string) and records the egress: corpus text leaving for another machine must show in the
    ledger like any other send. Loopback embeddings are never wrapped."""

    def __init__(self, inner, model: str, host: str):
        self._inner, self._model, self._host = inner, model, host

    def _gate(self, texts) -> None:
        if egress.airgap_on():
            egress.record("embedding", self._host, self._model,
                          provider="ollama", status=egress.BLOCKED)
            raise RuntimeError(
                f"Air-gap is ON — OLLAMA_HOST points off this machine ({self._host}), so "
                f"embedding would send document text across the network. Unset OLLAMA_HOST or "
                f"turn the air-gap off with `/privacy airgap off`."
            )
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
    crossing the network is egress, exactly like a cloud chat call."""
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

    name: str            # the tag you bind (e.g. "gemma4:e4b")
    size_bytes: int      # on-disk size
    parameter_size: str  # e.g. "4B", "29.9B" ("" if Ollama didn't report it)
    quantization: str    # e.g. "Q4_K_M" ("" if absent)
    family: str          # e.g. "gemma", "glm4moelite" ("" if absent)
    is_embedding: bool   # heuristic: an embed-only model (can't serve a chat role)

    @property
    def size_h(self) -> str:
        """Human-readable on-disk size (GiB/MiB)."""
        gib = self.size_bytes / 1024**3
        if gib >= 1:
            return f"{gib:.1f}G"
        return f"{self.size_bytes / 1024**2:.0f}M"


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
# Surfaces a missing daemon / un-pulled model / missing cloud key at STARTUP with an actionable
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
    (empty when all is well): the Ollama daemon being down, local model tags not pulled, or a
    role still bound to a (shelved) cloud provider. Non-fatal — `agent.main` prints these as
    warnings and continues (a degraded tier still runs the commands/REPL; the first affected turn
    fails cleanly rather than the app refusing to start)."""
    cfg = get_config()
    problems: list[str] = []

    need_ollama: list[str] = []
    for role in MODEL_ROLES:
        spec = cfg.model_for_role(role)
        if spec.provider == "ollama":
            need_ollama.append(spec.model)
        else:
            # Cloud support is SHELVED (2026-07-03): a binding a pre-shelve config still carries
            # loads fine but cannot run — say so at startup, not as a mid-turn failure.
            problems.append(
                f"role '{role}' is bound to {spec.provider}:{spec.model} — cloud model support "
                f"is shelved; rebind it to a local Ollama model (`/models {role} <id>`)"
            )
    try:
        need_ollama.append(cfg.embedder_model)  # embeddings always run through Ollama
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

    # Capability advisories for the loop-driving roles. These used to print lazily on a model's
    # first use (mid-turn, colliding with the live TUI); surfacing them here puts them next to
    # the other startup warnings with the rest of the health report.
    for role, attr, needs, consequence in (
        ("tool_caller", "supports_tools", "native tool-calling", "the agent loop may misbehave"),
        ("planner", "supports_structured_output", "structured output",
         "the planner will lean on its fallback plan"),
        ("judge", "supports_structured_output", "structured output",
         "the replan judge may misfire"),
    ):
        spec = cfg.model_for_role(role)
        if not getattr(cfg.capability_of(spec.model), attr):
            problems.append(
                f"model `{spec.model}` (role {role}) does not advertise {needs} — {consequence}"
            )

    return problems


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


def active_context_window(role: str = "tool_caller") -> int:
    """Effective context window (`num_ctx`) of the model serving `role` — the denominator of the
    UI's fill gauge and the /config context readout. Defaults to the agent (tool_caller) role, the one
    the status bar's model label tracks."""
    return get_config().num_ctx_for(model_id(role))
