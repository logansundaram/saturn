"""
Model factory (Phase 3) — `get_model(role)` instead of hard-coded globals.

The agent references model ROLES (planner, tool_caller, synthesizer, utility, judge); this
module resolves each role to a concrete model against the active hardware tier in
`config.yaml` and builds the LangChain chat model. Swapping hardware is a config edit; graph
code never names a model.

Provider abstraction: Ollama goes through `ChatOllama` directly (the confirmed-working local
path); any other provider goes through LangChain's `init_chat_model`, so OpenAI/Anthropic/etc.
are just config. Built models are cached per (provider, model) so repeated `get_model` calls in
the loop are free; `reset_models()` clears the cache after a live model change (the `/model`
command).

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
from tools.registry import tool
from core.state import Plan, ReplanVerdict


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
    """Thin proxy around a cloud chat model that makes the network boundary observable + safe.

    EVERY cloud model call (any node — planner, agent, judge, synthesizer) funnels through one of
    these because `_build` wraps every non-Ollama model in it, so this is the single place to:
      - record the egress to the ledger (`egress.record`) — what left, where to, how big; and
      - run the outgoing messages through `redaction.process_messages` first, stripping secrets
        when `runtime.redaction` is on.
    Everything else (bind_tools, with_structured_output, attribute access) delegates to the inner
    model and re-wraps any derived runnable so the boundary survives `.bind_tools(...)` /
    `.with_structured_output(...)`. LOOPBACK Ollama models are never wrapped — there is no
    boundary; a REMOTE Ollama (OLLAMA_HOST off-machine) is wrapped exactly like a cloud provider,
    with the real endpoint as the ledger host."""

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

    def bind_tools(self, *args, **kwargs):
        return _CloudBoundaryModel(
            self._inner.bind_tools(*args, **kwargs), self._provider, self._model, self._host
        )

    def with_structured_output(self, *args, **kwargs):
        return _CloudBoundaryModel(
            self._inner.with_structured_output(*args, **kwargs), self._provider, self._model, self._host
        )

    def __getattr__(self, name):
        # Anything we don't override (get_name, config_specs, etc.) defers to the inner model.
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
# Derived handles (bound tools / structured output) are cached separately and also cleared.
_DERIVED_CACHE: dict[str, object] = {}


def _wrap_ollama(m, model: str):
    """Loopback Ollama is handed back bare — there is no boundary to guard. A REMOTE Ollama
    (OLLAMA_HOST pointing off-machine) IS one: wrap it in the same cloud boundary proxy so every
    call is redacted (per runtime.redaction) and recorded to the egress ledger with the real
    endpoint as the host — 'local model' must never silently mean 'someone else's machine'."""
    if egress.ollama_is_local():
        return m
    return _CloudBoundaryModel(m, "ollama", model, host=f"ollama @ {egress.ollama_endpoint()}")


def _build(provider: str, model: str):
    if provider == "ollama":
        # Bind num_ctx to the effective window (runtime.num_ctx override, else the model's declared
        # window) so it actually runs at the size the UI gauges against — Ollama otherwise silently
        # caps at 2048, making the context-fill % lie. /context drops the cache to rebind live.
        # client_kwargs carries the request timeout (guards a wedged daemon; see _ollama_client_kwargs).
        return _wrap_ollama(
            ChatOllama(
                model=model,
                num_ctx=get_config().num_ctx_for(model),
                **_ollama_client_kwargs(),
            ),
            model,
        )
    # Any other provider (cloud): lean on LangChain's universal initializer, then wrap it in the
    # cloud boundary proxy so every call is redacted (per runtime.redaction) and recorded to the
    # egress ledger. Local Ollama above is never wrapped — it never leaves the machine.
    from langchain.chat_models import init_chat_model

    return _CloudBoundaryModel(init_chat_model(model, model_provider=provider), provider, model)


def get_model(role: str):
    """Return the chat model bound to `role` under the active tier (cached).

    Air-gap enforcement for cloud roles lives here (not in a wrapper) because a cached cloud model
    would otherwise sneak a call through after the gate engaged: when `runtime.airgap` is on and the
    role is cloud-bound, refuse to hand back a model at all — the turn fails with an actionable
    message instead of quietly reaching the network. `/privacy airgap` drops the cache so this
    re-checks."""
    spec = get_config().model_for_role(role)
    if spec.provider != "ollama" and egress.airgap_on():
        egress.record("llm", f"{spec.provider} API", f"{role} → {spec.model}",
                      provider=spec.provider, status=egress.BLOCKED)
        raise RuntimeError(
            f"Air-gap is ON — role '{role}' is bound to a cloud model "
            f"({spec.provider}:{spec.model}), which cannot run with network egress blocked. "
            f"Switch to an all-local tier (`/models tier` lists them) or turn it off with "
            f"`/privacy airgap off`."
        )
    if spec.provider == "ollama" and not egress.ollama_is_local() and egress.airgap_on():
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


def get_tool_model():
    """The agent role's model with the tool registry bound natively (cached).

    `get_model` runs first even on a derived-cache hit: its live air-gap guard is what stops a
    cloud handle cached while the boundary was open from serving calls after it sealed (e.g. a
    `/policy import` flipping runtime.airgap without anyone dropping the cache)."""
    base = get_model("tool_caller")
    if "tool_caller" not in _DERIVED_CACHE:
        # Capability advisories surface at startup via check_models() — no mid-turn print here
        # (print collides with the rich.Live TUI; see the diag.log design rule).
        _DERIVED_CACHE["tool_caller"] = base.bind_tools(tool)
    return _DERIVED_CACHE["tool_caller"]


def get_plan_model():
    """The planner role's model constrained to emit a structured Plan (cached; the unconditional
    get_model call keeps the live air-gap guard in front of the derived cache — see
    get_tool_model).

    Ollama is most reliable with method="json_schema" (constrains generation at the server);
    other providers use the default structured-output method."""
    base = get_model("planner")
    if "planner" not in _DERIVED_CACHE:
        spec = get_config().model_for_role("planner")
        if spec.provider == "ollama":
            _DERIVED_CACHE["planner"] = base.with_structured_output(
                Plan, method="json_schema"
            )
        else:
            _DERIVED_CACHE["planner"] = base.with_structured_output(Plan)
    return _DERIVED_CACHE["planner"]


def get_judge_model():
    """The judge role's model constrained to emit a structured ReplanVerdict (cached).

    Powers the in-loop replan node (nodes/replan.py): the verifier/repair step that
    decides whether the agent's draft answer is grounded or needs an inserted web-search step.
    Mirrors get_plan_model — Ollama uses method="json_schema" for server-constrained generation."""
    spec = get_config().model_for_role("judge")
    # Live air-gap guard even on a derived-cache hit (see get_tool_model). Unconditional: a
    # cloud-bound judge AND an Ollama judge behind a remote OLLAMA_HOST both cross the boundary
    # (the guard lives in get_model; a loopback Ollama judge passes through untouched).
    get_model("judge")
    if "judge" not in _DERIVED_CACHE:
        if spec.provider == "ollama":
            # A dedicated temperature-0 instance: groundedness is a classification, so we want a
            # stable verdict, not sampled variety (at the default temperature the same draft flips
            # grounded/ungrounded run to run). Built separately from the shared get_model base —
            # on single-model tiers every role shares one ChatOllama, so lowering temperature there
            # would change planner/agent/synthesizer sampling too.
            base = _wrap_ollama(
                ChatOllama(
                    model=spec.model,
                    num_ctx=get_config().num_ctx_for(spec.model),
                    temperature=0,
                    **_ollama_client_kwargs(),
                ),
                spec.model,
            )
            _DERIVED_CACHE["judge"] = base.with_structured_output(
                ReplanVerdict, method="json_schema"
            )
        else:
            _DERIVED_CACHE["judge"] = get_model("judge").with_structured_output(ReplanVerdict)
    return _DERIVED_CACHE["judge"]


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
    _DERIVED_CACHE.clear()


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


# Cloud providers and the env var that unlocks them (mirrors env_keys.KNOWN_KEYS; extend together).
_PROVIDER_KEY = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# ... and the LangChain integration package each provider needs (init_chat_model imports it lazily,
# so a missing one otherwise surfaces as an ImportError mid-turn instead of at startup).
_PROVIDER_PKG = {"anthropic": "langchain_anthropic", "openai": "langchain_openai"}


def check_models() -> list[str]:
    """Startup health report for the active tier. Returns a list of human-readable PROBLEM strings
    (empty when all is well): the Ollama daemon being down, local model tags not pulled, or a
    missing API key for a cloud-bound role. Non-fatal — `agent.main` prints these as warnings and
    continues (a degraded tier still runs the commands/REPL; the first affected turn fails cleanly
    rather than the app refusing to start)."""
    cfg = get_config()
    problems: list[str] = []

    need_ollama: list[str] = []
    cloud: dict[str, set[str]] = {}
    for role in MODEL_ROLES:
        spec = cfg.model_for_role(role)
        if spec.provider == "ollama":
            need_ollama.append(spec.model)
        else:
            cloud.setdefault(spec.provider, set()).add(spec.model)
    need_ollama.append(cfg.embedder_model)  # embeddings always run through Ollama
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

    if cloud:
        import importlib.util

        import env_keys

        for provider, models in sorted(cloud.items()):
            key = _PROVIDER_KEY.get(provider)
            if key and not env_keys.is_set(key):
                problems.append(
                    f"{provider} model(s) {', '.join(sorted(models))} need {key} — "
                    f"set it with `/config key set {key} <value>`"
                )
            pkg = _PROVIDER_PKG.get(provider)
            if pkg and importlib.util.find_spec(pkg) is None:
                problems.append(
                    f"{provider} model(s) {', '.join(sorted(models))} need the "
                    f"`{pkg.replace('_', '-')}` package — run `pip install {pkg.replace('_', '-')}`"
                )

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


def extract_total_tokens(response) -> int:
    """Total tokens (input + output) consumed by one LLM call, best-effort — feeds the session
    token budget (budget.py). Prefers the standard usage_metadata; falls back to Ollama's
    response_metadata counters (prompt_eval_count + eval_count); 0 when neither is present (the
    budget then simply doesn't see that call — an undercount, never a crash)."""
    usage = getattr(response, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total:
        return int(total)
    n = int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
    if n:
        return n
    meta = getattr(response, "response_metadata", None) or {}
    return int(meta.get("prompt_eval_count", 0) or 0) + int(meta.get("eval_count", 0) or 0)


def active_context_window(role: str = "tool_caller") -> int:
    """Effective context window (`num_ctx`) of the model serving `role` — the denominator of the
    UI's fill gauge and the /context readout. Defaults to the agent (tool_caller) role, the one
    the status bar's model label tracks."""
    return get_config().num_ctx_for(model_id(role))
