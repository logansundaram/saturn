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

from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import get_config
from registry import tool
from state import Plan, ReplanVerdict

# (provider, model) -> BaseChatModel.  Cleared by reset_models().
_MODEL_CACHE: dict[tuple[str, str], object] = {}
# Derived handles (bound tools / structured output) are cached separately and also cleared.
_DERIVED_CACHE: dict[str, object] = {}


def _build(provider: str, model: str):
    if provider == "ollama":
        # Bind num_ctx to the effective window (runtime.num_ctx override, else the model's declared
        # window) so it actually runs at the size the UI gauges against — Ollama otherwise silently
        # caps at 2048, making the context-fill % lie. /context drops the cache to rebind live.
        return ChatOllama(model=model, num_ctx=get_config().num_ctx_for(model))
    # Any other provider: lean on LangChain's universal initializer.
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider)


def get_model(role: str):
    """Return the chat model bound to `role` under the active tier (cached)."""
    spec = get_config().model_for_role(role)
    key = (spec.provider, spec.model)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build(spec.provider, spec.model)
    return _MODEL_CACHE[key]


def capability_of(role: str):
    """Capability descriptor for the model currently bound to `role`."""
    spec = get_config().model_for_role(role)
    return get_config().capability_of(spec.model)


def model_id(role: str) -> str:
    """The concrete model id serving `role` (for display: banners, /model)."""
    return get_config().model_for_role(role).model


def get_tool_model():
    """The agent role's model with the tool registry bound natively (cached)."""
    if "tool_caller" not in _DERIVED_CACHE:
        cap = capability_of("tool_caller")
        if not cap.supports_tools:
            print(
                f"[llms] WARNING: model '{model_id('tool_caller')}' for role 'tool_caller' "
                f"does not advertise native tool-calling; the agent loop may misbehave."
            )
        _DERIVED_CACHE["tool_caller"] = get_model("tool_caller").bind_tools(tool)
    return _DERIVED_CACHE["tool_caller"]


def get_plan_model():
    """The planner role's model constrained to emit a structured Plan (cached).

    Ollama is most reliable with method="json_schema" (constrains generation at the server);
    other providers use the default structured-output method."""
    if "planner" not in _DERIVED_CACHE:
        spec = get_config().model_for_role("planner")
        cap = get_config().capability_of(spec.model)
        if not cap.supports_structured_output:
            print(
                f"[llms] WARNING: model '{spec.model}' for role 'planner' does not advertise "
                f"structured output; the planner will lean on its fallback plan."
            )
        base = get_model("planner")
        if spec.provider == "ollama":
            _DERIVED_CACHE["planner"] = base.with_structured_output(
                Plan, method="json_schema"
            )
        else:
            _DERIVED_CACHE["planner"] = base.with_structured_output(Plan)
    return _DERIVED_CACHE["planner"]


def get_judge_model():
    """The judge role's model constrained to emit a structured ReplanVerdict (cached).

    Powers the in-loop replan node (node_registry/replan.py): the verifier/repair step that
    decides whether the agent's draft answer is grounded or needs an inserted web-search step.
    Mirrors get_plan_model — Ollama uses method="json_schema" for server-constrained generation."""
    if "judge" not in _DERIVED_CACHE:
        spec = get_config().model_for_role("judge")
        cap = get_config().capability_of(spec.model)
        if not cap.supports_structured_output:
            print(
                f"[llms] WARNING: model '{spec.model}' for role 'judge' does not advertise "
                f"structured output; the replan node will skip escalation rather than misfire."
            )
        if spec.provider == "ollama":
            # A dedicated temperature-0 instance: groundedness is a classification, so we want a
            # stable verdict, not sampled variety (at the default temperature the same draft flips
            # grounded/ungrounded run to run). Built separately from the shared get_model base —
            # on single-model tiers every role shares one ChatOllama, so lowering temperature there
            # would change planner/agent/synthesizer sampling too.
            base = ChatOllama(
                model=spec.model,
                num_ctx=get_config().num_ctx_for(spec.model),
                temperature=0,
            )
            _DERIVED_CACHE["judge"] = base.with_structured_output(
                ReplanVerdict, method="json_schema"
            )
        else:
            _DERIVED_CACHE["judge"] = get_model("judge").with_structured_output(ReplanVerdict)
    return _DERIVED_CACHE["judge"]


def get_embeddings():
    """Embedding model for the RAG store (the `embedder` slot of the active tier)."""
    return OllamaEmbeddings(model=get_config().embedder_model)


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
        def field(obj, *names, default=None):
            for n in names:
                v = getattr(obj, n, None)
                if v is None and isinstance(obj, dict):
                    v = obj.get(n)
                if v is not None:
                    return v
            return default

        name = field(m, "model", "name", default="") or ""
        if not name:
            continue
        details = field(m, "details", default=None)
        family = field(details, "family", default="") or "" if details is not None else ""
        families = field(details, "families", default=[]) if details is not None else []
        out.append(
            LocalModel(
                name=name,
                size_bytes=int(field(m, "size", default=0) or 0),
                parameter_size=(field(details, "parameter_size", default="") or "")
                if details is not None else "",
                quantization=(field(details, "quantization_level", default="") or "")
                if details is not None else "",
                family=family,
                is_embedding=_looks_like_embedder(name, family, families),
            )
        )
    return sorted(out, key=lambda lm: lm.name.lower())


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
    UI's fill gauge and the /context readout. Defaults to the agent (tool_caller) role, the one
    the status bar's model label tracks."""
    return get_config().num_ctx_for(model_id(role))
