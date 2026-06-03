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

from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import get_config
from registry import tool
from state import Plan

# (provider, model) -> BaseChatModel.  Cleared by reset_models().
_MODEL_CACHE: dict[tuple[str, str], object] = {}
# Derived handles (bound tools / structured output) are cached separately and also cleared.
_DERIVED_CACHE: dict[str, object] = {}


def _build(provider: str, model: str):
    if provider == "ollama":
        return ChatOllama(model=model)
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


def get_embeddings():
    """Embedding model for the RAG store (the `embedder` slot of the active tier)."""
    return OllamaEmbeddings(model=get_config().embedder_model)


def reset_models() -> None:
    """Drop all cached models so the next get_* call rebuilds from current config. Called
    after a live model/tier change (e.g. the /model slash command)."""
    _MODEL_CACHE.clear()
    _DERIVED_CACHE.clear()
