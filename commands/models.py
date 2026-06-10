from __future__ import annotations

from typing import Optional

from commands._framework import command, _print
from commands._utils import _ROLES, _resync_rag_after_model_change

_BIND_TARGETS = ("all", *_ROLES, "embedder")


def _set_role_binding(cfg, role: str, model: str, provider: Optional[str]) -> None:
    key = f"tiers.{cfg.active_tier}.roles.{role}"
    if provider:
        cfg.set(key, {"provider": provider, "model": model})
    else:
        cfg.set(key, model)


def _bind(cfg, target: str, model: str, provider: Optional[str] = None) -> None:
    from llms import reset_models

    if target == "embedder":
        cfg.set(f"tiers.{cfg.active_tier}.embedder", model)
        reset_models()
        _print(f"  embedder -> {model} on tier '{cfg.active_tier}' (session only).")
        _resync_rag_after_model_change()
        return

    if target == "all":
        for role in _ROLES:
            _set_role_binding(cfg, role, model, provider)
        reset_models()
        _print(f"  all roles -> {model} on tier '{cfg.active_tier}' (session only).")
    else:
        _set_role_binding(cfg, target, model, provider)
        reset_models()
        bound = f"{provider}:{model}" if provider else model
        _print(f"  {target} -> {bound} on tier '{cfg.active_tier}' (session only).")
    _print("  edit config.yaml to make it permanent.")
    _resync_rag_after_model_change()


def _models_picker(ctx, cfg, local) -> None:
    from tui import ui

    if not local:
        return
    sel = ui.ask("bind a model — enter # (or blank to cancel) » ")
    if not sel:
        _print("  (cancelled)")
        return
    try:
        choice = local[int(sel) - 1]
        if int(sel) < 1:
            raise IndexError
    except (ValueError, IndexError):
        _print(f"  not a valid selection: {sel!r}")
        return

    default = "embedder" if choice.is_embedding else "all"
    tgt = ui.ask(
        f"drive what with {choice.name}? [{'|'.join(_BIND_TARGETS)}] (default {default}) » "
    ).lower()
    target = tgt or default
    if target not in _BIND_TARGETS:
        _print(f"  unknown target: {target} (choose one of {', '.join(_BIND_TARGETS)})")
        return
    _bind(cfg, target, choice.name)


@command(
    "models",
    "List installed models; pick or switch what drives each role / the embedder.",
    aliases=("model",),
    usage="/models | /models <role|all|embedder> <id> | /models tier <name>",
    details="""
With no args, pings the local Ollama daemon, renders every installed model (size, params,
quantization, and what each currently drives) as a numbered table, then drops into an interactive
picker: choose a model by number, then choose what it should drive. Picking a chat model defaults
to 'all' roles (the common 'run everything locally on this model' case); picking an embed-only
model defaults to the embedder. Blank input cancels at either step.

You can also bind directly, without the picker:
  /models                      list + interactive picker
  /models all <id>             point every role at one model
  /models <role> <id> [prov]   re-point one role (bare id = tier default provider)
  /models embedder <id>        switch the embedding model (re-embeds the corpus)
  /models tier <name>          switch the whole hardware tier

Roles: planner, tool_caller, synthesizer, utility, judge.

All switches are session-only — edit config.yaml to persist — and rebuild the cached models on
next use. Any change that moves the embedder re-embeds the document corpus. An explicit provider
(3rd arg on a single role) writes the cross-provider {provider, model} form, e.g.:
  /models planner claude-sonnet-4-6 anthropic
""",
)
def _models(ctx, args):
    from config import get_config
    from llms import model_id, reset_models, list_local_models
    from tui import ui

    cfg = get_config()
    bindings = {role: model_id(role) for role in _ROLES}

    if not args:
        local = list_local_models()
        ui.show_models(local, bindings, cfg.active_tier, cfg.embedder_model, numbered=True)
        _models_picker(ctx, cfg, local)
        return

    sub = args[0].lower()

    if sub == "tier":
        if len(args) < 2:
            _print("  tiers (switch with /models tier <name>):")
            for name in cfg.get("tiers", {}):
                mark = "*" if name == cfg.active_tier else " "
                _print(f"   {mark} {name}")
            return
        tier = args[1]
        if cfg.get(f"tiers.{tier}") is None:
            _print(f"  unknown tier: {tier} (defined: {list(cfg.get('tiers', {}))})")
            return
        cfg.set("active_tier", tier)
        reset_models()
        _print(f"  active tier -> {tier}; models will rebuild on next use (session only).")
        _resync_rag_after_model_change()
        return

    if sub == "embedder":
        if len(args) < 2:
            _print("  usage: /models embedder <model_id>")
            return
        _bind(cfg, "embedder", args[1])
        return

    if sub == "all":
        if len(args) < 2:
            _print("  usage: /models all <model_id>")
            return
        _bind(cfg, "all", args[1])
        return

    role = sub
    if role not in _ROLES:
        _print(f"  unknown target: {role} (roles: {', '.join(_ROLES)}; or 'all'/'embedder'/'tier')")
        return
    if len(args) < 2:
        _print(f"  usage: /models {role} <model_id> [provider]")
        return
    new_model = args[1]

    if len(args) > 2:
        provider = args[2]
    else:
        existing = cfg.get(f"tiers.{cfg.active_tier}.roles.{role}")
        provider = existing.get("provider") if isinstance(existing, dict) else None

    _bind(cfg, role, new_model, provider)
