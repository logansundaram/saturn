from __future__ import annotations

from typing import Optional

from commands._framework import command, _print
from commands._utils import _ROLES, _resync_rag_after_model_change, split_save_flag

_BIND_TARGETS = ("all", *_ROLES, "embedder")


def _set_role_binding(cfg, role: str, model: str, provider: Optional[str]) -> None:
    key = f"tiers.{cfg.active_tier}.roles.{role}"
    if provider:
        cfg.set(key, {"provider": provider, "model": model})
    else:
        cfg.set(key, model)


def _persist_bindings(cfg, keys: list[str]) -> None:
    """Persist session-set binding keys to config.yaml through the one persist seam (the same
    machinery as /config <key> --save). A provider-form {provider, model} binding is a container,
    not a scalar leaf — _persist_key reports it as session-only instead of failing the command."""
    from commands.config import _persist_key

    for key in keys:
        _persist_key(cfg, key)


def _bind(cfg, target: str, model: str, provider: Optional[str] = None, *, save: bool = False) -> None:
    from core.llms import reset_models

    tag = "" if save else " (session only)"

    if target == "embedder":
        key = f"tiers.{cfg.active_tier}.embedder"
        cfg.set(key, model)
        reset_models()
        _print(f"  embedder -> {model} on tier '{cfg.active_tier}'{tag}.")
        if save:
            _persist_bindings(cfg, [key])
        else:
            _print("  add --save to persist to config.yaml.")
        _resync_rag_after_model_change()
        return

    if target == "all":
        for role in _ROLES:
            _set_role_binding(cfg, role, model, provider)
        reset_models()
        _print(f"  all roles -> {model} on tier '{cfg.active_tier}'{tag}.")
        keys = [f"tiers.{cfg.active_tier}.roles.{role}" for role in _ROLES]
    else:
        _set_role_binding(cfg, target, model, provider)
        reset_models()
        bound = f"{provider}:{model}" if provider else model
        _print(f"  {target} -> {bound} on tier '{cfg.active_tier}'{tag}.")
        keys = [f"tiers.{cfg.active_tier}.roles.{target}"]
    if save:
        _persist_bindings(cfg, keys)
    else:
        _print("  add --save to persist to config.yaml.")
    _resync_rag_after_model_change()


def _models_picker(ctx, cfg, local, *, save: bool = False) -> None:
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
    tokens = ui.ask(
        f"drive what with {choice.name}? [{'|'.join(_BIND_TARGETS)}] (default {default}) » "
    ).lower().split()
    # The target prompt accepts a trailing --save too, same as the direct-bind forms.
    tokens, picked_save = split_save_flag(tokens)
    save = save or picked_save
    target = tokens[0] if tokens else default
    if target not in _BIND_TARGETS:
        _print(f"  unknown target: {target} (choose one of {', '.join(_BIND_TARGETS)})")
        return
    _bind(cfg, target, choice.name, save=save)


@command(
    "models",
    "List installed models; pick or switch what drives each role / the embedder.",
    aliases=("model",),
    usage="/models [--save] | /models <role|all|embedder> <id> [--save] | /models tier <name> [--save]",
    details="""
With no args, pings the local Ollama daemon, renders every installed model (size, params,
quantization, and what each currently drives) as a numbered table, then drops into an interactive
picker: choose a model by number, then choose what it should drive. Picking a chat model defaults
to 'all' roles (the common 'run everything locally on this model' case); picking an embed-only
model defaults to the embedder. Blank input cancels at either step; the target prompt accepts a
trailing --save like the direct forms below.

You can also bind directly, without the picker:
  /models                            list + interactive picker
  /models all <id> [--save]          point every role at one model
  /models <role> <id> [prov] [--save]  re-point one role (bare id = tier default provider)
  /models embedder <id> [--save]     switch the embedding model (re-embeds the corpus)
  /models tier <name> [--save]       switch the whole hardware tier

Roles: planner, tool_caller, synthesizer, utility, judge.

Every switch is session-only by default and rebuilds the cached models on next use; append --save
to also write the change back to config.yaml in place (the same dotted key(s) the session edit
sets, via the /config <key> --save machinery). A provider-form {provider, model} binding has no
scalar leaf to write — --save reports it and leaves config.yaml for a hand edit. Any change that
moves the embedder re-embeds the document corpus. An explicit provider (3rd arg on a single role)
writes the cross-provider {provider, model} form, e.g.:
  /models planner claude-sonnet-4-6 anthropic
""",
)
def _models(ctx, args):
    from config import get_config
    from core.llms import model_id, reset_models, list_local_models
    from tui import ui

    cfg = get_config()
    bindings = {role: model_id(role) for role in _ROLES}

    args, save = split_save_flag(args)

    if not args:
        local = list_local_models()
        ui.show_models(local, bindings, cfg.active_tier, cfg.embedder_model, numbered=True)
        _models_picker(ctx, cfg, local, save=save)
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
        tag = "" if save else " (session only)"
        _print(f"  active tier -> {tier}; models will rebuild on next use{tag}.")
        if save:
            _persist_bindings(cfg, ["active_tier"])
        else:
            _print("  add --save to persist to config.yaml.")
        _resync_rag_after_model_change()
        return

    if sub == "embedder":
        if len(args) < 2:
            _print("  usage: /models embedder <model_id> [--save]")
            return
        _bind(cfg, "embedder", args[1], save=save)
        return

    if sub == "all":
        if len(args) < 2:
            _print("  usage: /models all <model_id> [--save]")
            return
        _bind(cfg, "all", args[1], save=save)
        return

    role = sub
    if role not in _ROLES:
        _print(f"  unknown target: {role} (roles: {', '.join(_ROLES)}; or 'all'/'embedder'/'tier')")
        return
    if len(args) < 2:
        _print(f"  usage: /models {role} <model_id> [provider] [--save]")
        return
    new_model = args[1]

    if len(args) > 2:
        provider = args[2]
    else:
        existing = cfg.get(f"tiers.{cfg.active_tier}.roles.{role}")
        provider = existing.get("provider") if isinstance(existing, dict) else None

    _bind(cfg, role, new_model, provider, save=save)
