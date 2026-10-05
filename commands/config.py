from commands._framework import command, _print
from commands._utils import (
    _resync_rag_after_model_change,
    split_persist_flags,
)
from config import TRUST_KEYS, in_template

_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos

# The trust-posture key set is declared in config.py (a security classification, not a UI
# detail — see TRUST_KEYS there).
_TRUST_KEYS = TRUST_KEYS

# Existence sentinel for cfg.get: distinguishes a key that is ABSENT from one present with an
# explicit null value (cfg.get's None default conflates the two, so a typo'd key would read back
# as a success-shaped `= None`).
_MISSING = object()


def _leaf_keys(node: dict, prefix: str = "") -> list[str]:
    """Every dotted path to a non-mapping leaf in the live config — the did-you-mean candidate
    list for a typo'd key. Callers snapshot this BEFORE a cfg.set, so a just-created typo can
    never suggest itself."""
    out: list[str] = []
    for k, v in node.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.extend(_leaf_keys(v, dotted + "."))
        else:
            out.append(dotted)
    return out


def _did_you_mean(cfg, key: str) -> str:
    """` — did you mean X?` for the closest existing dotted leaf, or "". The exact /policy risk
    suggestion wording, so the two typo surfaces read identically."""
    import difflib

    hint = difflib.get_close_matches(key, _leaf_keys(cfg._data), n=1)
    return f" — did you mean {hint[0]}?" if hint else ""


# Retired /config subcommands -> where that job lives now.
_RETIRED = {
    "setup": "/models checks and pulls the models; startup warns about anything missing.",
    "doctor": "/models checks and pulls the models; startup warns about anything missing.",
    "check": "/models checks and pulls the models; startup warns about anything missing.",
    "context": "the status bar shows the fill; /config runtime.num_ctx <size|auto> sets it.",
    "persist": "use /config <key> --save.",
}


def _config_keys_cut() -> None:
    import env_keys

    _print("  /config key was cut — no Saturn feature takes an API key (web search is keyless,")
    _print("  inference is local). For MCP servers' ${VAR} expansion, put plain env vars in:")
    _print(f"    {env_keys._ENV_PATH}")
    _print("  (or export them in your shell; /mcp reload picks up changes).")


@command(
    "config",
    "View or edit runtime config (config.yaml); edits persist by default.",
    usage="/config | /config <dotted.key> [value] [--session] | /config reload",
    details="""
With no args, prints the key runtime settings (active_tier, runtime.max_iterations,
runtime.auto_approve, runtime.num_ctx), the working folder (the folder Saturn was launched from,
plus any /add-dir folders) and the data paths.

With a dotted key, reads that value; with a key and a value, sets it AND writes it back to
config.yaml in place (comments and layout preserved) so it survives a restart — a setting you
change should stick. Append --session to apply an edit for this session only:
  /config runtime.max_iterations 12            set AND persist to config.yaml (the default)
  /config runtime.max_iterations 12 --session  set for this session only
  /config runtime.max_iterations --save        persist the CURRENT value unchanged
A trust setting (runtime.auto_approve, runtime.airgap, …) never persists silently: it needs
--save. A setting your config.yaml was written before (the template has it, your file does not)
gains its line on the first save; any other unknown key stays session-only with a note.
`/config reload` re-reads config.yaml from disk, discarding session-only edits.

The context window: `/config runtime.num_ctx 16384` resizes it (the models rebuild on next use),
`/config runtime.num_ctx auto` goes back to each model's declared window. The status bar shows
the fill during a turn; older turns compact on their own as it fills (runtime.auto_compact).

Models and tiers: /models. Secrets for MCP servers' ${VAR} expansion are plain env vars — put
them in .env or export them in your shell.

Examples:
  /config                              show the summary
  /config runtime.max_iterations       read one key
  /config runtime.max_iterations 12    set it and persist to config.yaml
  /config reload                       re-read config.yaml from disk
""",
)
def _config(ctx, args):
    from config import get_config, reload

    cfg = get_config()

    if args and args[0].lower() in ("key", "keys", "secret", "secrets"):
        _config_keys_cut()
        return

    if args and args[0].lower() in _RETIRED:
        _print(f"  /config {args[0].lower()} is gone — {_RETIRED[args[0].lower()]}")
        return

    if not args:
        _print("  runtime config:")
        _print(f"    active_tier           : {cfg.active_tier}")
        _print(f"    runtime.max_iterations: {cfg.max_iterations}")
        _print(f"    runtime.auto_approve  : {cfg.auto_approve}")
        _print(f"    runtime.num_ctx       : {cfg.num_ctx_override or 'auto (per-model capability)'}")
        # The working folder is the launch folder (+ /add-dir), not a config path —
        # `paths.workspace` is only the fallback when nothing set it (tests, the benchmark).
        from core import workspace

        _print(f"  working folder          : {workspace.display(workspace.root())}")
        for added in workspace.extra():
            _print(f"    + {workspace.display(added)}   (/add-dir, this session)")
        _print("  paths:")
        for name in ("documents", "memory", "db_sqlite"):
            _print(f"    {name:<10}: {cfg.get('paths.' + name)}")
        _print("  (memory resolves live; documents/db_sqlite apply on re-ingest/restart)")
        _print("  set a value: /config <dotted.key> <value>   (e.g. /config runtime.max_iterations 12)")
        return

    if args[0].lower() == "reload":  # case-insensitive like every sibling subcommand match
        reload()
        from core.llms import reset_models
        from tools import registry
        registry.apply_toolkits()  # the bound tools follow the file's `toolkits:` again
        reset_models()
        _print("  config.yaml reloaded from disk (any session edits discarded).")
        _resync_rag_after_model_change()
        return

    # Settings PERSIST to config.yaml by default; --session / --session-only applies an edit for
    # this session only. --save / -s is accepted too (persist the current value, or a trust key).
    # (split_persist_flags: case-insensitive, any position, exact token only — the bare words
    # save/persist are NOT flags, guarded below.)
    rest, session, save = split_persist_flags(args)
    if not rest:
        _print("  usage: /config <dotted.key> [value] [--session]")
        return
    key = rest[0]
    values = rest[1:]

    if not values:
        if save and not session:
            # A bare `--save` with no value persists the CURRENT value (it mutates nothing live).
            _persist_key(cfg, key)
            return
        current = cfg.get(key, _MISSING)
        if current is _MISSING:
            # An absent key must not read back success-shaped as `= None` — None stays the
            # rendering only for a key explicitly present with a null value.
            _print(f"  {key} is not set{_did_you_mean(cfg, key)}")
            return
        _print(f"  {key} = {current!r}")
        return

    # A trailing bare save/persist is a mistyped flag; storing it silently as value text would
    # corrupt the setting — refuse and point at the one spelling instead.
    if values[-1].lower() in ("save", "persist", "--persist"):
        _print(f"  did you mean --save? (the bare word {values[-1]!r} is not a persist flag; "
               "use --save / -s) — nothing set")
        return

    value = " ".join(values)
    if key == "runtime.num_ctx" and value.isdigit() and int(value) < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {value} (minimum {_MIN_NUM_CTX}) — nothing set")
        return

    # Section guard: a dotted key naming a whole MAPPING must refuse — cfg.set would replace the
    # mapping with a scalar (every `web.*`-style read silently degrades to defaults for the rest
    # of the session), and a later persist would rewrite the bare `web:` header line into
    # `web: foo` above its still-indented children: unparseable YAML that kills the next launch
    # (_set_yaml_scalar also refuses headers, but the session-side corruption must stop here
    # too). The guard lives in this handler, NOT in Config.set.
    current = cfg.get(key, _MISSING)
    if isinstance(current, dict):
        children = ", ".join(f"{key}.{child}" for child in current)
        _print(f"  {key} is a section, not a setting — set one of: {children}")
        return
    if isinstance(current, list):
        _print(f"  {key} is a list, not a scalar setting — edit config.yaml by hand")
        return

    # A key the config has never seen still sets — the default-tolerant knobs must keep working
    # on a config.yaml predating them — but the success-shaped line is replaced with a plain
    # warning so a misspelled safety knob can't masquerade as applied. The suggestion snapshots
    # the leaf list BEFORE the set, so the typo never suggests itself.
    # A setting the template declares is not unknown, only newer than this config.yaml: it
    # takes the ordinary path below, and persist adds its line.
    unknown = current is _MISSING and not in_template(key)
    suggestion = _did_you_mean(cfg, key) if unknown else ""
    cfg.set(key, value)
    if unknown:
        # A key neither config.yaml nor the template holds can't be persisted (a typo must not
        # become a line in the file), so it is inherently session-only — say so plainly instead
        # of attempting a persist that would only fail.
        _print(f"  note: {key!r} was not an existing config key{suggestion} "
               "(set for this session; only keys the code reads have any effect, and a key that "
               "is not already in config.yaml cannot be persisted)")
    elif session:
        _print(
            f"  {key} = {cfg.get(key)!r}  (session only — omit --session to save to config.yaml)"
        )
    elif key in _TRUST_KEYS and not save:
        # Never persist a security-posture change silently (the trust toggles' fail-closed
        # convention, kept even through the generic setter).
        _print(
            f"  {key} = {cfg.get(key)!r}  (session only — a trust setting never persists "
            "silently; add --save to write config.yaml)"
        )
    else:
        _persist_key(cfg, key)
    if key.startswith("tiers.") or key == "active_tier":
        from core.llms import reset_models
        reset_models()
        _print("  (models will rebuild on next use)")
        _resync_rag_after_model_change()
    elif key == "runtime.num_ctx":
        from core.llms import reset_models
        reset_models()
        _print("  (models will rebuild with the new context window on next use)")
    elif key.startswith("toolkits."):
        # /tools is the front door, but a toolkit set here must not leave the bound tools and
        # the prompt disagreeing: the prompt reads the config, the bind reads the registry.
        from core.llms import reset_models
        from tools import registry
        registry.apply_toolkits()
        reset_models()
        _print("  (toolkits re-applied — /tools shows what is bound)")


def _persist_key(cfg, key: str) -> None:
    """Write the current in-memory value of `key` back to config.yaml, reporting the outcome."""
    from config import persist

    try:
        path = persist(key)
        _print(f"  {key} = {cfg.get(key)!r}  (saved to {path.name})")
    except (KeyError, ValueError) as exc:
        _print(f"  set for this session, but not persisted: {exc}")
    except Exception as exc:
        _print(f"  set for this session, but persist failed: {exc}")
