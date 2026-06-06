from commands._framework import command, _print
from commands._utils import _resync_rag_after_model_change


def _config_keys(ctx, args):
    """`/config key …` — the API-key (secrets) front end."""
    import env_keys

    sub = args[0].lower() if args else "list"

    if sub in ("list", "ls", "status"):
        _print("  API keys (stored in .env, applied live, masked here):")
        for k in env_keys.KNOWN_KEYS:
            state = env_keys.mask(env_keys.get(k.name)) if env_keys.is_set(k.name) else "not set"
            _print(f"    {k.name:<20} {state}")
            _print(f"      {k.label} — {k.purpose}")
            if k.url:
                _print(f"      get one: {k.url}")
        _print("  set:   /config key set <NAME> <value>")
        _print("  clear: /config key unset <NAME>")
        return

    if sub == "get":
        if len(args) < 2:
            _print("  usage: /config key get <NAME>")
            return
        name = args[1].upper()
        _print(f"    {name} = {env_keys.mask(env_keys.get(name))}")
        return

    if sub == "set":
        if len(args) < 3:
            _print("  usage: /config key set <NAME> <value>")
            return
        name = args[1].upper()
        value = " ".join(args[2:]).strip()
        env_keys.set_value(name, value)
        managed = env_keys.find(name)
        tag = "" if managed else "  (unmanaged key — stored, but no client reset hook)"
        _print(f"    {name} set -> {env_keys.mask(value)} (saved to .env, applied now){tag}")
        return

    if sub in ("unset", "clear", "remove", "rm"):
        if len(args) < 2:
            _print("  usage: /config key unset <NAME>")
            return
        name = args[1].upper()
        if env_keys.unset_value(name):
            _print(f"    {name} removed from .env and the live environment.")
        else:
            _print(f"    {name} was not set.")
        return

    _print(f"  unknown /config key subcommand: {sub!r} — try: list, set, unset, get")


@command(
    "config",
    "View or edit runtime config (config.yaml) and API keys (.env). Edits are session-only.",
    usage="/config | /config <dotted.key> [value] | /config key … | /config reload",
    details="""
With no args, prints the key runtime settings (active_tier, runtime.max_iterations,
runtime.auto_approve), the resolved paths, and which API keys are set.

With a dotted key, reads that value; with a key and a value, sets it for this session only.
`/config reload` re-reads config.yaml from disk, discarding any session edits.

API keys live in .env, not config.yaml, so they have their own subcommand:
  /config key                       list known keys and whether each is set (masked)
  /config key set <NAME> <value>    save a key to .env, apply it live, reset any cached client
  /config key unset <NAME>          remove a key from .env and the environment
  /config key get <NAME>            show one key (masked)

Known keys: TAVILY_API_KEY (web tools; optional — they fall back to keyless search without it),
ANTHROPIC_API_KEY (cloud-hybrid tier). Add more by registering a ManagedKey in env_keys.py.

Model/tier keys rebuild the cached models on next use; an embedder change re-embeds the corpus.
To change model bindings specifically, /models is the friendlier front end.

Examples:
  /config                              show the summary
  /config runtime.max_iterations       read one key
  /config runtime.max_iterations 12    set it (session only)
  /config key set TAVILY_API_KEY tvly-... add an API key
  /config reload                       re-read config.yaml from disk
""",
)
def _config(ctx, args):
    from config import get_config, reload

    cfg = get_config()

    if args and args[0].lower() in ("key", "keys", "secret", "secrets"):
        _config_keys(ctx, args[1:])
        return

    if not args:
        import env_keys

        _print("  runtime config:")
        _print(f"    active_tier           : {cfg.active_tier}")
        _print(f"    runtime.max_iterations: {cfg.max_iterations}")
        _print(f"    runtime.auto_approve  : {cfg.auto_approve}")
        _print(f"    runtime.num_ctx       : {cfg.num_ctx_override or 'auto (per-model capability)'}")
        _print("  paths:")
        for name in ("documents", "workspace", "memory", "db_sqlite"):
            _print(f"    {name:<10}: {cfg.get('paths.' + name)}")
        _print("  api keys (.env):")
        for k in env_keys.KNOWN_KEYS:
            _print(f"    {k.name:<20}: {'set' if env_keys.is_set(k.name) else 'not set'}")
        _print("  (workspace & memory resolve live; documents/db_sqlite apply on re-ingest/restart)")
        _print("  set a value: /config <dotted.key> <value>   (e.g. /config runtime.max_iterations 12)")
        _print("  manage keys: /config key   (see /config --help)")
        return

    if args[0] == "reload":
        reload()
        from llms import reset_models
        reset_models()
        _print("  config.yaml reloaded from disk (any session edits discarded).")
        _resync_rag_after_model_change()
        return

    key = args[0]
    if len(args) == 1:
        _print(f"  {key} = {cfg.get(key)!r}")
        return

    value = " ".join(args[1:])
    cfg.set(key, value)
    _print(f"  {key} = {cfg.get(key)!r}  (session only; edit config.yaml to persist)")
    if key.startswith("tiers.") or key == "active_tier":
        from llms import reset_models
        reset_models()
        _print("  (models will rebuild on next use)")
        _resync_rag_after_model_change()
    elif key == "runtime.num_ctx":
        from llms import reset_models
        reset_models()
        _print("  (models will rebuild with the new context window on next use)")
