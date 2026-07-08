from commands._framework import command, _print
from commands._utils import (
    _ROLES,
    _resync_rag_after_model_change,
    is_remove_verb,
    split_persist_flags,
)

_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos

# Existence sentinel for cfg.get: distinguishes a key that is ABSENT from one present with an
# explicit null value (cfg.get's None default conflates the two — exactly how a typo'd key used
# to read back as a success-shaped `= None`).
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


def _list_keys() -> None:
    """The numbered key listing — also the menu the set-picker selects from. The managed
    registry is EMPTY today (no Saturn feature takes an API key: web search is keyless since
    2026-07-06, inference is local) — the listing says so and points at the one remaining use,
    custom env vars (MCP servers' `${VAR}` expansion)."""
    import env_keys

    _print("  API keys (stored in .env, applied live, masked here):")
    if not env_keys.KNOWN_KEYS:
        _print("    no managed keys — no Saturn feature takes an API key (web search is keyless,")
        _print("    inference is local). Custom env vars (e.g. for MCP servers' ${VAR}) still work:")
    for i, k in enumerate(env_keys.KNOWN_KEYS, start=1):
        state = env_keys.mask(env_keys.get(k.name)) if env_keys.is_set(k.name) else "not set"
        _print(f"    {i}. {k.name:<20} {state}")
        _print(f"       {k.label} — {k.purpose}")
        if k.url:
            _print(f"       get one: {k.url}")
    _print("  set:   /config key set MY_VAR <value>   (an ALL-CAPS name is a custom env var)")
    _print("  clear: /config key unset <name>")


def _resolve_key_name(token: str) -> str | None:
    """A token the user typed where a key name goes → the env-var name to use. Managed keys match
    fuzzily (name, label, or unique substring); an ALL-CAPS token is taken verbatim as a deliberate
    unmanaged env var. Anything else is reported (so a typo can't silently create a new key)."""
    import env_keys

    key = env_keys.resolve(token)
    if key:
        return key.name
    if token == token.upper():  # deliberately-typed env-var name (e.g. MY_SERVICE_TOKEN)
        return token
    known = ", ".join(k.label.lower() for k in env_keys.KNOWN_KEYS) or "none managed"
    _print(f"  no known key matches {token!r} (known: {known}).")
    _print("  to store a custom env var, type its name in ALL CAPS: /config key set MY_VAR <value>")
    return None


def _set_key(args: list[str]) -> None:
    """`/config key set …` — every arg is optional: no name opens a picker, no value prompts for
    one, and a bare pasted secret (tvly-…, sk-ant-…) picks its own key."""
    import env_keys
    from tui import ui

    name: str | None = None
    value: str | None = None

    if not args:
        _list_keys()
        if not env_keys.KNOWN_KEYS:
            return  # nothing to pick from — the listing already explained what still works
        sel = ui.ask("which key — enter # (or blank to cancel) » ")
        if not sel:
            _print("  (cancelled)")
            return
        try:
            idx = int(sel)
            if idx < 1:
                raise IndexError
            name = env_keys.KNOWN_KEYS[idx - 1].name
        except (ValueError, IndexError):
            _print(f"  not a valid selection: {sel!r}")
            return
    elif len(args) == 1 and env_keys.detect(args[0]):
        detected = env_keys.detect(args[0])
        _print(f"  that looks like a {detected.label} key — storing it as {detected.name}.")
        name, value = detected.name, args[0]
    else:
        name = _resolve_key_name(args[0])
        if name is None:
            return
        value = " ".join(args[1:]).strip() or None

    if value is None:
        value = ui.ask(f"{name} = ").strip()
        if not value:
            _print("  (cancelled — nothing set)")
            return

    env_keys.set_value(name, value)
    managed = env_keys.find(name)
    tag = "" if managed else "  (unmanaged key — stored, but no client reset hook)"
    _print(f"    {name} set -> {env_keys.mask(value)} (saved to .env, applied now){tag}")


def _config_keys(ctx, args):
    """`/config key …` — the API-key (secrets) front end."""
    import env_keys

    sub = args[0].lower() if args else "list"

    if sub in ("list", "ls", "status"):
        _list_keys()
        return

    if sub == "get":
        if len(args) < 2:
            _print("  usage: /config key get <name>")
            return
        name = _resolve_key_name(args[1])
        if name:
            _print(f"    {name} = {env_keys.mask(env_keys.get(name))}")
        return

    if sub == "set":
        _set_key(args[1:])
        return

    if sub in ("unset", "clear") or is_remove_verb(sub):
        if len(args) < 2:
            _print("  usage: /config key unset <name>")
            return
        name = _resolve_key_name(args[1])
        if name is None:
            return
        if env_keys.unset_value(name):
            _print(f"    {name} removed from .env and the live environment.")
        else:
            _print(f"    {name} was not set.")
        return

    # Shorthand — `set` is implied: a pasted secret sets its own key; a key name alone shows it,
    # with a value sets it. (Inert while KNOWN_KEYS is empty; kept for a managed key's return.)
    if env_keys.detect(args[0]):
        _set_key(args)
        return
    if env_keys.resolve(args[0]):
        if len(args) == 1:
            name = env_keys.resolve(args[0]).name
            _print(f"    {name} = {env_keys.mask(env_keys.get(name))}")
        else:
            _set_key(args)
        return

    _print(f"  unknown /config key subcommand: {sub!r} — try: list, set, unset, get, or a key name")


@command(
    "config",
    "View or edit runtime config (config.yaml) and API keys (.env); edits persist by default.",
    usage="/config | /config <dotted.key> [value] [--session] | /config persist <key> | /config setup | /config key … | /config reload",
    details="""
With no args, prints the key runtime settings (active_tier, runtime.max_iterations,
runtime.auto_approve), the resolved paths, and which API keys are set.

With a dotted key, reads that value; with a key and a value, sets it AND writes it back to
config.yaml in place (comments and layout preserved) so it survives a restart — persisting is the
default, because a setting you change should stick. Append --session (or --session-only) to apply
an edit for this session only, without touching config.yaml:
  /config runtime.max_iterations 12            set AND persist to config.yaml (the default)
  /config runtime.max_iterations 12 --session  set for this session only
  /config runtime.max_iterations --save        persist the CURRENT value unchanged
  /config persist runtime.max_iterations       same thing, as a verb
(--save / -s is still accepted; it's the default now, so it only matters as the bare
"persist the current value" form above. A key not already in config.yaml can't be persisted, so
it stays session-only with a note.) `/config reload` re-reads config.yaml from disk, discarding
any unsaved session-only edits.

/config context — the runtime readout (context window + fill, per-role windows, CPU/RAM/GPU) and
the live num_ctx setter (folded in from the old /context):
  /config context                   show the window + current fill + hardware snapshot
  /config context 16384 [--session] resize every local role's window (persists; --session = live only)
  /config context auto [--session]  back to per-model capability windows
The window also frees itself automatically: older turns are compacted as the context fills
(runtime.auto_compact past runtime.compact_threshold) — no manual step needed.

/config setup (doctor, check) — first-run / health check: is the Ollama daemon up, are the
active tier's models pulled, and are the keys the tier needs set, with the exact command to fix
each genuine gap (keys the tier doesn't need are just labeled optional). When models are missing
and the daemon is up, it offers — y/N, default no — to run the `ollama pull`s for you inline with
live progress. Runs automatically on first launch; re-run any time with /config setup.

API keys live in .env, not config.yaml, so they have their own subcommand (already persistent):
  /config key                       list managed keys and whether each is set (masked)
  /config key set MY_VAR <value>    store a custom env var (ALL-CAPS name), e.g. for an MCP
                                    server's ${VAR} expansion
  /config key unset <name>          remove a key from .env and the environment (clear, or any
                                    removal verb — remove/rm/delete/del/forget/drop — works)
  /config key get <name>            show one key (masked)

No Saturn feature takes an API key today: web search is keyless (DuckDuckGo) and extraction/
inference are local — the managed-key registry (env_keys.py) is empty. (The Anthropic/OpenAI
keys left with the cloud-model shelve, 2026-07-03; the Tavily key left with the API-less web
pivot, 2026-07-06 — an already-set key in .env is untouched, just unmanaged.)

Model/tier keys rebuild the cached models on next use; an embedder change re-embeds the corpus.
To change model bindings specifically, /models is the friendlier front end.

Examples:
  /config                              show the summary
  /config setup                        check the install (Ollama, models, keys)
  /config runtime.max_iterations       read one key
  /config runtime.max_iterations 12    set it and persist to config.yaml (the default)
  /config runtime.max_iterations 12 --session   set it for this session only
  /config key set MY_VAR <value>       store a custom env var (e.g. for an MCP server)
  /config reload                       re-read config.yaml from disk
""",
)
def _config(ctx, args):
    from config import get_config, reload

    cfg = get_config()

    if args and args[0].lower() in ("key", "keys", "secret", "secrets"):
        _config_keys(ctx, args[1:])
        return

    if args and args[0].lower() in ("doctor", "setup", "check", "health"):
        _config_doctor(ctx)
        return

    if args and args[0].lower() in ("context", "ctx", "window"):
        _config_context(ctx, args[1:])
        return

    if args and args[0].lower() == "persist":
        if len(args) < 2:
            _print("  usage: /config persist <dotted.key>   (writes the current value to config.yaml)")
            return
        _persist_key(cfg, args[1])
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
        if not env_keys.KNOWN_KEYS:
            _print("    none needed — web search is keyless, inference is local (/config key "
                   "for custom vars)")
        for k in env_keys.KNOWN_KEYS:
            _print(f"    {k.name:<20}: {'set' if env_keys.is_set(k.name) else 'not set'}")
        _print("  (workspace & memory resolve live; documents/db_sqlite apply on re-ingest/restart)")
        _print("  set a value: /config <dotted.key> <value>   (e.g. /config runtime.max_iterations 12)")
        _print("  manage keys: /config key   ·   window + hardware: /config context   (see /config --help)")
        return

    if args[0].lower() == "reload":  # case-insensitive like every sibling subcommand match
        reload()
        from core.llms import reset_models
        reset_models()
        _print("  config.yaml reloaded from disk (any session edits discarded).")
        _resync_rag_after_model_change()
        return

    # Settings PERSIST to config.yaml by default now (a changed setting should survive a restart —
    # what people expect from "change a setting"); --session / --session-only applies an edit for
    # this session only. --save / -s is still accepted (it's the default) so old habits keep
    # working. (split_persist_flags: case-insensitive, any position, exact token only — the bare
    # words save/persist are NOT flags, guarded below.)
    rest, session, save = split_persist_flags(args)
    if not rest:
        _print("  usage: /config <dotted.key> [value] [--session]")
        return
    key = rest[0]
    values = rest[1:]

    if not values:
        if save and not session:
            # A bare `--save` with no value persists the CURRENT value (unchanged convention —
            # identical to /config persist <key>; it mutates nothing live).
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

    # The old grammar took a trailing bare save/persist as the flag; storing it silently as
    # value text now would corrupt the setting — refuse and point at the one spelling instead.
    if values[-1].lower() in ("save", "persist", "--persist"):
        _print(f"  did you mean --save? (the bare word {values[-1]!r} is not a persist flag; "
               "use --save / -s, or /config persist <key>) — nothing set")
        return

    value = " ".join(values)

    # Section guard: a dotted key naming a whole MAPPING must refuse — cfg.set would replace the
    # mapping with a scalar (every `web.*`-style read silently degrades to defaults for the rest
    # of the session), and a later persist would rewrite the bare `web:` header line into
    # `web: foo` above its still-indented children: unparseable YAML that kills the next launch
    # (_set_yaml_scalar now also refuses headers, but the session-side corruption must stop here
    # too). The guard lives in this handler, NOT in Config.set — /models legitimately replaces a
    # {provider, model} role-binding dict with a bare scalar model id via cfg.set.
    current = cfg.get(key, _MISSING)
    if isinstance(current, dict):
        children = ", ".join(f"{key}.{child}" for child in current)
        _print(f"  {key} is a section, not a setting — set one of: {children}")
        return
    if isinstance(current, list):
        _print(f"  {key} is a list, not a scalar setting — edit config.yaml by hand")
        return

    # A key the config has never seen still sets — the default-tolerant knobs and absent
    # per-tier role leaves must keep working on a config.yaml predating
    # them — but the success-shaped line is replaced
    # with a plain warning so a misspelled safety knob can't masquerade as applied. The
    # suggestion snapshots the leaf list BEFORE the set, so the typo never suggests itself.
    suggestion = _did_you_mean(cfg, key) if current is _MISSING else ""
    cfg.set(key, value)
    if current is _MISSING:
        # A key not already in config.yaml can't be persisted (persist edits existing scalar
        # leaves), so it is inherently session-only — say so plainly instead of attempting a
        # persist that would only fail.
        _print(f"  note: {key!r} was not an existing config key{suggestion} "
               "(set for this session; only keys the code reads have any effect, and a key that "
               "is not already in config.yaml cannot be persisted)")
    elif session:
        _print(
            f"  {key} = {cfg.get(key)!r}  (session only — omit --session to save to config.yaml)"
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


def _config_context(ctx, args):
    """`/config context [size|auto [--save]]` — the runtime readout (context window + fill,
    per-role windows, CPU/RAM/GPU) and the live num_ctx setter. Folded in from the old standalone
    /context 2026-07-07: the readout and the setter both belong under /config, the one runtime-
    settings front door."""
    from config import get_config
    from core.llms import reset_models, active_context_window, model_id
    from tui import ui

    cfg = get_config()

    if not args:
        window = active_context_window()
        used = int(ctx.state.get("context_tokens", 0) or 0)
        if cfg.num_ctx_override:
            source = "override · runtime.num_ctx"
        else:
            source = f"auto · {model_id('tool_caller')} capability"
        per_role = {role: cfg.num_ctx_for(model_id(role)) for role in _ROLES}
        ui.show_context(window, used, source, per_role)
        # The hardware half of the runtime readout (absorbed from the old /system).
        from tui.system_monitor import get_system_metrics

        ui.show_system_metrics(get_system_metrics())
        return

    args, session, save = split_persist_flags(args)
    if not args:
        if save and not session:  # `/config context --save` persists the CURRENT window
            _persist_key(cfg, "runtime.num_ctx")
            return
        _print("  usage: /config context <size>|auto [--session]")
        return

    arg = args[0].lower()
    if arg in ("auto", "default", "reset", "off"):
        cfg.set("runtime.num_ctx", None)
        reset_models()
        _print("  context window -> auto (each model uses its capability window).")
        _print("  models rebuild on next use.")
        if session:
            _print("  (session only; omit --session to persist to config.yaml.)")
        else:
            _persist_key(cfg, "runtime.num_ctx")
        return

    try:
        n = int(arg)
    except ValueError:
        _print(f"  not a size: {args[0]!r} — usage: /config context <size>|auto [--session]")
        return
    if n < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {n} (minimum {_MIN_NUM_CTX}).")
        return

    cfg.set("runtime.num_ctx", n)
    reset_models()
    _print(f"  context window -> {n:,} tokens for all local roles.")
    _print("  models rebuild on next use.")
    if session:
        _print("  (session only; omit --session to persist to config.yaml.)")
    else:
        _persist_key(cfg, "runtime.num_ctx")


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


# Why a missing OPTIONAL key is fine, per key (default: the active tier simply doesn't bind the
# provider). Display notes only — which keys are REQUIRED is derived live in _required_keys.
# Empty since the API-less web pivot (2026-07-06); repopulate alongside env_keys.KNOWN_KEYS.
_OPTIONAL_KEY_NOTES: dict[str, str] = {}


def _required_keys(cfg) -> set[str]:
    """The env keys the ACTIVE tier genuinely needs. With cloud model support SHELVED
    (2026-07-03) and the web tools API-less (2026-07-06) nothing can require a key. Kept as the
    seam: when cloud returns, this re-derives one required key per cloud provider bound to a
    role (via the restored llms provider→key table)."""
    return set()


def _key_line(name: str, is_set: bool, required) -> str:
    """One api-key status line for the doctor (ASCII-only; the caller indents). The fix arrow
    (`->`) is reserved for keys the active tier genuinely needs: an unset OPTIONAL key is labeled
    optional with why that's fine, so a privacy-pitched product's first screen never reads as a
    list of API keys to go get."""
    if is_set:
        return f"ok       {name:<18} set"
    if name in required:
        return f"MISSING  {name:<18} -> /config key set {name} <value>"
    note = _OPTIONAL_KEY_NOTES.get(name, "not needed by the active tier")
    return f"optional {name:<18} ({note})"


def _tier_honesty_line(cfg) -> "str | None":
    """The doctor's closing tier-honesty line, when the active tier is the smallest preset
    configured (the quick-install default): the smallest local models are fine for trying Saturn
    but measurably less reliable at structured plans and tool calls, and the first screen should
    say so instead of leaving it to be discovered. Convention: config.yaml's `tiers:` mapping is
    written smallest -> largest (YAML mapping order is preserved), so the FIRST declared tier IS
    the smallest — declaration order, never a size heuristic (summing context windows ranks
    capacity, not model size: a 4B/128k model outsums a 32B/32k one). None when the active tier
    isn't the first-declared preset, or only one tier exists (nothing to upgrade to)."""
    tiers = cfg.get("tiers", {}) or {}
    names = list(tiers)
    if len(names) < 2 or cfg.active_tier != names[0]:
        return None
    model = cfg.model_for_role("tool_caller").model
    return (f"you are on the smallest model tier ({model}) - fine for trying Saturn; "
            "/models to upgrade if your hardware allows.")


def _should_offer_pull(missing: list, daemon_up: bool, interactive: bool) -> bool:
    """Whether the doctor ends with the inline `ollama pull` offer: something to pull, a
    reachable daemon to pull into, and a human at a TTY to ask — off-TTY/headless never
    prompts."""
    return bool(missing) and daemon_up and interactive


def _stdin_is_tty() -> bool:
    import sys

    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _offer_pull(missing: list[str]) -> None:
    """Offer to run the `ollama pull`s the doctor just prescribed, inline, instead of telling the
    user to go run them elsewhere at the exact moment they can't do anything else. Plain prompt
    (ui.ask tears down any live bar first — input never runs under a rich.Live — and answers no
    on Ctrl-C/EOF), default NO. The pulls run as ordinary foreground subprocesses with live
    output (ollama prints each download's size and progress) — the same trust boundary as the
    installer pulling the default models — and are Ctrl-C-able; a failure stops the batch with
    the copy-paste commands still on screen above."""
    from tui import ui

    # ASCII-only like the rest of the doctor (no » glyph) — see _config_doctor.
    reply = ui.ask(
        f"[y] pull {len(missing)} missing model(s) now? "
        "(sizes shown as each pull starts)  [y/N] "
    ).lower()
    if reply not in ("y", "yes"):
        _print("  ok - pull when ready:")
        for m in missing:
            _print(f"    ollama pull {m}")
        _print("")
        return

    import subprocess

    for m in missing:
        _print(f"  pulling {m} ...")
        try:
            rc = subprocess.run(["ollama", "pull", m]).returncode
        except KeyboardInterrupt:
            _print("")
            _print(f"  pull cancelled - finish later with `ollama pull {m}`.")
            _print("")
            return
        except OSError as exc:
            _print(f"  could not run `ollama pull {m}`: {exc}")
            _print("")
            return
        if rc != 0:
            _print(f"  `ollama pull {m}` exited with code {rc} - fix and re-run /config setup.")
            _print("")
            return

    from core.llms import check_models

    remaining = check_models()
    if remaining:
        _print(f"  models pulled - {len(remaining)} other thing(s) still to fix "
               "(re-run /config setup for details).")
    else:
        _print("  models pulled - the active tier is ready to run.")
    _print("")


def _config_doctor(ctx) -> None:
    """First-run / health view: Ollama up? active-tier models pulled? needed API keys set? Each
    GENUINE gap is paired with the exact fix; optional keys are labeled optional (an unset one is
    not a thing to fix). When local models are missing, the daemon is reachable, and a human is
    at a TTY, it ends with a y/N offer to run the `ollama pull`s inline — the one consented
    action it can take; otherwise it remains a read-only diagnostic."""
    from config import get_config
    from core.llms import check_models, list_local_models, ollama_reachable
    from commands._utils import _ROLES
    import env_keys

    cfg = get_config()
    # ASCII-only output on purpose: this is the FIRST command a fresh install runs, possibly in a
    # legacy console where the fancy glyphs the other commands use would raise an encoding error.
    _print("")
    _print(f"  saturday.ai setup check - tier '{cfg.active_tier}'")

    # Ollama daemon.
    up = ollama_reachable()
    _print(f"    ollama daemon   {'ok (reachable)' if up else 'DOWN (not reachable)'}")
    if not up:
        _print("        -> install from https://ollama.com, then run `ollama serve`")

    # Local (Ollama-served) models the active tier binds (+ the embedder), and whether each is
    # pulled. Cloud-bound roles don't belong in this list: their gaps (key, package) surface via
    # check_models below, and `ollama pull` could never fix them.
    have = {m.name for m in list_local_models()} if up else set()
    bound = {
        spec.model
        for spec in (cfg.model_for_role(r) for r in _ROLES)
        if spec.provider == "ollama"
    }
    bound.add(cfg.embedder_model)
    _print("    models")
    from core.llms import _model_present
    missing: list[str] = []
    for m in sorted(bound):
        if not up:
            _print(f"        ?        {m}")
        elif _model_present(m, have):
            _print(f"        ok       {m}")
        else:
            missing.append(m)
            _print(f"        MISSING  {m}   -> run `ollama pull {m}`")

    # API keys — the fix arrow only for keys the active tier genuinely needs (see _key_line).
    # The managed registry is empty today (keyless web, local inference) — one honest line
    # instead of a bare header, so a fresh install's first screen never hints at keys to go get.
    _print("    api keys (.env)")
    required = _required_keys(cfg)
    if not env_keys.KNOWN_KEYS:
        _print("        ok       none needed (web search is keyless; inference is local)")
    for k in env_keys.KNOWN_KEYS:
        _print(f"        {_key_line(k.name, env_keys.is_set(k.name), required)}")

    # MCP servers (only when any are configured — most installs have none).
    from tools import mcp_client
    statuses = mcp_client.status()
    if statuses:
        _print("    mcp servers")
        for s in statuses:
            if s.state == "connected":
                _print(f"        ok       {s.name:<18} {len(s.tools)} tool(s)")
            elif s.state == "disabled":
                _print(f"        off      {s.name:<18} disabled in config.yaml")
            else:
                _print(f"        FAILED   {s.name:<18} {s.error or s.state}   -> /mcp reload")

    problems = check_models()
    _print("")
    if problems:
        _print(f"  {len(problems)} thing(s) to fix before this tier runs cleanly:")
        for p in problems:
            _print(f"    - {p}")
    else:
        _print("  all set - the active tier is ready to run.")
    honesty = _tier_honesty_line(cfg)
    if honesty:
        _print(f"  {honesty}")
    _print("")

    if _should_offer_pull(missing, up, _stdin_is_tty()):
        _offer_pull(missing)
