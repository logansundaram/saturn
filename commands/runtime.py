"""
Runtime-inventory commands — what the agent is running on and with, in one module (the /help
"observability" readouts; consolidated from one-file-per-command 2026-06-11):

  /tools    the registered tools + risk tiers
  /models   installed models; bind roles / the embedder / the tier
  /mcp      MCP server status + remote tools; reload

(/context folded into /config as `/config context` 2026-07-07 — the runtime readout + num_ctx
setter belong under the one runtime-settings front door.)
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import _ROLES, _resync_rag_after_model_change, is_list_verb, split_persist_flags
from core import model_family
from tools.registry import tool as TOOLS, risk_of


# ── /tools ───────────────────────────────────────────────────────────────────────────────────
@command(
    "tools",
    "View the registered tools and their risk tiers.",
    details="""
Lists every tool the agent can call, each with its approval risk tier
(read_only, side_effecting, destructive) and a one-line description.

The risk tier drives the approval gate: read_only runs freely, the others prompt (unless
auto-approve is on). Override a tier for the session with /policy risk; open/close the gate
with /policy open.

Example:
  /tools
""",
)
def _tools(ctx, args):
    from config import get_config
    from tui import ui

    gated = sum(1 for t in TOOLS if not get_config().auto_approves(risk_of(t.name)))
    ui.section(
        "tools",
        f"{len(TOOLS)} registered  ·  {gated} gated  ·  auto-approve ≤ {get_config().auto_approve}",
    )
    rows = []
    for t in TOOLS:
        risk = risk_of(t.name)
        desc = (t.description or "").strip().splitlines()
        first = desc[0] if desc else ""
        rows.append((t.name, (risk, ui.risk_style(risk)), (first, "dim")))
    ui.table(rows)


# ── /models ──────────────────────────────────────────────────────────────────────────────────
_BIND_TARGETS = ("all", *_ROLES, "embedder")


def _persist_bindings(cfg, keys: list[str]) -> None:
    """Persist session-set binding keys to config.yaml through the one persist seam (the same
    machinery as /config <key> --save)."""
    from commands.config import _persist_key

    for key in keys:
        _persist_key(cfg, key)


def print_family_refusal(model: str) -> None:
    """THE family-gate refusal, in one place. `/models` binds through _bind; `/config` writes the
    same `tiers.*.roles.*` keys directly and must refuse identically — two hand-written messages
    would drift, and the second door silently persisting what the first refuses is worse than a
    wording drift (2026-08-16)."""
    _print(f"  {model} is outside the supported model family.")
    _print("  Saturday.ai binds qwen3.5 / qwen3.6 / qwen3.8 only — confidence coloring is")
    _print("  calibrated per model, so a red run is only a true claim for a measured one.")
    _print("  supported:")
    for key, tag in model_family.SIZE_LADDER:
        _print(f"    {key:<6} {tag}")
    _print("  switch the whole tier with `/models tier <size>`.")


def _bind(cfg, target: str, model: str, *, session: bool = False) -> None:
    """Bind a role / all roles / the embedder to a local Ollama model id (a bare scalar in
    config.yaml). The change PERSISTS to config.yaml by default (a model switch should stick);
    session=True applies it live only. A legacy {provider, model} cloud mapping on the role is
    simply overwritten — cloud support is shelved (2026-07-03), and rebinding is how a stale
    mapping gets fixed."""
    from core.llms import reset_models

    # The family gate (2026-08-16). The EMBEDDER is exempt — it is not a chat model, has no
    # raw-mode template and produces no logprobs, so no calibration claim rides on it.
    if target != "embedder" and not model_family.in_family(model):
        print_family_refusal(model)
        return

    tag = " (session only)" if session else ""

    if target == "embedder":
        key = f"tiers.{cfg.active_tier}.embedder"
        cfg.set(key, model)
        reset_models()
        _print(f"  embedder -> {model} on tier '{cfg.active_tier}'{tag}.")
        if session:
            _print("  omit --session to save to config.yaml.")
        else:
            _persist_bindings(cfg, [key])
        _resync_rag_after_model_change()
        return

    if target == "all":
        for role in _ROLES:
            cfg.set(f"tiers.{cfg.active_tier}.roles.{role}", model)
        reset_models()
        _print(f"  all roles -> {model} on tier '{cfg.active_tier}'{tag}.")
        keys = [f"tiers.{cfg.active_tier}.roles.{role}" for role in _ROLES]
    else:
        cfg.set(f"tiers.{cfg.active_tier}.roles.{target}", model)
        reset_models()
        _print(f"  {target} -> {model} on tier '{cfg.active_tier}'{tag}.")
        keys = [f"tiers.{cfg.active_tier}.roles.{target}"]
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        _persist_bindings(cfg, keys)
    _resync_rag_after_model_change()


def _models_picker(ctx, cfg, local, *, session: bool = False) -> None:
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
    # The target prompt accepts a trailing --session too, same as the direct-bind forms.
    tokens, picked_session, _ = split_persist_flags(tokens)
    session = session or picked_session
    target = tokens[0] if tokens else default
    if target not in _BIND_TARGETS:
        _print(f"  unknown target: {target} (choose one of {', '.join(_BIND_TARGETS)})")
        return
    _bind(cfg, target, choice.name, session=session)


# Parameter counts as `ollama show` reports them — display only, so the listing reads the same
# whether or not the tag is pulled locally. (Defined before _tier_rows, which reads it.)
_PARAMS_BY_CLASS = {
    "800m": "873M", "2b": "2.3B", "4b": "4.7B", "9b": "9.7B", "27b": "27.3B", "35b": "36.0B",
}


def _tier_model(cfg, key: str) -> str:
    """What a tier actually binds, read straight off the tiers mapping (dict access, never the
    dotted cfg.get path — a legacy tier name may contain a dot)."""
    tier = (cfg.get("tiers", {}) or {}).get(key) or {}
    roles = tier.get("roles", {}) or {}
    entry = roles.get("synthesizer") or next(iter(roles.values()), None)
    if isinstance(entry, dict):
        entry = entry.get("model", "")
    return str(entry or "")


def _tier_rows(active: "str | None" = None, cfg=None) -> list:
    """One row per tier the CONFIG defines, carrying the metrics the listing shows instead of a
    nickname: bound model, parameters, runtime + architectural context window, and whether the
    model has a confidence calibration behind it.

    Walks the config's OWN tiers, not the ladder (2026-08-16): `/models tier <name>` validates
    against config.yaml, so a listing built from the ladder offered an upgrading user six rows
    they could not select while hiding the three they had. A tier whose name is not a size class
    is marked `legacy` and shows what it really binds."""
    from config import get_config
    from core import confidence

    if cfg is None:
        cfg = get_config()
    if active is None:
        active = cfg.active_tier
    tiers = list(cfg.get("tiers", {}) or {})
    classes = model_family.classes()
    rows = []
    for key in tiers or classes:
        ladder = key in classes
        tag = model_family.tag_for(key) if ladder else _tier_model(cfg, key)
        cap = cfg.capability_of(tag)
        rows.append({
            "key": key,
            "model": tag,
            "params": _PARAMS_BY_CLASS.get(key, ""),
            "ctx": cap.context_window,
            "max_ctx": cap.max_context_window,
            "calibrated": confidence.calibration_for(tag) is not None,
            "legacy": not ladder,
            "active": key == active,
        })
    return rows


def _config_migrations() -> dict:
    """This session's family substitutions, so the listing never claims the file's value is
    what is running."""
    import config as _config

    return _config.migrated_bindings()


@command(
    "models",
    "List installed models; pick or switch what drives each role / the embedder.",
    aliases=("model",),
    usage="/models [list] | /models <role|all|embedder> <id> [--session] | /models tier <name> [--session]",
    details="""
With no args, pings the local Ollama daemon, renders every installed model (size, params,
quantization, and what each currently drives) as a numbered table, then drops into an interactive
picker: choose a model by number, then choose what it should drive. Picking a chat model defaults
to 'all' roles (the common 'run everything locally on this model' case); picking an embed-only
model defaults to the embedder. Blank input cancels at either step; the target prompt accepts a
trailing --session like the direct forms below.

You can also bind directly, without the picker:
  /models list                          just the table + bindings, no picker (`ls` works too)
  /models all <id> [--session]          point every role at one model
  /models <role> <id> [--session]       re-point one role
  /models embedder <id> [--session]     switch the embedding model (re-embeds the corpus)
  /models tier <name> [--session]       switch the whole hardware tier

Roles: planner, tool_caller, synthesizer, utility, judge.

Every switch PERSISTS to config.yaml by default (a model switch should survive the next launch)
and rebuilds the cached models on next use — the change writes the same dotted key(s) the session
edit sets, via the /config persist machinery. Append --session (or --session-only) to apply a
switch for this session only, without touching config.yaml. Any change that moves the embedder
re-embeds the document corpus.

Models are local Ollama ids only — cloud model support is SHELVED (2026-07-03, local-first is
the edge), and the old --provider grammar left with it. Rebinding a role that still carries a
legacy {provider, model} cloud mapping (a pre-shelve config.yaml) replaces it with the local bind.
""",
)
def _models(ctx, args):
    from config import get_config
    from core.llms import model_id, reset_models, list_local_models
    from tui import ui

    cfg = get_config()
    bindings = {role: model_id(role) for role in _ROLES}

    args, session, save = split_persist_flags(args)

    # The old cross-provider grammar (--provider <p> / a bare provider as 3rd arg) left with the
    # cloud-model shelve (2026-07-03): refuse it loudly rather than binding something surprising.
    if any(a.lower() == "--provider" for a in args):
        _print("  --provider was removed with the cloud-model shelve — models are local Ollama "
               "ids only; usage: /models <role|all> <model_id> [--save]")
        return

    if not args:
        local = list_local_models()
        ui.show_models(local, bindings, cfg.active_tier, cfg.embedder_model, numbered=True)
        _models_picker(ctx, cfg, local, session=session)
        return

    sub = args[0].lower()

    if is_list_verb(sub):
        # The non-interactive view (`ollama list`-style): the same table, no picker.
        ui.show_models(list_local_models(), bindings, cfg.active_tier, cfg.embedder_model)
        return

    if sub == "tier":
        if len(args) < 2:
            from tui import ui

            ui.section("tiers", "switch with /models tier <name>")
            tiers = _tier_rows()
            rows = [
                (
                    ("* " if r["active"] else "  ") + r["key"],
                    r["model"] or "—",
                    ("legacy tier", "yellow") if r["legacy"] else (r["params"], "dim"),
                    (f"{r['ctx']} / {r['max_ctx']}", "dim"),
                    ("calibrated", "green") if r["calibrated"] else ("uncalibrated", "yellow"),
                )
                for r in tiers
            ]
            ui.table(rows)
            if any(r["legacy"] for r in tiers):
                # These are the tiers this config can actually switch between; the size classes
                # are not among them, so name the bind that DOES work here.
                _print("  legacy tier names predate the size-class ladder. Rebind a tier's")
                _print("  models in place with `/models all <tag>`:")
                for key, tag in model_family.SIZE_LADDER:
                    _print(f"    {key:<6} {tag}")
            migrated = _config_migrations()
            for original, replacement in migrated.items():
                _print(f"  note: '{original}' in config.yaml is running as '{replacement}'.")
            return
        tier = args[1]
        # Dict membership, not the dotted cfg.get("tiers.<tier>") path lookup — a tier key
        # containing a literal dot would otherwise be misread as two path segments and always
        # report unknown (hit this with the size-class key "0.8b", renamed to "800m" for the
        # same reason — see core/model_family.py).
        if tier not in cfg.get("tiers", {}):
            _print(f"  unknown tier: {tier} (defined: {list(cfg.get('tiers', {}))})")
            return
        cfg.set("active_tier", tier)
        reset_models()
        tag = " (session only)" if session else ""
        _print(f"  active tier -> {tier}; models will rebuild on next use{tag}.")
        if session:
            _print("  omit --session to save to config.yaml.")
        else:
            _persist_bindings(cfg, ["active_tier"])
        _resync_rag_after_model_change()
        return

    if sub == "embedder":
        if len(args) < 2:
            _print("  usage: /models embedder <model_id> [--session]")
            return
        _bind(cfg, "embedder", args[1], session=session)
        return

    if sub == "all":
        if len(args) < 2:
            _print("  usage: /models all <model_id> [--session]")
            return
        _bind(cfg, "all", args[1], session=session)
        return

    role = sub
    if role not in _ROLES:
        _print(f"  unknown target: {role} (roles: {', '.join(_ROLES)}; or 'all'/'embedder'/'tier'/'list')")
        return
    if len(args) < 2:
        _print(f"  usage: /models {role} <model_id> [--session]")
        return
    if len(args) > 2:
        # The old bare-positional provider spelling — gone with the cloud shelve.
        _print(f"  too many arguments — usage: /models {role} <model_id> [--session] "
               "(the provider argument was removed with the cloud-model shelve).")
        return
    # A scalar bind; if the role still carried a legacy {provider, model} cloud mapping
    # (pre-shelve config.yaml), this simply replaces it — rebinding IS the fix.
    _bind(cfg, role, args[1], session=session)


# ── /mcp ─────────────────────────────────────────────────────────────────────────────────────
@command(
    "mcp",
    "MCP servers: connection status, the remote tools they add, reconnect.",
    usage="/mcp [list | reload]",
    details="""
Saturn is an MCP client: servers declared under `mcp.servers:` in config.yaml are connected at
startup and every remote tool they expose registers behind the SAME risk-tier approval gate as
the local tools (named `mcp_<server>_<tool>`; they show in /tools and the planner sees them).

Trust model — a remote tool never picks its own tier. Every MCP tool fails closed to
`destructive` (always prompts) unless YOU relax it: per server with `risk:` in config.yaml, or
per tool with /policy risk <tool> <tier> [--save]. The server's own annotations (read-only etc.)
are shown here as advisory hints only — they never drive the gate.

  /mcp           server connection status + the remote tools each one added (also: list, ls,
                 status)
  /mcp reload    tear down every connection, re-read `mcp:` from config.yaml, reconnect and
                 re-register the tools (the recovery path after a config edit or a crashed
                 server; session-only /config edits to `mcp.*` apply too). Persisted
                 /policy risk --save overrides re-apply; session-only overrides reset to the
                 declared tier, like every session-only setting.

Adding a server (config.yaml; secrets via ${VAR} from .env — /config key):

  mcp:
    servers:
      github:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
      internal-docs:
        url: https://mcp.example.com/mcp
        risk: read_only

Examples:
  /mcp
  /mcp reload
""",
)
def _mcp(ctx, args):
    from tools import mcp_client
    from tui import ui

    if args:
        sub = args[0].lower()
        if sub in ("reload", "reconnect", "refresh"):
            _print("  reconnecting MCP servers…")
            mcp_client.reload()
        elif is_list_verb(sub) or sub == "status":
            pass  # the default status view below
        else:
            # A typo'd /mcp relod must error, not silently render status as if it reloaded.
            _print(f"  unknown subcommand '{args[0]}' — usage: /mcp [list | reload]")
            return

    statuses = mcp_client.status()
    if not statuses:
        if mcp_client.configured():
            # Configured but nothing connected this session (e.g. servers added to config.yaml
            # after startup) — a reload picks them up.
            _print("  MCP servers are configured but not loaded — run /mcp reload.")
        else:
            _print("  no MCP servers configured.")
            _print("  declare them under `mcp.servers:` in config.yaml (see /mcp --help for an")
            _print("  example), then run /mcp reload. Remote tools always face the approval gate")
            _print("  unless you lower their risk tier yourself.")
        return

    connected = [s for s in statuses if s.state == "connected"]
    n_tools = sum(len(s.tools) for s in statuses)
    ui.section(
        "mcp",
        f"{len(connected)}/{len(statuses)} server(s) connected  ·  {n_tools} remote tool(s)"
        "  ·  unconfigured risk fails closed to destructive",
    )

    state_style = {
        "connected": ui.risk_style("read_only"),       # green — healthy
        "disabled": "dim",
        "starting": "dim",
        "disconnected": ui.risk_style("side_effecting"),
        "error": ui.risk_style("destructive"),
    }
    rows = []
    for s in statuses:
        label = s.name + (f"  ({s.server_info})" if s.server_info else "")
        rows.append(
            (
                label,
                (s.state, state_style.get(s.state, "")),
                (f"{s.transport}: {s.target}", "dim"),
            )
        )
    ui.table(rows)
    for s in statuses:
        if s.error and s.state != "connected":
            _print(f"    {s.name}: {s.error}")

    if n_tools:
        _print("  remote tools (hints are the server's own claims — advisory, never the gate)")
        tool_rows = []
        for s in connected:
            for t in s.tools:
                risk = risk_of(t.name)
                desc = (t.hints + "  " if t.hints else "") + t.description
                tool_rows.append((t.name, (risk, ui.risk_style(risk)), (desc, "dim")))
        ui.table(tool_rows)

    for p in mcp_client.problems():
        ui.warn(p)
