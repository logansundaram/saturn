"""
Runtime-inventory commands — what the agent is running on and with, in one module (the /help
"observability" readouts; consolidated from one-file-per-command 2026-06-11):

  /tools    the registered tools + risk tiers
  /models   installed models; bind roles / the embedder / the tier
  /context  context window + fill, CPU/RAM/GPU; resize the window
  /mcp      MCP server status + remote tools; reload
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import _ROLES, _resync_rag_after_model_change, is_list_verb, split_save_flag
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


def _bind(cfg, target: str, model: str, *, save: bool = False) -> None:
    """Bind a role / all roles / the embedder to a local Ollama model id (a bare scalar in
    config.yaml). A legacy {provider, model} cloud mapping on the role is simply overwritten —
    cloud support is shelved (2026-07-03), and rebinding is how a stale mapping gets fixed."""
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
            cfg.set(f"tiers.{cfg.active_tier}.roles.{role}", model)
        reset_models()
        _print(f"  all roles -> {model} on tier '{cfg.active_tier}'{tag}.")
        keys = [f"tiers.{cfg.active_tier}.roles.{role}" for role in _ROLES]
    else:
        cfg.set(f"tiers.{cfg.active_tier}.roles.{target}", model)
        reset_models()
        _print(f"  {target} -> {model} on tier '{cfg.active_tier}'{tag}.")
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
    usage="/models [list] | /models <role|all|embedder> <id> [--save] | /models tier <name> [--save]",
    details="""
With no args, pings the local Ollama daemon, renders every installed model (size, params,
quantization, and what each currently drives) as a numbered table, then drops into an interactive
picker: choose a model by number, then choose what it should drive. Picking a chat model defaults
to 'all' roles (the common 'run everything locally on this model' case); picking an embed-only
model defaults to the embedder. Blank input cancels at either step; the target prompt accepts a
trailing --save like the direct forms below.

You can also bind directly, without the picker:
  /models list                       just the table + bindings, no picker (`ls` works too)
  /models all <id> [--save]          point every role at one model
  /models <role> <id> [--save]       re-point one role
  /models embedder <id> [--save]     switch the embedding model (re-embeds the corpus)
  /models tier <name> [--save]       switch the whole hardware tier

Roles: planner, tool_caller, synthesizer, utility, judge.

Every switch is session-only by default and rebuilds the cached models on next use; append --save
to also write the change back to config.yaml in place (the same dotted key(s) the session edit
sets, via the /config <key> --save machinery). Any change that moves the embedder re-embeds the
document corpus.

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

    args, save = split_save_flag(args)

    # The old cross-provider grammar (--provider <p> / a bare provider as 3rd arg) left with the
    # cloud-model shelve (2026-07-03): refuse it loudly rather than binding something surprising.
    if any(a.lower() == "--provider" for a in args):
        _print("  --provider was removed with the cloud-model shelve — models are local Ollama "
               "ids only; usage: /models <role|all> <model_id> [--save]")
        return

    if not args:
        local = list_local_models()
        ui.show_models(local, bindings, cfg.active_tier, cfg.embedder_model, numbered=True)
        _models_picker(ctx, cfg, local, save=save)
        return

    sub = args[0].lower()

    if is_list_verb(sub):
        # The non-interactive view (`ollama list`-style): the same table, no picker.
        ui.show_models(list_local_models(), bindings, cfg.active_tier, cfg.embedder_model)
        return

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
        _print(f"  unknown target: {role} (roles: {', '.join(_ROLES)}; or 'all'/'embedder'/'tier'/'list')")
        return
    if len(args) < 2:
        _print(f"  usage: /models {role} <model_id> [--save]")
        return
    if len(args) > 2:
        # The old bare-positional provider spelling — gone with the cloud shelve.
        _print(f"  too many arguments — usage: /models {role} <model_id> [--save] "
               "(the provider argument was removed with the cloud-model shelve).")
        return
    # A scalar bind; if the role still carried a legacy {provider, model} cloud mapping
    # (pre-shelve config.yaml), this simply replaces it — rebinding IS the fix.
    _bind(cfg, role, args[1], save=save)


# ── /context ─────────────────────────────────────────────────────────────────────────────────
_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos


@command(
    "context",
    "Runtime readout: context window + fill, CPU/RAM/GPU; resize the window.",
    aliases=("ctx",),
    usage="/context [size|auto [--save]]",
    details="""
With no args, the combined runtime readout: the active context window (num_ctx), how full it was
on the last LLM call (a fill bar + token count), the per-role windows, and a point-in-time
CPU/RAM/GPU/VRAM snapshot (the same gauges the status bar streams live during a turn, colored
green→yellow→red by load). (/system was folded in here — one runtime readout, not two.)

With a size, sets the Ollama context window for every local role at once (session only) and
rebuilds the models so it takes effect next turn — num_ctx is fixed when a model is built, so
the cache is dropped. With `auto`, clears the override so each model uses its capability
context_window from config.yaml. Add --save to also persist runtime.num_ctx to config.yaml so
the setting survives a restart.

Note: without an explicit window Ollama silently caps at 2048 tokens; this binds the full
window so the gauge is truthful.

To free the window up rather than resize it, /compact runs the LLM summary over older turns.

Examples:
  /context                show the window + current fill
  /context 16384          resize every local role to 16k tokens (session only)
  /context 16384 --save   resize AND persist to config.yaml
  /context auto           back to per-model capability windows
  /context --save         persist the CURRENT window setting unchanged (like /config persist)
""",
)
def _context(ctx, args):
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

    args, save = split_save_flag(args)
    if not args:
        if save:  # `/context --save` persists the CURRENT window setting (like /config persist)
            from commands.config import _persist_key
            _persist_key(cfg, "runtime.num_ctx")
            return
        _print("  usage: /context <size>|auto [--save]")
        return

    arg = args[0].lower()
    if arg in ("auto", "default", "reset", "off"):
        cfg.set("runtime.num_ctx", None)
        reset_models()
        _print("  context window -> auto (each model uses its capability window).")
        _print("  models rebuild on next use.")
        if save:
            from commands.config import _persist_key
            _persist_key(cfg, "runtime.num_ctx")
        else:
            _print("  (session only; add --save to persist to config.yaml.)")
        return

    try:
        n = int(arg)
    except ValueError:
        _print(f"  not a size: {args[0]!r} — usage: /context <size>|auto [--save]")
        return
    if n < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {n} (minimum {_MIN_NUM_CTX}).")
        return

    cfg.set("runtime.num_ctx", n)
    reset_models()
    _print(f"  context window -> {n:,} tokens for all local roles.")
    _print("  models rebuild on next use.")
    if save:
        from commands.config import _persist_key
        _persist_key(cfg, "runtime.num_ctx")
    else:
        _print("  (session only; add --save to persist to config.yaml.)")


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
