"""
Slash-command layer for the interactive CLI loop (`agent.py`).

This is a *REPL meta-command* system — it sits in front of the graph, not inside it.
A line that starts with `/` never reaches the agent; it's intercepted here, run against
the live session, and the loop continues. Everything else is a normal user turn.

Design mirrors the rest of the repo: a flat registry (`COMMANDS`) populated by a
`@command(...)` decorator, handlers kept stateless apart from the `CommandContext` they're
handed. Commands that are fully wired set `implemented=True`; the rest are deliberate
scaffolds (`implemented=False`) that print their intended behaviour and a TODO so the
surface exists and is discoverable before the plumbing lands.

To implement a scaffolded command: flip `implemented=True` and fill in the handler body.
To add a new one: write a handler and decorate it. Nothing else in the loop changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Callable, Optional

# Handlers reach into these registries directly (same pattern the nodes use).
from registry import tool as TOOLS, TOOL_RISK, risk_of
from document_registry import read_documents_manifest, read_workspace_manifest


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """Everything a handler is allowed to touch. Handlers mutate `state` in place (or
    reassign via `ctx.state = ...`), and flip `should_quit` to end the loop.

    `make_initial_state` is injected so handlers don't import from `agent.py` (which would
    be circular) — it's how `/reset` gets a clean state without knowing its shape."""

    state: dict
    make_initial_state: Callable[[], dict]
    db_path: str
    show_ui: bool = True
    auto_approve: bool = False
    should_quit: bool = False


Handler = Callable[[CommandContext, list[str]], None]


@dataclass
class SlashCommand:
    name: str
    summary: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    usage: str = ""
    implemented: bool = True


COMMANDS: dict[str, SlashCommand] = {}
_ALIASES: dict[str, str] = {}


def command(
    name: str,
    summary: str,
    *,
    aliases: tuple[str, ...] = (),
    usage: str = "",
    implemented: bool = True,
) -> Callable[[Handler], Handler]:
    """Register a slash command. The handler keeps its bare signature so it stays unit-testable."""

    def register(fn: Handler) -> Handler:
        cmd = SlashCommand(
            name=name,
            summary=summary,
            handler=fn,
            aliases=aliases,
            usage=usage,
            implemented=implemented,
        )
        COMMANDS[name] = cmd
        for alias in aliases:
            _ALIASES[alias] = name
        return fn

    return register


def is_command(line: str) -> bool:
    """A command is any non-empty line whose first non-space character is `/`."""
    return line.lstrip().startswith("/")


@cache
def command_completions() -> list[tuple[str, str]]:
    """(token, summary) pairs for every invocable command — canonical names and aliases. Handed to
    the input prompt, which Tab-completes the leading `/command` token and derives its live-highlight
    set (valid vs. typo) from the same tokens. Aliases borrow their target's name as the summary;
    scaffolds are tagged so an unimplemented command is obvious in the menu. Sorted for a stable menu
    order. Cached — the command registry is frozen after import and callers treat the list as
    read-only."""
    out: list[tuple[str, str]] = []
    for cmd in COMMANDS.values():
        summary = cmd.summary + ("" if cmd.implemented else "  (scaffold)")
        out.append((cmd.name.lower(), summary))
        for alias in cmd.aliases:
            out.append((alias.lower(), f"alias for /{cmd.name}"))
    return sorted(out)


def _print(line: str = "") -> None:
    # Single choke point for output so a future swap to rich/Textual is one edit.
    print(line)


def _todo(cmd: SlashCommand, args: list[str]) -> None:
    """Uniform 'not wired yet' notice for scaffolded commands."""
    _print(f"  /{cmd.name} is scaffolded but not implemented yet.")
    _print(f"  intended: {cmd.summary}")
    if cmd.usage:
        _print(f"  usage:    {cmd.usage}")
    _print("  (flip implemented=True and fill in the handler in commands.py)")


def dispatch(line: str, ctx: CommandContext) -> None:
    """Parse and run a slash command. Always returns; signals exit via `ctx.should_quit`.
    Unknown commands and arg errors are reported, never raised, so the REPL never dies on a typo."""
    parts = line.lstrip().lstrip("/").split()
    if not parts:
        _print("  empty command — try /help")
        return

    key = parts[0].lower()
    args = parts[1:]

    name = key if key in COMMANDS else _ALIASES.get(key)
    cmd = COMMANDS.get(name) if name else None
    if cmd is None:
        _print(f"  unknown command: /{key} - try /help")
        return

    if not cmd.implemented:
        _todo(cmd, args)
        return

    try:
        cmd.handler(ctx, args)
    except Exception as exc:  # a bad command must not kill the session
        _print(f"  /{cmd.name} failed: {exc}")


# ---------------------------------------------------------------------------
# Implemented commands
# ---------------------------------------------------------------------------


@command("help", "List all slash commands.", aliases=("?", "h"))
def _help(ctx: CommandContext, args: list[str]) -> None:
    _print("  slash commands:")
    for cmd in sorted(COMMANDS.values(), key=lambda c: c.name):
        mark = " " if cmd.implemented else "*"
        names = "/" + cmd.name
        if cmd.aliases:
            names += " (" + ", ".join("/" + a for a in cmd.aliases) + ")"
        _print(f"   {mark} {names:<22} {cmd.summary}")
    _print("   * = scaffolded, not yet implemented")


@command("quit", "Exit the agent.", aliases=("exit", "q"))
def _quit(ctx: CommandContext, args: list[str]) -> None:
    ctx.should_quit = True


@command("clear", "Clear the terminal screen.", aliases=("cls",))
def _clear(ctx: CommandContext, args: list[str]) -> None:
    import os

    os.system("cls" if os.name == "nt" else "clear")


@command("state", "Dump a summary of the current agent state.")
def _state(ctx: CommandContext, args: list[str]) -> None:
    s = ctx.state
    _print("  agent state:")
    _print(f"    messages      : {len(s.get('messages', []))}")
    _print(f"    current_query : {s.get('current_query', '')!r}")
    _print(f"    iteration     : {s.get('iteration', 0)}")
    _print(f"    verified      : {s.get('verified', False)}")
    _print(f"    plan steps    : {len(s.get('plan', []))}")
    _print(f"    tools_called  : {s.get('tools_called', [])}")
    _print(f"    docs_retrieved: {len(s.get('documents_retrieved', []))}")
    if "--full" in args:
        _print("    ---- raw ----")
        _print(f"    {s}")


@command("reset", "Reset the conversation (clears messages + per-turn state).", aliases=("new",))
def _reset(ctx: CommandContext, args: list[str]) -> None:
    ctx.state = ctx.make_initial_state()
    _print("  conversation reset — fresh state, no message history.")


@command("tools", "View the registered tools and their risk tiers.")
def _tools(ctx: CommandContext, args: list[str]) -> None:
    _print("  registered tools:")
    for t in TOOLS:
        risk = risk_of(t.name)
        desc = (t.description or "").strip().splitlines()
        first = desc[0] if desc else ""
        _print(f"    [{risk:<14}] {t.name:<22} {first}")


@command("docs", "View ingested RAG documents and workspace files.", aliases=("documents",))
def _docs(ctx: CommandContext, args: list[str]) -> None:
    docs = read_documents_manifest().strip()
    ws = read_workspace_manifest().strip()
    _print("  === ingested documents (RAG corpus) ===")
    _print(docs if docs else "  (none ingested)")
    _print("")
    _print("  === workspace files ===")
    _print(ws if ws else "  (empty)")


_ROLES = ("planner", "tool_caller", "synthesizer", "utility", "judge")


def _resync_rag_after_model_change() -> None:
    """A model/tier change may have swapped the active tier's embedder. reset_models() only
    drops the chat-model caches, so re-embed the corpus here if the embedder actually changed."""
    from rag import sync_to_config

    if sync_to_config():
        _print("  embedder changed -> re-embedded the document corpus.")


@command(
    "model",
    "Show or switch the per-role model bindings / hardware tier.",
    usage="/model | /model tier <name> | /model <role> <model_id> [provider]",
)
def _model(ctx: CommandContext, args: list[str]) -> None:
    # Phase 3: roles resolve to models through config + the get_model factory, so switching is
    # just re-pointing config and dropping the cached models (nodes call get_model at run time).
    from config import get_config
    from llms import model_id, capability_of, reset_models

    cfg = get_config()

    if not args:
        _print(f"  active tier: {cfg.active_tier}   (embedder: {cfg.embedder_model})")
        _print("  role bindings:")
        for role in _ROLES:
            mid = model_id(role)
            cap = capability_of(role)
            flags = []
            if cap.supports_tools:
                flags.append("tools")
            if cap.supports_structured_output:
                flags.append("structured")
            if cap.supports_vision:
                flags.append("vision")
            _print(f"    {role:<12} {mid:<22} [{', '.join(flags) or 'no caps'}]")
        _print("  switch: /model tier <name>   or   /model <role> <model_id> [provider]")
        return

    if args[0] == "tier":
        if len(args) < 2:
            _print("  usage: /model tier <name>")
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

    role = args[0]
    if role not in _ROLES:
        _print(f"  unknown role: {role} (roles: {', '.join(_ROLES)})")
        return
    if len(args) < 2:
        _print(f"  usage: /model {role} <model_id> [provider]")
        return
    new_model = args[1]
    key = f"tiers.{cfg.active_tier}.roles.{role}"

    # A role can be a bare model id (served by the tier's default provider) or a
    # {provider, model} mapping (a per-role provider override, e.g. cloud-hybrid pointing the
    # planner at anthropic). Take an explicit provider as the 3rd arg; otherwise preserve the
    # provider already on this role so we don't silently re-point it at the tier default.
    if len(args) > 2:
        provider = args[2]
    else:
        existing = cfg.get(key)
        provider = existing.get("provider") if isinstance(existing, dict) else None

    if provider:
        cfg.set(key, {"provider": provider, "model": new_model})
        bound = f"{provider}:{new_model}"
    else:
        cfg.set(key, new_model)
        bound = new_model
    reset_models()
    _print(f"  {role} -> {bound} on tier '{cfg.active_tier}' (session only).")
    _print("  edit config.yaml to make it permanent.")
    _resync_rag_after_model_change()


@command("system", "Show live CPU, RAM, and GPU metrics.", aliases=("sys",))
def _system(ctx: CommandContext, args: list[str]) -> None:
    from system_monitor import get_system_metrics
    import ui

    ui.show_system_metrics(get_system_metrics())


# ---------------------------------------------------------------------------
# Scaffolded commands (implemented=False) — surface exists, plumbing pending.
# Each summary doubles as the spec for whoever implements it.
# ---------------------------------------------------------------------------


@command(
    "history",
    "Print the conversation messages for this session.",
    aliases=("hist",),
    usage="/history [n]",
)
def _history(ctx: CommandContext, args: list[str]) -> None:
    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  (no messages yet)")
        return

    # Optional positional n -> show only the last n messages.
    shown = messages
    if args:
        try:
            n = int(args[0])
            shown = messages[-n:] if n > 0 else messages
        except ValueError:
            _print(f"  ignoring non-numeric count: {args[0]!r}")

    _print(f"  conversation history ({len(shown)} of {len(messages)} messages):")
    for i, msg in enumerate(shown, start=len(messages) - len(shown) + 1):
        role = getattr(msg, "type", type(msg).__name__)
        content = msg.content
        if isinstance(content, list):  # multimodal / structured content blocks
            content = " ".join(str(p) for p in content)
        content = " ".join(str(content).split())  # collapse whitespace to one line
        if len(content) > 100:
            content = content[:99] + "…"
        line = f"    {i:>3}  {role:<6} {content}"
        # AI turns that only call tools carry empty content — surface the calls instead.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            names = ", ".join(tc.get("name", "?") for tc in tool_calls)
            line += f"[tool_calls: {names}]" if not content else f"  → {names}"
        _print(line)


@command(
    "plan",
    "Show the most recent plan and step statuses.",
    usage="/plan",
)
def _plan(ctx: CommandContext, args: list[str]) -> None:
    import ui

    _print("  most recent plan:")
    ui.render_plan(ctx.state.get("plan", []))


@command(
    "trace",
    "Show recent runs/events from the trace DB.",
    usage="/trace [n]",
)
def _trace(ctx: CommandContext, args: list[str]) -> None:
    import sqlite3

    n = 5
    if args:
        try:
            n = max(1, int(args[0]))
        except ValueError:
            _print(f"  ignoring non-numeric count: {args[0]!r}")

    conn = sqlite3.connect(ctx.db_path)
    try:
        rows = conn.execute(
            "SELECT run_id, started_at, status, query, "
            "(SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS n_events "
            "FROM runs r ORDER BY run_id DESC LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        _print("  (no runs recorded yet)")
        return

    _print(f"  last {len(rows)} run(s) — newest first:")
    for run_id, started_at, status, query, n_events in rows:
        when = (started_at or "")[:19].replace("T", " ")  # trim ISO microseconds
        q = " ".join(str(query or "").split())
        if len(q) > 60:
            q = q[:59] + "…"
        _print(f"    #{run_id:<4} {when}  {str(status):<7} {n_events:>2}ev  {q}")


@command(
    "reingest",
    "Rebuild the RAG vector store + cache from database/documents/.",
    usage="/reingest",
)
def _reingest(ctx: CommandContext, args: list[str]) -> None:
    from rag import sync

    s = sync(force=True, verbose=False)
    n = s["added"] + s["updated"]
    _print(f"  reingested {n} document(s) — full rebuild, disk cache refreshed.")


@command(
    "ingest",
    "Add a document to the RAG corpus and embed it.",
    usage="/ingest <path>",
)
def _ingest(ctx: CommandContext, args: list[str]) -> None:
    from rag import ingest_file

    if not args:
        _print("  usage: /ingest <path-to-file>")
        return
    path = " ".join(args)  # tolerate unquoted paths with spaces
    s = ingest_file(path)
    if s["added"] or s["updated"]:
        _print(f"  ingested {Path(path).name} — +{s['added']} ~{s['updated']} (cache updated).")
    else:
        _print(f"  {Path(path).name} already up to date in the corpus.")


@command(
    "forget",
    "Remove a document from the RAG corpus and drop its vectors.",
    aliases=("remove",),
    usage="/forget <name>",
)
def _forget(ctx: CommandContext, args: list[str]) -> None:
    from rag import forget_document

    if not args:
        _print("  usage: /forget <document-name>")
        return
    name = " ".join(args)
    if forget_document(name):
        _print(f"  removed {name} from the corpus — vectors + manifest entry dropped.")
    else:
        _print(f"  no document named {name} in the corpus (see /docs).")


@command(
    "workspace",
    "List files in the read/write workspace sandbox.",
    aliases=("ws",),
    usage="/workspace",
)
def _workspace(ctx: CommandContext, args: list[str]) -> None:
    from config import get_config

    ws = get_config().path("workspace")
    _print(f"  workspace: {ws}")
    if not ws.exists():
        _print("  (workspace directory does not exist yet)")
        return

    # Skip dotfiles (the .manifest.md is one) — same convention rag.iter_documents uses.
    entries = sorted(p for p in ws.iterdir() if not p.name.startswith("."))
    if not entries:
        _print("  (empty)")
        return

    for p in entries:
        if p.is_dir():
            _print(f"    {p.name + '/':<32} <dir>")
        else:
            size = p.stat().st_size
            _print(f"    {p.name:<32} {size:>9,} B")


@command(
    "tool",
    "Enable or disable a tool for this session.",
    usage="/tool <name> on|off",
    implemented=False,
)
def _tool_toggle(ctx: CommandContext, args: list[str]) -> None:
    # TODO: maintain a session set of disabled tools and filter the bound tool list. Needs the
    #       model-rebind path (same blocker as /model) since tools are bound at import in llms.py.
    ...


_RISK_TIERS = ("read_only", "side_effecting", "destructive")


@command(
    "risk",
    "Override a tool's approval risk tier for this session.",
    usage="/risk <tool> read_only|side_effecting|destructive",
)
def _risk(ctx: CommandContext, args: list[str]) -> None:
    import registry

    if not args:
        _print("  current risk tiers:")
        for t in TOOLS:
            _print(f"    {risk_of(t.name):<14} {t.name}")
        _print("  set: /risk <tool> read_only|side_effecting|destructive")
        return

    if len(args) < 2:
        _print(f"  usage: /risk <tool> {'|'.join(_RISK_TIERS)}")
        return

    name, tier = args[0], args[1]
    if name not in registry.tools_by_name:
        _print(f"  unknown tool: {name} (see /tools)")
        return
    if tier not in _RISK_TIERS:
        _print(f"  unknown tier: {tier} (choose one of {', '.join(_RISK_TIERS)})")
        return

    old = risk_of(name)
    registry.TOOL_RISK[name] = tier
    _print(f"  {name}: {old} -> {tier} (session only; the approval gate reads this live).")


@command(
    "autoapprove",
    "Toggle the approval gate (auto-approve side-effecting tools).",
    aliases=("yolo",),
    usage="/autoapprove on|off",
)
def _autoapprove(ctx: CommandContext, args: list[str]) -> None:
    new = _parse_toggle(args, ctx.auto_approve)
    if new is None:
        _print(f"  usage: /autoapprove on|off   (currently {'on' if ctx.auto_approve else 'off'})")
        return
    ctx.auto_approve = new
    if new:
        # Loud banner: the safety gate is off — side-effecting/destructive tools run unprompted.
        _print("  ┏━ ⚠  AUTO-APPROVE ON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        _print("  ┃  the approval gate is DISABLED. every tool call —")
        _print("  ┃  including side-effecting and destructive ones —")
        _print("  ┃  will run WITHOUT asking. /autoapprove off to restore.")
        _print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        _print("  auto-approve off — the approval gate is back on.")


def _parse_toggle(args: list[str], current: bool) -> Optional[bool]:
    """Parse an on/off argument. No arg flips the current value; an unrecognized arg returns None."""
    if not args:
        return not current
    val = args[0].lower()
    if val in ("on", "true", "yes", "1"):
        return True
    if val in ("off", "false", "no", "0"):
        return False
    return None


@command(
    "verbose",
    "Toggle live node/plan UI streaming on or off.",
    usage="/verbose on|off",
)
def _verbose(ctx: CommandContext, args: list[str]) -> None:
    new = _parse_toggle(args, ctx.show_ui)
    if new is None:
        _print(f"  usage: /verbose on|off   (currently {'on' if ctx.show_ui else 'off'})")
        return
    ctx.show_ui = new
    _print(f"  live node/plan trace {'on' if new else 'off'}.")


@command(
    "save",
    "Save the current session (messages + state) to disk.",
    usage="/save [name]",
    implemented=False,
)
def _save(ctx: CommandContext, args: list[str]) -> None:
    # TODO: serialize ctx.state["messages"] (langchain has message (de)serializers) to a named file.
    ...


@command(
    "load",
    "Load a previously saved session.",
    usage="/load <name>",
    implemented=False,
)
def _load(ctx: CommandContext, args: list[str]) -> None:
    # TODO: deserialize a saved session into ctx.state. Pairs with /save.
    ...


@command(
    "config",
    "View or edit runtime config (config.yaml). Edits are session-only.",
    usage="/config | /config <dotted.key> [value] | /config reload",
)
def _config(ctx: CommandContext, args: list[str]) -> None:
    from config import get_config, reload

    cfg = get_config()

    if not args:
        _print("  runtime config:")
        _print(f"    active_tier           : {cfg.active_tier}")
        _print(f"    runtime.max_iterations: {cfg.max_iterations}")
        _print(f"    runtime.auto_approve  : {cfg.auto_approve}")
        _print("  paths:")
        for name in ("documents", "workspace", "memory", "db_sqlite"):
            _print(f"    {name:<10}: {cfg.get('paths.' + name)}")
        _print("  (workspace & memory resolve live; documents/db_sqlite apply on re-ingest/restart)")
        _print("  set a value: /config <dotted.key> <value>   (e.g. /config runtime.max_iterations 12)")
        return

    if args[0] == "reload":
        reload()
        from llms import reset_models

        reset_models()  # discarding session edits changes the bindings; drop the stale caches
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
    # Model/tier keys need the cached models dropped to take effect.
    if key.startswith("tiers.") or key == "active_tier":
        from llms import reset_models

        reset_models()
        _print("  (models will rebuild on next use)")
        _resync_rag_after_model_change()
