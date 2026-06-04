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

import re
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
    # Long-form, `git <cmd> --help`-style blurb shown by `/help <name>` and `/<cmd> --help`.
    # Free-form text; leading/trailing blank lines are trimmed, embedded newlines preserved.
    details: str = ""


COMMANDS: dict[str, SlashCommand] = {}
_ALIASES: dict[str, str] = {}


def command(
    name: str,
    summary: str,
    *,
    aliases: tuple[str, ...] = (),
    usage: str = "",
    implemented: bool = True,
    details: str = "",
) -> Callable[[Handler], Handler]:
    """Register a slash command. The handler keeps its bare signature so it stays unit-testable.

    `summary` is the one-liner for `/help`; `details` is the long-form blurb surfaced by
    `/help <name>` and `/<cmd> --help` (see `_show_help`)."""

    def register(fn: Handler) -> Handler:
        cmd = SlashCommand(
            name=name,
            summary=summary,
            handler=fn,
            aliases=aliases,
            usage=usage,
            implemented=implemented,
            details=details,
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
    _print(f"  see /{cmd.name} --help for the full spec.")


_HELP_FLAGS = {"--help", "-h"}


def _show_help(cmd: SlashCommand) -> None:
    """`git <cmd> --help`-style detail view for one command: signature, status, usage, and the
    long-form `details` blurb. Shared by `/help <name>` and the `--help`/`-h` flag any command
    accepts (intercepted in `dispatch`, so handlers never see it)."""
    title = "/" + cmd.name
    if cmd.aliases:
        title += "   aliases: " + ", ".join("/" + a for a in cmd.aliases)
    _print("")
    _print(f"  {title}")
    _print(f"  {'─' * min(len(title), 60)}")
    _print(f"  {cmd.summary}")
    if not cmd.implemented:
        _print("  (scaffolded — prints intended behaviour only; not yet wired)")
    _print("")
    _print(f"  usage:  {cmd.usage or ('/' + cmd.name)}")
    if cmd.details:
        _print("")
        for line in cmd.details.strip("\n").splitlines():
            _print(f"  {line}" if line.strip() else "")
    _print("")


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

    # `--help`/`-h` on any command shows its detail view and short-circuits — handlers (and the
    # scaffold notice) never run, so even unimplemented commands document themselves.
    if args and args[0].lower() in _HELP_FLAGS:
        _show_help(cmd)
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


@command(
    "help",
    "List all slash commands, or detail one.",
    aliases=("?", "h"),
    usage="/help [command]",
    details="""
With no argument, prints the full command list (scaffolds marked `*`).
With a command name, prints its detailed help — identical to `/<command> --help`.

Every command also accepts --help / -h directly.

Examples:
  /help              list all commands
  /help risk         detail one command
  /risk --help       same thing, the git-style way
""",
)
def _help(ctx: CommandContext, args: list[str]) -> None:
    # `/help <command>` (or `/help /command`) -> the same detail view as `/<command> --help`.
    if args and args[0].lower() not in _HELP_FLAGS:
        key = args[0].lstrip("/").lower()
        name = key if key in COMMANDS else _ALIASES.get(key)
        cmd = COMMANDS.get(name) if name else None
        if cmd is None:
            _print(f"  unknown command: /{key} - try /help")
            return
        _show_help(cmd)
        return

    _print("  slash commands:")
    for cmd in sorted(COMMANDS.values(), key=lambda c: c.name):
        mark = " " if cmd.implemented else "*"
        names = "/" + cmd.name
        if cmd.aliases:
            names += " (" + ", ".join("/" + a for a in cmd.aliases) + ")"
        _print(f"   {mark} {names:<22} {cmd.summary}")
    _print("   * = scaffolded, not yet implemented")
    _print("  /help <command> or /<command> --help for details on one.")


@command(
    "quit",
    "Exit the agent.",
    aliases=("exit", "q"),
    details="""
Ends the interactive session and returns you to the shell. In-process conversation
memory is discarded; the trace DB and RAG corpus on disk are untouched.

Example:
  /quit
""",
)
def _quit(ctx: CommandContext, args: list[str]) -> None:
    ctx.should_quit = True


@command(
    "clear",
    "Clear the terminal screen.",
    aliases=("cls",),
    details="""
Clears the visible terminal (runs `cls` on Windows, `clear` elsewhere). Affects only the
screen — conversation state, message history, and the trace are all left intact. Use
/reset to actually clear the conversation.

Example:
  /clear
""",
)
def _clear(ctx: CommandContext, args: list[str]) -> None:
    import os

    os.system("cls" if os.name == "nt" else "clear")


@command(
    "state",
    "Dump a summary of the current agent state.",
    usage="/state [--full]",
    details="""
Prints a one-line-per-field summary of the live AgentState: message count, current query,
loop iteration, the verified flag, plan step count, the tools called this turn, and how many
documents were retrieved.

Pass --full to also dump the raw state dict (verbose — useful for debugging, noisy otherwise).

Examples:
  /state
  /state --full
""",
)
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


@command(
    "reset",
    "Reset the conversation (clears messages + per-turn state).",
    aliases=("new",),
    details="""
Rebuilds a clean AgentState: drops the message history and every per-turn field (plan,
iteration, accumulators). Starts a fresh conversation without restarting the process.

Config, model/tier bindings, the RAG corpus, and persistent memory are all unaffected —
only the in-process conversation is cleared.

Example:
  /reset
""",
)
def _reset(ctx: CommandContext, args: list[str]) -> None:
    ctx.state = ctx.make_initial_state()
    _print("  conversation reset — fresh state, no message history.")


@command(
    "tools",
    "View the registered tools and their risk tiers.",
    details="""
Lists every tool the agent can call, each prefixed with its approval risk tier
([read_only], [side_effecting], [destructive]) and a one-line description.

The risk tier drives the approval gate: read_only runs freely, the others prompt (unless
auto-approve is on). Override a tier for the session with /risk; toggle the gate with
/autoapprove.

Example:
  /tools
""",
)
def _tools(ctx: CommandContext, args: list[str]) -> None:
    _print("  registered tools:")
    for t in TOOLS:
        risk = risk_of(t.name)
        desc = (t.description or "").strip().splitlines()
        first = desc[0] if desc else ""
        _print(f"    [{risk:<14}] {t.name:<22} {first}")


@command(
    "docs",
    "View ingested RAG documents and workspace files.",
    aliases=("documents",),
    details="""
Prints two manifests: the ingested RAG corpus (documents the agent can search through the
search_knowledge_base tool) and the workspace sandbox files (where the file tools read/write).

Manage the corpus with /ingest (add), /forget (remove), and /reingest (full rebuild).

Example:
  /docs
""",
)
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


def _set_role_binding(cfg, role: str, model: str, provider: Optional[str]) -> None:
    """Re-point one role's config entry. A bare model id rides the tier's default provider; an
    explicit provider writes the {provider, model} mapping form (cross-provider bindings)."""
    key = f"tiers.{cfg.active_tier}.roles.{role}"
    if provider:
        cfg.set(key, {"provider": provider, "model": model})
    else:
        cfg.set(key, model)


def _bind(cfg, target: str, model: str, provider: Optional[str] = None) -> None:
    """Apply a model binding for `target` (a role name, "all", or "embedder"), drop the cached
    models, and re-embed the corpus if the embedder moved. Centralizes what the picker and the
    argument paths both do, so they can't drift. Session-only — edit config.yaml to persist."""
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
    else:  # a single role
        _set_role_binding(cfg, target, model, provider)
        reset_models()
        bound = f"{provider}:{model}" if provider else model
        _print(f"  {target} -> {bound} on tier '{cfg.active_tier}' (session only).")
    _print("  edit config.yaml to make it permanent.")
    _resync_rag_after_model_change()


# Picker targets: every chat role, plus the two convenience aggregates.
_BIND_TARGETS = ("all", *_ROLES, "embedder")


def _models_picker(ctx: CommandContext, cfg, local) -> None:
    """Interactive selector behind a bare `/models`: pick a pulled model by number, then pick what
    it should drive (default 'all' roles for a chat model, 'embedder' for an embed-only one).
    Cancellable at either prompt with an empty line."""
    import ui

    if not local:
        return  # nothing pulled locally to pick from; the table already said so
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
    # Picker only ever binds local Ollama tags -> ride the tier's default provider (no override).
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
def _models(ctx: CommandContext, args: list[str]) -> None:
    # Phase 3: roles resolve to models through config + the get_model factory, so switching is
    # just re-pointing config and dropping the cached models (nodes call get_model at run time).
    from config import get_config
    from llms import model_id, reset_models, list_local_models
    import ui

    cfg = get_config()
    bindings = {role: model_id(role) for role in _ROLES}

    # No args -> render the installed-model table + bindings, then drop into the picker.
    if not args:
        local = list_local_models()
        ui.show_models(local, bindings, cfg.active_tier, cfg.embedder_model, numbered=True)
        _models_picker(ctx, cfg, local)
        return

    sub = args[0].lower()

    if sub == "tier":
        if len(args) < 2:
            _print("  usage: /models tier <name>")
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

    # A role can be a bare model id (served by the tier's default provider) or a
    # {provider, model} mapping (a per-role provider override, e.g. cloud-hybrid pointing the
    # planner at anthropic). Take an explicit provider as the 3rd arg; otherwise preserve the
    # provider already on this role so we don't silently re-point it at the tier default.
    if len(args) > 2:
        provider = args[2]
    else:
        existing = cfg.get(f"tiers.{cfg.active_tier}.roles.{role}")
        provider = existing.get("provider") if isinstance(existing, dict) else None

    _bind(cfg, role, new_model, provider)


@command(
    "system",
    "Show live CPU, RAM, and GPU metrics.",
    aliases=("sys",),
    details="""
Renders a point-in-time readout of CPU, RAM, and (when available) GPU/VRAM usage as colored
bars in the trace-rail style — green/yellow/red by load. A snapshot, not a live monitor; run
it again for a fresh reading.

Example:
  /system
""",
)
def _system(ctx: CommandContext, args: list[str]) -> None:
    from system_monitor import get_system_metrics
    import ui

    ui.show_system_metrics(get_system_metrics())


_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos


@command(
    "context",
    "Show the model context window and how full it is; resize it.",
    aliases=("ctx",),
    usage="/context [size|auto]",
    details="""
With no args, shows the active context window (num_ctx), how full it was on the last LLM call
(a fill bar + token count), and the per-role windows. The same fill gauge rides the bottom
status bar live during a turn, colored green→yellow→red as it fills.

With a size, sets the Ollama context window for every local role at once (session only) and
rebuilds the models so it takes effect next turn — num_ctx is fixed when a model is built, so
the cache is dropped. With `auto`, clears the override so each model uses its capability
context_window from config.yaml.

Note: without an explicit window Ollama silently caps at 2048 tokens; this binds the full
window so the gauge is truthful. Edit runtime.num_ctx in config.yaml to persist.

Examples:
  /context              show the window + current fill
  /context 16384        resize every local role to 16k tokens
  /context auto         back to per-model capability windows
""",
)
def _context(ctx: CommandContext, args: list[str]) -> None:
    from config import get_config
    from llms import reset_models, active_context_window, model_id
    import ui

    cfg = get_config()

    if not args:
        window = active_context_window()
        used = int(ctx.state.get("context_tokens", 0) or 0)
        override = cfg.num_ctx_override
        if override:
            source = "override · runtime.num_ctx"
        else:
            source = f"auto · {model_id('tool_caller')} capability"
        per_role = {role: cfg.num_ctx_for(model_id(role)) for role in _ROLES}
        ui.show_context(window, used, source, per_role)
        return

    arg = args[0].lower()
    if arg in ("auto", "default", "reset", "off"):
        cfg.set("runtime.num_ctx", None)
        reset_models()
        _print("  context window -> auto (each model uses its capability window; session only).")
        _print("  models rebuild on next use; edit runtime.num_ctx in config.yaml to persist.")
        return

    try:
        n = int(arg)
    except ValueError:
        _print(f"  not a size: {args[0]!r} — usage: /context <size>|auto")
        return
    if n < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {n} (minimum {_MIN_NUM_CTX}).")
        return

    cfg.set("runtime.num_ctx", n)
    reset_models()
    _print(f"  context window -> {n:,} tokens for all local roles (session only).")
    _print("  models rebuild on next use; edit runtime.num_ctx in config.yaml to persist.")


# ---------------------------------------------------------------------------
# Session / inspection / safety commands. Most are live; the few remaining scaffolds
# (implemented=False) print their intended behaviour and document the blocker in `details`.
# ---------------------------------------------------------------------------


@command(
    "history",
    "Print the conversation messages for this session.",
    aliases=("hist",),
    usage="/history [n]",
    details="""
Prints the conversation messages held in memory this session — one line each: index, role
(human / ai / tool / system), and content with whitespace collapsed to a single line. AI turns
that only call tools (empty content) surface the tool names instead.

Pass n to show just the most recent n messages. This is the in-process scratchpad, not the
durable trace — see /trace for the on-disk run record, /reset to clear it.

Examples:
  /history
  /history 10
""",
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
    details="""
Renders the last plan the planner produced — every step with its status glyph and intended
tool. The plan persists in state between turns, so this shows the most recent one (empty until
you've run at least one turn).

Status glyphs:  · pending   ▸ active   ✓ done   ⨯ skipped

Example:
  /plan
""",
)
def _plan(ctx: CommandContext, args: list[str]) -> None:
    import ui

    _print("  most recent plan:")
    ui.render_plan(ctx.state.get("plan", []))


@command(
    "trace",
    "Show recent runs/events from the trace DB.",
    usage="/trace [n]",
    details="""
Lists the most recent runs recorded in the trace database (database/db.sqlite): run id, start
time, status (running / ok / error), event count, and the query. Defaults to the last 5.

Every turn is one run; the node updates streamed within it are its events. Unlike /history
(in-memory, cleared by /reset) this is the durable, queryable record that survives restarts.

Examples:
  /trace
  /trace 20
""",
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
    details="""
Forces a full rebuild of the RAG vector store from database/documents/: re-embeds every
document and refreshes the on-disk cache (vectors.json + index.json).

Slower than the startup sync, which only embeds new/changed files. Reach for this after
editing a document in place (same filename, new content the hash already covers) or to recover
from a corrupted/stale cache. To add or drop a single file, prefer /ingest or /forget.

Example:
  /reingest
""",
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
    details="""
Copies a file into the RAG corpus and embeds it so the search_knowledge_base tool can retrieve
from it. Supported types: pdf, txt, md, json, jsonl. Paths with spaces don't need quoting.

A no-op if the file is already present and unchanged (matched by content hash). See the corpus
with /docs; remove an entry with /forget.

Examples:
  /ingest C:\\notes\\spec.pdf
  /ingest database/documents/handbook.md
""",
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
    details="""
Removes a document from the RAG corpus: drops its vectors from the store and its entry from the
manifest, then re-syncs. Use the document name as shown by /docs.

This affects only the RAG corpus (database/documents/), not the workspace sandbox. To remove
everything and start clean, delete database/documents/ and run /reingest.

Example:
  /forget spec.pdf
""",
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
    details="""
Lists files in the read/write workspace sandbox — the directory the read_file, write_file, and
list_directory tools are confined to — with sizes (directories marked <dir>). Dotfiles,
including the internal .manifest.md, are hidden.

This is distinct from the RAG corpus (see /docs): the workspace is scratch space the agent
writes to, the corpus is the knowledge base it searches.

Example:
  /workspace
""",
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
    details="""
SCAFFOLD — not yet implemented.

Intended: enable or disable a single tool for the session by filtering the bound tool list, so
the agent can't call a disabled tool at all (distinct from /risk, which only changes whether a
call needs approval).

Blocked on the model-rebind path: get_tool_model() binds the full registry once and caches it,
so wiring this means threading a session disabled-set into that builder and keying / invalidating
the cache on it.

Example (planned):
  /tool web_search off
""",
)
def _tool_toggle(ctx: CommandContext, args: list[str]) -> None:
    # TODO: maintain a session set of disabled tools and filter the bound tool list. Needs the
    #       model-rebind path (same blocker as /models) since tools are bound at import in llms.py.
    ...


_RISK_TIERS = ("read_only", "side_effecting", "destructive")


@command(
    "risk",
    "Override a tool's approval risk tier for this session.",
    usage="/risk <tool> read_only|side_effecting|destructive",
    details="""
Changes the approval risk tier of a single tool for this session. The approval gate reads the
tier live, so the change takes effect on the next turn. With no args, lists every tool's current
tier.

Tiers:
  read_only       no side effects; runs freely, never prompts
  side_effecting  writes / external actions; prompts for approval
  destructive     irreversible / dangerous; prompts for approval

Session-only — edit registry.py (TOOL_RISK) to persist. To skip prompting entirely, see
/autoapprove (disables the gate for all tools at once).

Examples:
  /risk                            list current tiers
  /risk write_file destructive     tighten one tool
  /risk web_search side_effecting  require approval for a normally-free tool
""",
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
    details="""
Disables (or re-enables) the human-in-the-loop approval gate for the session. When ON, every
tool call — including side-effecting and destructive ones — runs WITHOUT prompting, and a loud
banner is printed on enable.

⚠  This removes the main safety check. Use it only when you trust the task and the tools.
Prefer /risk to relax a single tool while keeping the gate on. With no argument, flips the
current state.

Examples:
  /autoapprove on
  /autoapprove off
  /yolo            alias — same thing
""",
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
    details="""
Turns the live execution trace — the dim node/plan rail streamed during a turn — on or off.
When off, turns run quietly and only the final response prints.

The trace is still written to the trace DB either way (see /trace), so this only affects what
scrolls live, not what's recorded. With no argument, flips the current state.

Examples:
  /verbose off
  /verbose on
  /verbose       toggle
""",
)
def _verbose(ctx: CommandContext, args: list[str]) -> None:
    new = _parse_toggle(args, ctx.show_ui)
    if new is None:
        _print(f"  usage: /verbose on|off   (currently {'on' if ctx.show_ui else 'off'})")
        return
    ctx.show_ui = new
    _print(f"  live node/plan trace {'on' if new else 'off'}.")


_SESSION_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sessions_dir() -> Path:
    """The saved-sessions directory (config `paths.sessions`), created on first use."""
    from config import get_config

    d = get_config().path("sessions")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_file(name: str) -> Path:
    """Resolve a user-supplied session name to a `<dir>/<safe-stem>.json` path. Strips any
    directory components and a trailing `.json`, then sanitizes to a safe filename so a name
    can never escape the sessions directory."""
    stem = Path(name).name  # drop any path components a user might type
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    stem = _SAFE_NAME.sub("-", stem).strip("-") or "session"
    return _sessions_dir() / f"{stem}.json"


@command(
    "save",
    "Save the current session's messages to disk.",
    usage="/save [name]",
    details="""
Serializes this session's conversation (the message history) to a JSON file under the sessions
directory (database/sessions/ by default; configurable via paths.sessions). Reload it later with
/load to continue where you left off.

With no name, a timestamped one is generated. Names are sanitized to a safe filename, and a
matching name overwrites the existing save. Only messages are persisted — per-turn scratch
(plan, iteration, tool results) is rebuilt fresh on the next turn anyway.

Examples:
  /save                 timestamped autosave
  /save research-thread named save
""",
)
def _save(ctx: CommandContext, args: list[str]) -> None:
    import json
    from datetime import datetime
    from langchain_core.messages import messages_to_dict

    messages = ctx.state.get("messages", [])
    if not messages:
        _print("  nothing to save — no messages in this session yet.")
        return

    name = " ".join(args) if args else "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _session_file(name)
    existed = path.exists()
    payload = {
        "version": _SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages_to_dict(messages),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    note = " (overwrote existing)" if existed else ""
    _print(f"  saved {len(messages)} message(s) -> {path.name}{note}")
    _print(f"  restore it with /load {path.stem}")


@command(
    "load",
    "Load a previously saved session.",
    usage="/load [name]",
    details="""
Restores a session written by /save: rebuilds a clean state and injects the saved message
history, so the conversation continues where it left off (like /reset, but seeded with the saved
messages instead of empty).

With no name, lists the available saves. Config, model bindings, and the RAG corpus are
unaffected — only the conversation is replaced.

Examples:
  /load                 list saved sessions
  /load research-thread restore one
""",
)
def _load(ctx: CommandContext, args: list[str]) -> None:
    import json
    from langchain_core.messages import messages_from_dict

    if not args:
        files = sorted(_sessions_dir().glob("*.json"))
        if not files:
            _print("  no saved sessions yet — use /save [name] first.")
            return
        _print("  saved sessions:")
        for f in files:
            _print(f"    {f.stem}")
        _print("  restore one with /load <name>")
        return

    path = _session_file(" ".join(args))
    if not path.exists():
        _print(f"  no saved session named {path.stem!r} (run /load with no args to list).")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != _SESSION_VERSION:
        _print(f"  warning: session format v{payload.get('version')} != v{_SESSION_VERSION}; "
               "attempting to load anyway.")
    messages = messages_from_dict(payload.get("messages", []))

    # Fresh state seeded with the restored history — mirrors /reset, which is the only other
    # command that swaps state wholesale. agent.py picks up ctx.state after dispatch.
    state = ctx.make_initial_state()
    state["messages"] = messages
    ctx.state = state
    saved_at = payload.get("saved_at", "?")
    _print(f"  loaded {len(messages)} message(s) from {path.name} (saved {saved_at}).")
    _print("  fresh state — conversation history restored.")


@command(
    "config",
    "View or edit runtime config (config.yaml). Edits are session-only.",
    usage="/config | /config <dotted.key> [value] | /config reload",
    details="""
With no args, prints the key runtime settings (active_tier, runtime.max_iterations,
runtime.auto_approve) and the resolved paths.

With a dotted key, reads that value; with a key and a value, sets it for this session only.
`/config reload` re-reads config.yaml from disk, discarding any session edits.

Model/tier keys rebuild the cached models on next use; an embedder change re-embeds the corpus.
To change model bindings specifically, /models is the friendlier front end.

Examples:
  /config                              show the summary
  /config runtime.max_iterations       read one key
  /config runtime.max_iterations 12    set it (session only)
  /config reload                       re-read config.yaml from disk
""",
)
def _config(ctx: CommandContext, args: list[str]) -> None:
    from config import get_config, reload

    cfg = get_config()

    if not args:
        _print("  runtime config:")
        _print(f"    active_tier           : {cfg.active_tier}")
        _print(f"    runtime.max_iterations: {cfg.max_iterations}")
        _print(f"    runtime.auto_approve  : {cfg.auto_approve}")
        _print(f"    runtime.num_ctx       : {cfg.num_ctx_override or 'auto (per-model capability)'}")
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
    # Model/tier keys need the cached models dropped to take effect; so does num_ctx, which is
    # fixed at instantiation (the /context command is the friendlier front end for it).
    if key.startswith("tiers.") or key == "active_tier":
        from llms import reset_models

        reset_models()
        _print("  (models will rebuild on next use)")
        _resync_rag_after_model_change()
    elif key == "runtime.num_ctx":
        from llms import reset_models

        reset_models()
        _print("  (models will rebuild with the new context window on next use)")
