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


def command_names() -> set[str]:
    """Every invocable command token (canonical names + aliases), lowercased, no leading slash.
    Handed to the input prompt so it can highlight a typed `/command` live (valid vs. typo)."""
    return {n.lower() for n in COMMANDS} | {a.lower() for a in _ALIASES}


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


@command("model", "Show the active model (pass a name to switch - scaffolded).", usage="/model [name]")
def _model(ctx: CommandContext, args: list[str]) -> None:
    # Reading the current model is safe today; the switch path is the scaffolded part.
    from llms import llm

    current = getattr(llm, "model", "unknown")
    if not args:
        _print(f"  active model: {current}")
        _print("  to switch: /model <name>  (not wired yet — see TODO below)")
        return
    # TODO(model-switch): `llm`, `llm_with_tools`, `llm_with_plan` are module-level singletons
    # bound at import, and every node imports them directly. Hot-swapping needs a mutable holder
    # (e.g. a ModelManager the nodes call through) — this lands with the Phase 3 config.yaml
    # role/tier system. Until then, edit the model string in llms.py and restart.
    _print(f"  switching model is not wired yet (requested: {args[0]}, current: {current}).")
    _print("  edit llms.py and restart for now; see TODO(model-switch) in commands.py.")


# ---------------------------------------------------------------------------
# Scaffolded commands (implemented=False) — surface exists, plumbing pending.
# Each summary doubles as the spec for whoever implements it.
# ---------------------------------------------------------------------------


@command(
    "history",
    "Print the conversation messages for this session.",
    aliases=("hist",),
    usage="/history [n]",
    implemented=False,
)
def _history(ctx: CommandContext, args: list[str]) -> None:
    # TODO: pretty-print ctx.state["messages"] (role + content), optionally last n.
    ...


@command(
    "plan",
    "Show the most recent plan and step statuses.",
    usage="/plan",
    implemented=False,
)
def _plan(ctx: CommandContext, args: list[str]) -> None:
    # TODO: render ctx.state["plan"] via ui.show_plan (the last plan persists in state).
    ...


@command(
    "trace",
    "Show recent runs/events from the trace DB.",
    usage="/trace [n]",
    implemented=False,
)
def _trace(ctx: CommandContext, args: list[str]) -> None:
    # TODO: query the `runs`/`events` tables in ctx.db_path (see trace.py) and print the last n.
    ...


@command(
    "reingest",
    "Rebuild the RAG vector store from database/documents/.",
    usage="/reingest",
    implemented=False,
)
def _reingest(ctx: CommandContext, args: list[str]) -> None:
    # TODO: re-run rag.build_ingest().invoke({"documents": []}); the store is in-memory so this
    # is the only way to pick up newly dropped files without a restart.
    ...


@command(
    "workspace",
    "List files in the read/write workspace sandbox.",
    aliases=("ws",),
    usage="/workspace",
    implemented=False,
)
def _workspace(ctx: CommandContext, args: list[str]) -> None:
    # TODO: list database/workspace/ (see document_registry.WORKSPACE_DIR), skipping .manifest.md.
    ...


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


@command(
    "risk",
    "Override a tool's approval risk tier for this session.",
    usage="/risk <tool> read_only|side_effecting|destructive",
    implemented=False,
)
def _risk(ctx: CommandContext, args: list[str]) -> None:
    # TODO: mutate registry.TOOL_RISK[name] = tier after validating the tier. Cheap + safe;
    #       good first command to implement.
    ...


@command(
    "autoapprove",
    "Toggle the approval gate (auto-approve side-effecting tools).",
    aliases=("yolo",),
    usage="/autoapprove on|off",
    implemented=False,
)
def _autoapprove(ctx: CommandContext, args: list[str]) -> None:
    # TODO: thread a session flag into the approver passed to run_turn (ui.ask_approval). When on,
    #       the approver returns True without prompting. Keep a loud banner — this disables the safety gate.
    ...


@command(
    "verbose",
    "Toggle live node/plan UI streaming on or off.",
    usage="/verbose on|off",
    implemented=False,
)
def _verbose(ctx: CommandContext, args: list[str]) -> None:
    # TODO: flip ctx.show_ui and thread it into _make_on_update(show_ui=...) in agent.py.
    ...


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
    "View or edit runtime config (Phase 3 config.yaml).",
    usage="/config [key] [value]",
    implemented=False,
)
def _config(ctx: CommandContext, args: list[str]) -> None:
    # TODO: read/write the role/tier config.yaml once Phase 3 lands (see SATURDAY_MVP_PLAN.md).
    ...
