"""
Runtime-inventory commands — what the agent is running on and with (the /help "observability"
readouts):

  /tools    the toolkits: the tools in groups, each turned on or off
  /models   the model page: the hardware, the qwen ladders priced against it, pick a tier /
            embedder; also the first-launch setup and its consented `ollama pull`s
  /mcp      MCP server status + remote tools; reload
"""

from __future__ import annotations

from dataclasses import dataclass

from commands._framework import command, _print
from commands._utils import (
    _resync_rag_after_model_change, is_list_verb, pull_one as _pull_one, run_pulls,
    split_persist_flags,
)
from config import tier_chat_model
from core import model_family
from tools.registry import risk_of


# ── /tools ───────────────────────────────────────────────────────────────────────────────────
# Toolkits (tools/toolspec.TOOLKITS, the switch in tools/registry): the tools in groups the user
# turns on or off. Off is UNBOUND — the model is not shown the tools — so a toggle rebinds the
# model and re-primes the cached prefix. Spec: docs/superpowers/specs/2026-10-05-toolkits-design.md.
_TOOLS_ON = ("on", "enable")
_TOOLS_OFF = ("off", "disable")
_TOOLS_ALL = ("--all", "-a", "all")


def _tool_rows(tools, *, with_toolkit: bool = False) -> list:
    """One row per tool: name, (its toolkit), risk tier, the description's first line."""
    from tools import registry
    from tools.toolspec import toolkit_of
    from tui import ui

    rows = []
    for t in tools:
        risk = registry.risk_of(t.name)
        desc = (t.description or "").strip().splitlines()
        row = [t.name]
        if with_toolkit:
            kit = toolkit_of(t.name) or ""
            row.append((f"{kit} off" if registry.is_off(t.name) else kit, "dim"))
        rows.append((*row, (risk, ui.risk_style(risk)), (desc[0] if desc else "", "dim")))
    return rows


def _toolkit_state(key: str) -> "tuple[str, str]":
    """(the state cell, its style) for one toolkit's row."""
    from tools import registry
    from tools.toolspec import TOOLKITS

    kit = TOOLKITS[key]
    if kit.core:
        return "always on", "dim"
    if kit.managed_by:
        return kit.managed_by, "dim"
    return ("on", "default") if registry.toolkit_on(key) else ("off", "yellow")


def _show_toolkits() -> None:
    from tools import registry
    from tools.toolspec import TOOLKITS
    from tui import ui

    off = registry.off_toolkits()
    ui.section(
        "tools",
        f"{len(registry.all_tools)} tools  ·  {len(registry.tool)} bound  ·  "
        f"{len(off)} toolkit{'' if len(off) == 1 else 's'} off  ·  /tools on|off <toolkit>",
    )
    rows = []
    for key, kit in TOOLKITS.items():
        n = len(registry.toolkit_tools(key))
        if n:
            rows.append((key, (str(n), "dim"), _toolkit_state(key), (kit.about, "dim")))
    ui.table(rows)


def _show_toolkit(key: str) -> None:
    from tools import registry
    from tools.toolspec import TOOLKITS
    from tui import ui

    tools = registry.toolkit_tools(key)
    ui.section(f"tools · {key}",
               f"{_toolkit_state(key)[0]}  ·  {len(tools)} tool{'' if len(tools) == 1 else 's'}"
               f"  ·  {TOOLKITS[key].about}")
    ui.table(_tool_rows(tools))


def _show_all_tools() -> None:
    from config import get_config
    from tools import registry
    from tools.toolspec import TOOLKITS, toolkit_of
    from tui import ui

    gated = sum(1 for t in registry.tool if not get_config().auto_approves(risk_of(t.name)))
    ui.section(
        "tools",
        f"{len(registry.all_tools)} registered  ·  {len(registry.tool)} bound  ·  {gated} gated"
        f"  ·  auto-approve ≤ {get_config().auto_approve}",
    )
    # Grouped by toolkit, in the table's order (a stable sort keeps registration order within).
    order = list(TOOLKITS)
    tools = sorted(registry.all_tools, key=lambda t: order.index(toolkit_of(t.name)))
    ui.table(_tool_rows(tools, with_toolkit=True))


def _no_toolkit(key: str) -> str:
    """Why `key` cannot be switched or shown — worded for the user, with the closest name."""
    import difflib

    from tools.toolspec import TOOLKITS

    hint = difflib.get_close_matches(key, TOOLKITS, n=1)
    return (f"  no toolkit named {key!r}" + (f" — did you mean {hint[0]}?" if hint else "")
            + "  (/tools lists them)")


def _switch_toolkits(keys: list[str], on: bool, *, session: bool) -> None:
    from config import append_block, get_config
    from commands.config import _persist_key
    from core import prime
    from tools import registry
    from tools.toolspec import TOOLKITS

    word = "on" if on else "off"
    if not keys:
        _print(f"  usage: /tools {word} <toolkit> [<toolkit> …] [--session]")
        return
    # The whole line or nothing: one name that cannot be switched refuses every other.
    for key in keys:
        kit = TOOLKITS.get(key)
        if kit is None:
            _print(_no_toolkit(key))
            return
        if kit.core:
            _print(f"  {key} is always on — the engine and the prompt lean on its tools")
            return
        if kit.managed_by:
            _print(f"  {key} is managed by {kit.managed_by}")
            return

    changed = registry.set_toolkits(keys, on)
    for key in dict.fromkeys(keys):
        n = len(registry.toolkit_tools(key))
        tools = f"{n} tool{'' if n == 1 else 's'}"
        if key not in changed:
            _print(f"  {key} is already {word}")
        else:
            _print(f"  {key} {word} — {tools} {'bound' if on else 'unbound'}")
    if not changed:
        return

    _print(f"  {len(registry.tool)} of {len(registry.all_tools)} tools bound; "
           "this applies to your next request.")
    if not registry.toolkit_on("contacts") and registry.toolkit_on("messages") and (
            "contacts" in changed or "messages" in changed):
        _print("  note: with contacts off, messages can text a number you type but cannot "
               "look a name up.")
    if session:
        _print("  session only — omit --session to save to config.yaml.")
    else:
        try:
            if append_block("toolkits", registry.toolkit_block()):
                _print("  added a toolkits: section to config.yaml")
        except OSError as exc:
            _print(f"  could not add the toolkits: section to config.yaml: {exc}")
        for key in changed:
            _persist_key(get_config(), f"toolkits.{key}")
    # The system text and the bound schemas just changed — the whole cached prefix. Re-plant
    # it now so the next request does not pay for re-reading it (a no-op when priming is off).
    prime.start_priming()


@command(
    "tools",
    "Toolkits: the tools Saturn has, in groups you can turn on or off.",
    usage="/tools [<toolkit> | on|off <toolkit>… [--session] | --all]",
    details="""
Saturn's tools come in toolkits — files, web, mail, calendar, messages and so on. A toolkit
that is off is unbound: the model is not shown its tools and cannot call them, the prompt is
that much smaller, and Saturn can never reach that app. When a request needs a toolkit that
is off, Saturn says so and names the switch.

  /tools                      the toolkits: how many tools, on or off, what each is for
  /tools mail                 the mail toolkit's tools, each with its risk tier
  /tools off messages mail    turn toolkits off (saved to config.yaml)
  /tools on messages          turn one back on
  /tools off shell --session  for this session only
  /tools --all                every tool in one flat list, with its toolkit

The core toolkit (the checklist, questions, memory, the calculator and clock) is always on.
An MCP server is listed here and turned on and off with /mcp.

Turning a toolkit on never changes the approval gate: its tools keep their risk tiers
(/policy risk), and a send or a skill save still always asks. A change applies to your next
request.
""",
)
def _tools(ctx, args):
    from tools.toolspec import TOOLKITS

    args, session, _save = split_persist_flags(args)
    args = [a.lower() for a in args]
    if not args:
        _show_toolkits()
    elif args[0] in _TOOLS_ALL:
        _show_all_tools()
    elif args[0] in _TOOLS_ON or args[0] in _TOOLS_OFF:
        _switch_toolkits(args[1:], args[0] in _TOOLS_ON, session=session)
    elif args[0] in TOOLKITS:
        _show_toolkit(args[0])
    else:
        _print(_no_toolkit(args[0]))


# ── /models ──────────────────────────────────────────────────────────────────────────────────
def _finish_switch(cfg, what: str, keys: list[str], *, session: bool,
                   unpersisted: str = "") -> None:
    """The shared tail of every /models switch: rebuild models on next use, say what moved,
    persist `keys` unless session-only (through /config's persist seam), re-embed if the embedder
    moved. `unpersisted` is a note printed after a persist that could not cover every key."""
    from commands.config import _persist_key
    from core.llms import reset_models

    reset_models()
    _print(f"  {what}{' (session only)' if session else ''}.")
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        for key in keys:
            _persist_key(cfg, key)
        if unpersisted:
            _print(unpersisted)
    _resync_rag_after_model_change()


def _bind(cfg, target: str, model: str, *, session: bool = False) -> None:
    """Bind the chat model ("model") or the embedder to a local Ollama model id (a bare scalar in
    config.yaml). The change PERSISTS to config.yaml by default (a model switch should stick);
    session=True applies it live only."""
    if target == "embedder":
        # Machine-wide, like the page's pick: one embedder switch, one set of semantics.
        _switch_embedder(cfg, model, session=session)
        return

    key = f"tiers.{cfg.active_tier}.model"
    cfg.set(key, model)
    _finish_switch(cfg, f"model -> {model} on tier '{cfg.active_tier}'", [key], session=session)


# ── the /models page ─────────────────────────────────────────────────────────────────────────
# One page: the machine, its memory budget, and the two ladders —
# the six chat tiers and the three qwen3-embedding sizes — each priced against the budget at the
# window this config gives it (core/hardware.py), marked pulled / recommended / too big, and
# numbered so one keystroke picks a tier or an embedder. The ladder is one recommended tag per
# size, priced against this machine — a model off the ladder still binds (`/models use <id>`)
# and is priced by the size in its tag. Under the ladders, every other pulled model (`_Other`):
# chat models and embedders apart, numbered on from the ladder rows.
# Bare /models prompts; `list` renders only; the probe itself is cached at startup (hardware
# doesn't change mid-session).


def _probe():
    """The cached machine profile (app.startup warms it). A seam the tests replace."""
    from core.hardware import profile

    return profile()


def _k(n: int) -> str:
    """32768 -> "32k"; anything not a whole multiple of 1024 prints as-is."""
    return f"{n // 1024}k" if n and n % 1024 == 0 else str(n)


def _active_embedder(cfg) -> "str | None":
    """The active tier's embedder id, or None when config.yaml declares none for it (or names an
    active tier it never declared) — Config.embedder_model raises for both, and the page must
    still render: it is where the user fixes exactly that."""
    try:
        return cfg.embedder_model
    except KeyError:
        return None


def _tier_binding(cfg, key: str) -> "tuple[str, str]":
    """(declared, running) for a size-class tier: what config.yaml literally binds as its chat
    model ("" when the tier is not declared) and what selecting it would
    RUN — the declaration itself, or the ladder tag for a class this config never declared."""
    declared = _tier_model(cfg, key)
    if not declared:
        return "", model_family.tag_for(key)
    return declared, declared


def _class_windows(cfg) -> dict:
    """The context window each size class would actually run at under THIS config: the tier's
    bound model (its `capabilities.<model>.context_window`, or the `runtime.num_ctx` override
    when set). This is the number ChatOllama is handed (config.num_ctx_for), so the fit table
    prices the window the session would really allocate — raise num_ctx and the page re-prices."""
    return {key: cfg.num_ctx_for(_tier_binding(cfg, key)[1]) for key in model_family.classes()}


def _cost_classes(cfg) -> dict:
    """The class each tier is PRICED as: that of the model it actually runs (a tier rebound to
    another size costs what it runs, not what its name says)."""
    return {key: model_family.class_of(_tier_binding(cfg, key)[1]) for key in model_family.classes()}


def _switch_tier(cfg, key: str, *, session: bool) -> None:
    """The tier switch a page pick performs: set live, rebuild models on next use, persist
    unless session-only, re-embed if the embedder moved."""
    cfg.set("active_tier", key)
    _finish_switch(cfg, f"active tier -> {key}; models will rebuild on next use", ["active_tier"],
                   session=session)


def _switch_embedder(cfg, model: str, *, session: bool) -> None:
    """Bind the embedder on EVERY declared tier — the one embedder switch, behind the page's pick
    and `/models embedder <id>` alike: the embedder is a machine choice (what fits beside the chat
    model), not a per-tier one, so a later tier switch must not silently bring a different
    embedder — and a different corpus embedding — back. Tiers are written by dict access:
    Config.set splits a dotted path, so a tier key with a dot in it (a pre-rename `0.8b`, a
    user's own name) would land in a phantom nested tier. persist() walks the same dotted path,
    so such a tier is set live and named as not persisted."""
    tiers = cfg.get("tiers", {}) or {}
    keys, unpersistable = [], []
    for key, tier in tiers.items():
        if not isinstance(tier, dict):
            continue
        tier["embedder"] = model
        (unpersistable if "." in key else keys).append(key)
    note = (f"  set for this session, but not persisted for tier(s) {', '.join(unpersistable)}: "
            "the name contains a dot — edit config.yaml by hand") if unpersistable else ""
    _finish_switch(cfg, f"embedder -> {model} on every tier",
                   [f"tiers.{key}.embedder" for key in keys], session=session, unpersisted=note)


@dataclass(frozen=True)
class _Other:
    """A pulled model that sits on neither ladder, as the page lists and picks it."""

    kind: str           # "chat" | "embedder"
    name: str
    weights_gb: float   # size on disk; 0 when the daemon did not say
    params: str
    tools: bool         # False only for a chat model the daemon SAID cannot call tools

    @property
    def need_gb(self) -> float:
        """Size on disk plus the flat headroom of its kind. Unlike a ladder row, the cache its
        window costs is not priced: the architecture behind an arbitrary tag is not known."""
        from core.hardware import EMBEDDER_HEADROOM_GB, HEADROOM_GB

        return self.weights_gb + (EMBEDDER_HEADROOM_GB if self.kind == "embedder" else HEADROOM_GB)


def _other_models(cfg, local: list) -> "list[_Other]":
    """Every pulled model no ladder row already shows, chat models first, then embedders, each
    group in the daemon's name order. The daemon's capability list tells the two apart (and
    marks a chat model that cannot call tools); with no answer for a model, "embed" in its name
    is the fallback and tool calling is not doubted."""
    from core.llms import _model_present, model_capabilities

    shown = {_tier_binding(cfg, key)[1] for key in model_family.classes()}
    shown |= {model_family.embedder_tag_for(key) for key in model_family.embedder_classes()}
    rest = [m for m in local if not _model_present(m.name, shown)]
    if not rest:
        return []
    caps = model_capabilities([m.name for m in rest])
    out = []
    for m in rest:
        known = caps.get(m.name)
        embedder = "embedding" in known if known else "embed" in m.name.lower()
        out.append(_Other(
            kind="embedder" if embedder else "chat",
            name=m.name,
            weights_gb=(getattr(m, "size_bytes", 0) or 0) / 1e9,
            params=getattr(m, "params", "") or "",
            tools=embedder or not known or "tools" in known,
        ))
    return [o for o in out if o.kind == "chat"] + [o for o in out if o.kind == "embedder"]


def _other_fits(other: _Other, rec) -> bool:
    """A chat model against the whole budget; an embedder against what the recommended tier
    leaves (the same question the embedder ladder asks)."""
    left = rec.budget_gb - (rec.needs[rec.size_class] if other.kind == "embedder" else 0)
    return other.need_gb <= left


def _pulled_cell(models: list, up: bool, have: set) -> tuple:
    from core.llms import _model_present

    if not up:
        return ("?", "dim")
    if all(_model_present(m, have) for m in models):
        return ("✓", "green")
    return ("·", "dim")


def _render_page(cfg, prof, rec, *, up: bool, have: set, others: "list[_Other]" = ()) -> None:
    """The readout, in the app's one listing vocabulary (section / table / note — the shapes
    /policy renders with; they own the no-rich fallback)."""
    from core.hardware import (
        BASELINE_FAMILY,
        CLASS_COSTS,
        DECODE_FLOOR_TOK_S,
        DECODE_SLOW_TOK_S,
        EMBEDDER_WEIGHTS_GB,
        HEADROOM_GB,
    )
    from core.llms import _model_present
    from tui import ui

    active = cfg.active_tier
    defined = cfg.get("tiers", {}) or {}
    emb_model = _active_embedder(cfg)
    active_emb = model_family.embedder_class_of(emb_model)

    facts = [prof.chip or "unknown chip"]
    if prof.cores:
        facts.append(f"{prof.cores} cores")
    if prof.gpu_cores:
        facts.append(f"{prof.gpu_cores} GPU cores")
    facts.append(f"{prof.ram_gb:g} GB unified memory")
    facts.append(f"{prof.bandwidth_gbps:g} GB/s")
    ui.section("models", " · ".join(facts))
    ui.table([("budget", f"{rec.budget_gb:g} GB for models", (rec.reason, "dim"))], styles=["dim"])
    _print("")

    dim = lambda *cells: tuple((c, "dim") for c in cells)  # noqa: E731
    rows = [dim("#", "tier", "model", "weights", "window", "need", "speed", "", "")]
    n = 0
    for key in model_family.classes():
        n += 1
        declared, running = _tier_binding(cfg, key)
        speed = rec.decode.get(key, 0.0)
        if key not in defined:
            status = ("not in config.yaml", "yellow")
        elif key == rec.size_class:
            status = ("▸ recommended" + (" (slow)" if rec.slow else ""), "accent")
        elif not rec.fits.get(key):
            status = ("too big", "yellow")
        elif rec.usable.get(key):
            status = ("fits", "dim")
        elif speed >= DECODE_SLOW_TOK_S:
            status = ("fits · slow", "yellow")
        else:
            status = ("fits · too slow", "yellow")
        rows.append((
            (str(n), "dim"),
            ("* " if key == active else "  ") + key,
            running,
            (f"{CLASS_COSTS[rec.cost_classes.get(key, key)].weights_gb:>5.1f} GB", "dim"),
            (f"{_k(rec.windows[key]):>4} ctx", "dim"),
            f"{rec.needs[key]:>5.1f} GB",
            (f"~{speed:.0f} tok/s".rjust(10), "dim"),
            _pulled_cell([running], up, have),
            status,
        ))
    rows.append(("",))
    rows.append(dim("", "embedder", "model", "weights", "", "need", "", "", ""))
    for key in model_family.embedder_classes():
        n += 1
        tag = model_family.embedder_tag_for(key)
        if key == rec.embedder:
            status = ("▸ recommended", "accent")
        elif rec.embedder_fits.get(key):
            status = (f"fits beside {rec.size_class}", "dim")
        else:
            status = (f"swaps beside {rec.size_class}", "yellow")
        rows.append((
            (str(n), "dim"),
            ("* " if key == active_emb else "  ") + key,
            tag,
            (f"{EMBEDDER_WEIGHTS_GB[key]:>5.1f} GB", "dim"),
            "",
            f"{rec.embedder_needs[key]:>5.1f} GB",
            "",
            _pulled_cell([tag], up, have),
            status,
        ))
    running = {"chat": _tier_model(cfg, active), "embedder": emb_model or ""}
    for kind, title in (("chat", "other chat models"), ("embedder", "other embedders")):
        group = [o for o in others if o.kind == kind]
        if not group:
            continue
        rows.append(("",))
        rows.append(dim("", "", title, "weights", "params", "need", "", "", ""))
        for o in group:
            n += 1
            if not o.tools:
                status = ("no tool calling", "yellow")
            elif not o.weights_gb:
                status = ("size unknown", "dim")
            elif kind == "embedder":
                status = ((f"fits beside {rec.size_class}", "dim") if _other_fits(o, rec)
                          else (f"swaps beside {rec.size_class}", "yellow"))
            else:
                status = ("fits", "dim") if _other_fits(o, rec) else ("too big", "yellow")
            rows.append((
                (str(n), "dim"),
                "* " if running[kind] and _model_present(o.name, {running[kind]}) else "  ",
                o.name,
                (f"{o.weights_gb:>5.1f} GB" if o.weights_gb else "", "dim"),
                (o.params, "dim"),
                f"{o.need_gb:>5.1f} GB" if o.weights_gb else "",
                "",
                _pulled_cell([o.name], up, have),
                status,
            ))
    ui.table(rows)

    override = cfg.num_ctx_override
    src = (f"runtime.num_ctx = {override} overrides every window" if override
           else "windows from config.yaml context_window (/config runtime.num_ctx to change)")
    ui.note(f"* active · ✓ pulled · need = weights + KV cache at that window + {HEADROOM_GB:g} GB headroom"
            f" · speed = est. decode at {prof.bandwidth_gbps:g} GB/s")
    ui.note(src)
    if others:
        ui.note("other models: need = size on disk + headroom (the cache a window costs is not "
                f"priced) · a chat pick binds the model on tier '{active}'")
    if not rec.cramped:
        decode = rec.decode[rec.size_class]
        ui.note(f"{rec.size_class} feels like: ~{decode:.0f} tok/s (a paragraph in ~{150 / decode:.0f} s)"
                f" · first prompt ~{rec.first_prompt_s():.0f} s cold, then cached"
                + (f" · under the {DECODE_FLOOR_TOK_S:g} tok/s floor: every tier that fits is slow here"
                   if rec.slow else ""))
    if prof.baseline:
        ui.warn(f"chip not recognised ({prof.chip or 'no brand string'}) — speed is priced at the "
                f"{BASELINE_FAMILY} baseline; the fit column is still this machine's memory")
    if not up:
        ui.warn("ollama daemon not reachable — start it with `ollama serve` (pulled state unknown)")
    if defined and emb_model is None:
        ui.warn(f"tier '{active}' has no embedder in config.yaml — pick an embedder row to set "
                "one on every tier")
    elif defined and active_emb is None:
        ui.note(f"embedder in config.yaml: {emb_model} (not on the ladder; pick a row to move "
                "onto it)")
    if rec.cramped:
        ui.warn(f"this machine is too small for any tier ({rec.budget_gb:g} GB budget; the smallest "
                f"wants {rec.needs[rec.size_class]:.1f} GB) — {rec.size_class} is the best effort "
                "and will be tight")


def _pick(rec, active: str, others: "list[_Other]" = ()) -> "list[tuple] | None":
    """The human's say after the readout: Enter takes the recommended TIER (the embedder, whose
    switch re-embeds the whole knowledge base, is confirmed separately — see _confirm_embedder),
    a row number picks that one tier, embedder, or other pulled model, n/q/cancel keeps things
    as they are. Anything
    unparseable is treated as cancel — an auto-select must never land on a row nobody chose —
    and so is Ctrl-C / Ctrl-D: ui.ask would otherwise hand back the empty reply Enter produces,
    and an interrupt is the one keypress that must never select. Returns [(kind, class), ...]
    to apply in order — ("other", the _Other row) for a model off the ladders — or None."""
    from tui import ui

    tiers = model_family.classes()
    embs = model_family.embedder_classes()
    total = len(tiers) + len(embs) + len(others)
    reply = ui.ask(
        f"[Enter] {rec.size_class} · 1-{total} pick a row · n keep {active} » ",
        on_interrupt="n",
    ).strip().lower()
    if not reply:
        return [("tier", rec.size_class)]
    if reply in ("n", "no", "q", "quit", "cancel"):
        return None
    try:
        idx = int(reply)
    except ValueError:
        ui.warn(f"not a valid selection: {reply!r}")
        return None
    if not 1 <= idx <= total:
        ui.warn(f"not a valid selection: {reply!r} (1-{total})")
        return None
    if idx <= len(tiers):
        return [("tier", tiers[idx - 1])]
    if idx <= len(tiers) + len(embs):
        return [("embedder", embs[idx - len(tiers) - 1])]
    return [("other", others[idx - len(tiers) - len(embs) - 1])]


def _confirm_embedder(rec, active_emb: "str | None") -> bool:
    """After an Enter, offer the recommended embedder ONLY when it differs from the active one,
    and only on a yes: moving the embedder re-embeds every document in the knowledge base — too
    heavy a side effect to ride on a default keypress."""
    from tui import ui

    if rec.embedder == active_emb:
        return False
    tag = model_family.embedder_tag_for(rec.embedder)
    reply = ui.ask(
        f"embedder: {tag} is the largest that fits beside {rec.size_class} — switch to it too? "
        "(re-embeds the knowledge base)  [y/N] » "
    ).strip().lower()
    return reply in ("y", "yes")


def _offer_pull(missing: list[str], what: str) -> bool:
    """The consented `ollama pull` for a pick whose models aren't here (the caller has already
    established a TTY). True when every pull landed."""
    from tui import ui

    reply = ui.ask(
        f"pull {len(missing)} model(s) for {what}? (sizes shown as each pull starts)  [y/N] » "
    ).lower()
    if reply not in ("y", "yes"):
        ui.note("ok — to do it later:")
        for m in missing:
            _print(f"    ollama pull {m}")
        return False
    return run_pulls(missing, pull=_pull_one)


def _apply_pick(cfg, kind: str, key: str, rec, *, have: set, session: bool) -> None:
    """Land one pick (the caller has established a reachable daemon and a TTY). Only a row whose
    models are here is switched to: otherwise the pull is offered first and the switch follows a
    successful pull; any other outcome keeps the current binding working. A pick that is ALREADY
    active still gets its missing models pulled (a fresh install with a custom pull list is
    exactly this case)."""
    from core.llms import _model_present
    from tui import ui

    if kind == "other":
        _apply_other(cfg, key, rec, session=session)
        return
    tiers = cfg.get("tiers", {}) or {}
    if kind == "tier":
        if key not in tiers:
            ui.warn(f"tier '{key}' is not defined in this config.yaml — add it from "
                    "config.default.yaml, then re-run /models")
            return
        if not rec.fits.get(key, True):
            ui.warn(f"by the numbers tier '{key}' wants {rec.needs[key]:.1f} GB against a "
                    f"{rec.budget_gb:g} GB budget — it may fail to load, or run slowly, on this machine")
        models = [_tier_binding(cfg, key)[1]]
        current = key == cfg.active_tier
        what = f"tier {key}"
    else:
        tag = model_family.embedder_tag_for(key)
        if not rec.embedder_fits.get(key, True):
            ui.warn(f"embedder {tag} will swap in and out beside tier {rec.size_class} — "
                    "knowledge-base lookups pay a reload")
        models = [tag]
        # The switch is machine-wide, so "already on it" means EVERY tier binds it: an active
        # tier that does while another does not is precisely the drift the switch removes.
        current = bool(tiers) and all(
            isinstance(t, dict) and str(t.get("embedder") or "") == tag for t in tiers.values()
        )
        what = f"embedder {tag}"

    missing = [m for m in models if not _model_present(m, have)]
    if missing and not _offer_pull(missing, what):
        return
    if current:
        ui.note(f"already on {what}" + (" — models pulled" if missing else " — nothing to change"))
        return
    if kind == "tier":
        _switch_tier(cfg, key, session=session)
    else:
        _switch_embedder(cfg, models[0], session=session)


def _apply_other(cfg, other: _Other, rec, *, session: bool) -> None:
    """Land a pick from the other-models rows: a chat model is bound on the ACTIVE tier (what
    `/models use <id>` does), an embedder on every tier (`/models embedder <id>`). The model is
    pulled by definition — the rows are the daemon's own list. A model the daemon says cannot
    call tools is never bound from the page: the loop is one tool-calling call per pass."""
    from core.llms import _model_present
    from tui import ui

    if not other.tools:
        ui.warn(f"{other.name} has no tool calling — the agent loop needs it, so it was not bound")
        return
    if other.kind == "chat":
        if other.weights_gb and not _other_fits(other, rec):
            ui.warn(f"by the numbers {other.name} wants {other.need_gb:.1f} GB against a "
                    f"{rec.budget_gb:g} GB budget — it may fail to load, or run slowly, on this machine")
        if _model_present(other.name, {_tier_model(cfg, cfg.active_tier)}):
            ui.note(f"already on {other.name} — nothing to change")
            return
        _bind(cfg, "model", other.name, session=session)
        return
    if other.weights_gb and not _other_fits(other, rec):
        ui.warn(f"embedder {other.name} will swap in and out beside tier {rec.size_class} — "
                "knowledge-base lookups pay a reload")
    tiers = cfg.get("tiers", {}) or {}
    if tiers and all(isinstance(t, dict) and str(t.get("embedder") or "") == other.name
                     for t in tiers.values()):
        ui.note(f"already on embedder {other.name} — nothing to change")
        return
    _switch_embedder(cfg, other.name, session=session)


def _models_page(cfg, *, prompt: bool, session: bool = False) -> None:
    """Render the page; with `prompt`, ask and apply."""
    from core.hardware import recommend
    from core.llms import list_local_models, ollama_reachable
    from tui import ui

    prof = _probe()
    rec = recommend(prof, _class_windows(cfg), _cost_classes(cfg))
    # One daemon round trip: the model list answers reachability too; the probe only has to
    # tell "daemon down" from "nothing pulled" when the list came back empty (check_models'
    # shape).
    local = list_local_models()
    up = bool(local) or ollama_reachable()
    have = {m.name for m in local}
    others = _other_models(cfg, local)
    _render_page(cfg, prof, rec, up=up, have=have, others=others)
    if not prompt or not up:
        # With no daemon there is nothing a pick could be applied to (the models can't even be
        # listed).
        return
    from commands._utils import _stdin_is_tty

    if not _stdin_is_tty():
        return
    picks = _pick(rec, cfg.active_tier, others)
    if picks is None:
        ui.note(f"staying on '{cfg.active_tier}'")
        # Keeping a tier whose model isn't here still offers the pull (the embedder waits for
        # the first /docs add).
        from core.llms import _model_present

        running = _tier_binding(cfg, cfg.active_tier)[1]
        if not _model_present(running, have):
            _offer_pull([running], f"tier {cfg.active_tier}")
        return
    active_emb = model_family.embedder_class_of(_active_embedder(cfg))
    if picks == [("tier", rec.size_class)] and _confirm_embedder(rec, active_emb):
        # Embedder FIRST: every switch re-syncs the corpus against the active tier's embedder,
        # and with the embedder already set on every tier the tier switch's re-sync sees no
        # change — one re-embed, not one per pick (a config with per-tier embedders would
        # otherwise embed the whole corpus twice, the first time with a model then discarded).
        picks.insert(0, ("embedder", rec.embedder))
    for kind, key in picks:
        _apply_pick(cfg, kind, key, rec, have=have, session=session)


def _tier_model(cfg, key: str) -> str:
    """What a tier actually binds, read straight off the tiers mapping (dict access, never the
    dotted cfg.get path — a tier name may contain a dot)."""
    return tier_chat_model((cfg.get("tiers", {}) or {}).get(key) or {})


# Per-model target spellings a tier no longer takes: it runs one model.
_RETIRED_TARGETS = ("all", "tool_caller", "utility")


@command(
    "models",
    "The model page: your hardware, the qwen ladder priced against it, every other pulled model, pick one.",
    aliases=("model",),
    usage="/models [list] [--session] | /models use|embedder <id> [--session]",
    details="""
Shows the machine (Apple chip, cores, GPU cores, unified memory, memory bandwidth), the memory
budget the model runner can address, and the two ladders priced against it: the four chat tiers
(one recommended tag per size, the most advanced of the qwen3.5-3.8 line) and the three
qwen3-embedding sizes. Each row carries its weights, the context window this config gives it, the
memory it needs at that window, its estimated decode speed, whether it is pulled (✓), and whether
it fits — with the recommendation marked ▸: the largest tier that fits AND runs at 10 tok/s or
better. A tier that fits but would crawl reads `fits · slow` and is never the default.

  budget     ~75% of unified memory (what macOS lets the GPU address)
  need       weights + KV cache at the window + 1.5 GB headroom (only 1 in 4 layers of these
             hybrid models keeps a cache, which is why the numbers are small)
  speed      memory bandwidth over the weights streamed per token — the M1–M5 families'
             published numbers, the binned Max chips told apart by GPU core count; a chip the
             table does not know is priced at the M1 baseline and the page says so
  embedder   the largest that fits BESIDE the recommended tier, so lookups never evict it

Below the ladders come the models you have pulled that are on neither: other chat models, then
other embedders (the daemon says which is which). Each carries its size on disk, its parameter
count, and whether it fits — size plus headroom against the same budget; the cache its window
would cost is not priced. A chat model the daemon says cannot call tools reads `no tool
calling` and cannot be picked here.

The rows are numbered. Enter takes the recommended tier; a number picks that one row (a "too
big" pick is honored with a warning); n keeps things as they are. Picking another chat model
binds it on the active tier (what `/models use <id>` does); picking another embedder sets it on
every tier. When the recommended embedder
differs from the active one, Enter then asks — y/N, default no — whether to switch it too,
because an embedder switch re-embeds the whole corpus. A pick whose model isn't pulled asks
first — y/N, default no — and only switches after the pull succeeds. An embedder pick is set
on every tier (it is a machine choice).

  /models                    the page, then the prompt
  /models list               the page only (`ls` / --check work too)
  /models use <id>           run any Ollama model with native tool-calling on this tier
  /models embedder <id>      switch the embedding model by name (re-embeds the corpus)

Every switch PERSISTS to config.yaml by default; --session applies it live only. Runs on the
very first launch. Keeping a tier whose model isn't pulled offers the pull too. Models are
Ollama ids.
""",
)
def _models(ctx, args):
    from config import get_config

    cfg = get_config()
    args, session, save = split_persist_flags(args)

    if not args:
        _models_page(cfg, prompt=True, session=session)
        return

    sub = args[0].lower()

    if is_list_verb(sub) or sub in ("--check", "check"):
        _models_page(cfg, prompt=False)
        return

    if sub == "tier":
        _print(f"  /models {sub} is gone — /models shows the page; pick a tier by its number.")
        return

    if sub == "embedder":
        if len(args) < 2:
            _print("  usage: /models embedder <model_id> [--session]")
            return
        _bind(cfg, "embedder", args[1], session=session)
        return

    if sub in _RETIRED_TARGETS:
        _print(f"  /models {sub} is now /models use <model_id> — a tier runs one model.")
        return

    if sub != "use":
        _print(f"  unknown target: {sub} (list/use/embedder)")
        return
    if len(args) < 2:
        _print("  usage: /models use <model_id> [--session]")
        return
    if len(args) > 2:
        _print("  too many arguments — usage: /models use <model_id> [--session]")
        return
    _bind(cfg, "model", args[1], session=session)


# ── /mcp ─────────────────────────────────────────────────────────────────────────────────────
@command(
    "mcp",
    "MCP servers: connection status, the remote tools they add, reconnect.",
    usage="/mcp [list | reload]",
    details="""
Saturn is an MCP client: servers declared under `mcp.servers:` in config.yaml are connected at
startup and every remote tool they expose registers behind the SAME risk-tier approval gate as
the local tools (named `mcp_<server>_<tool>`; they show in /tools and the agent sees them).

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

Adding a server (config.yaml; secrets via ${VAR}, plain env vars from .env or your shell):

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
