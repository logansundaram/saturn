"""
Runtime-inventory commands — what the agent is running on and with, in one module (the /help
"observability" readouts; consolidated from one-file-per-command 2026-06-11):

  /tools    the registered tools + risk tiers
  /models   the model page: hardware scan, the qwen ladders priced against it, pick a tier /
            embedder (absorbed /scan 2026-09-01; `scan` stays as an alias)
  /mcp      MCP server status + remote tools; reload

(/context folded into /config as `/config context` 2026-07-07 — the runtime readout + num_ctx
setter belong under the one runtime-settings front door.)
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import (
    _ROLES, _resync_rag_after_model_change, is_list_verb, pull_one as _pull_one, run_pulls,
    split_persist_flags,
)
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
        # Machine-wide, like the page's pick: one embedder switch, one set of semantics.
        _switch_embedder(cfg, model, session=session)
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


# ── the /models page ─────────────────────────────────────────────────────────────────────────
# One page (2026-09-01, the /scan fold): the machine, its memory budget, and the two ladders —
# the six chat tiers and the three qwen3-embedding sizes — each priced against the budget at the
# window this config gives it (core/hardware.py), marked pulled / recommended / too big, and
# numbered so one keystroke picks a tier or an embedder. The old verbatim `ollama list` view is
# gone: Saturn binds ONE family with the most advanced tag per size, so the ladder IS the list.
# Bare /models prompts; `list` renders only; the probe itself is cached at startup (hardware
# doesn't change mid-session) and `rescan` re-reads it.


def _probe():
    """The cached machine profile (app.startup warms it). A seam the tests replace."""
    from core.hardware import profile

    return profile()


def _rescan():
    from core.hardware import profile

    return profile(rescan=True)


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
    """(declared, running) for a size-class tier: what config.yaml literally binds to its chat
    roles (the synthesizer's entry; "" when the tier is not declared) and what selecting it would
    actually RUN — a non-family declaration is substituted at the model_for_role seam, so the row
    must show the substitute (the migration note under the table names the substitution). A class
    this config never declared runs the ladder tag."""
    declared = _tier_model(cfg, key)
    if not declared:
        return "", model_family.tag_for(key)
    if model_family.in_family(declared):
        return declared, declared
    return declared, model_family.tag_for(model_family.migrate(declared))


def _tier_running_models(cfg, key: str) -> list[str]:
    """The chat models selecting a tier would RUN (roles only, deduplicated; the embedder is a
    separate, machine-wide pick): each role's declared id, family-substituted the way
    Config.model_for_role substitutes it — a non-family declaration never loads, so checking or
    pulling IT would fetch a model the agent refuses to run while the tag that does run stays
    missing. A tier declared without roles runs the ladder tag. Dict access, never the dotted
    path — a tier key may contain a dot."""
    tier = (cfg.get("tiers", {}) or {}).get(key) or {}
    out: list[str] = []
    for entry in (tier.get("roles", {}) or {}).values():
        if isinstance(entry, dict):
            entry = entry.get("model", "")
        model = str(entry or "")
        if not model:
            continue
        if not model_family.in_family(model):
            model = model_family.tag_for(model_family.migrate(model))
        if model not in out:
            out.append(model)
    return out or [model_family.tag_for(key)]


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


def _legacy_tiers(cfg) -> list[str]:
    """Tier names in config.yaml that are not size classes (laptop / workstation, from before
    the ladder). They are what `/models tier` validates against on that config, so the page
    names them with the bind that works there instead of pretending the ladder is selectable."""
    return [k for k in (cfg.get("tiers", {}) or {}) if k not in model_family.classes()]


def _switch_tier(cfg, key: str, *, session: bool) -> None:
    """The same switch `/models tier` performs: set live, rebuild models on next use, persist
    unless session-only, re-embed if the embedder moved."""
    from core.llms import reset_models

    cfg.set("active_tier", key)
    reset_models()
    tag = " (session only)" if session else ""
    _print(f"  active tier -> {key}; models will rebuild on next use{tag}.")
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        _persist_bindings(cfg, ["active_tier"])
    _resync_rag_after_model_change()


def _switch_embedder(cfg, model: str, *, session: bool) -> None:
    """Bind the embedder on EVERY declared tier — the one embedder switch, behind the page's pick
    and `/models embedder <id>` alike: the embedder is a machine choice (what fits beside the chat
    model), not a per-tier one, so a later tier switch must not silently bring a different
    embedder — and a different corpus embedding — back. Tiers are written by dict access:
    Config.set splits a dotted path, so a tier key with a dot in it (a pre-rename `0.8b`, a
    user's own name) would land in a phantom nested tier. persist() walks the same dotted path,
    so such a tier is set live and named as not persisted."""
    from core.llms import reset_models

    tiers = cfg.get("tiers", {}) or {}
    keys, unpersistable = [], []
    for key, tier in tiers.items():
        if not isinstance(tier, dict):
            continue
        tier["embedder"] = model
        (unpersistable if "." in key else keys).append(key)
    reset_models()
    tag = " (session only)" if session else ""
    _print(f"  embedder -> {model} on every tier{tag}.")
    if session:
        _print("  omit --session to save to config.yaml.")
    else:
        _persist_bindings(cfg, [f"tiers.{key}.embedder" for key in keys])
        if unpersistable:
            _print(f"  set for this session, but not persisted for tier(s) {', '.join(unpersistable)}: "
                   "the name contains a dot — edit config.yaml by hand")
    _resync_rag_after_model_change()


def _pulled_cell(models: list, up: bool, have: set) -> tuple:
    from core.llms import _model_present

    if not up:
        return ("?", "dim")
    if all(_model_present(m, have) for m in models):
        return ("✓", "green")
    return ("·", "dim")


def _render_page(cfg, prof, rec, *, up: bool, have: set) -> None:
    """The readout, in the app's one listing vocabulary (section / table / note — the shapes
    /privacy and /policy render with; they own the no-rich fallback)."""
    from core.hardware import CLASS_COSTS, EMBEDDER_WEIGHTS_GB, HEADROOM_GB
    from tui import ui

    active = cfg.active_tier
    defined = cfg.get("tiers", {}) or {}
    emb_model = _active_embedder(cfg)
    active_emb = model_family.embedder_class_of(emb_model)

    facts = [prof.chip or "unknown CPU"]
    if prof.cores:
        facts.append(f"{prof.cores} cores")
    facts.append(f"{prof.ram_gb:g} GB " + ("unified memory" if prof.backend == "apple" else "RAM"))
    if prof.vram_gb:
        facts.append(f"{prof.vram_gb:g} GB VRAM ({prof.gpu})")
    ui.section("models", " · ".join(facts))
    ui.table([("budget", f"{rec.budget_gb:g} GB for models", (rec.reason, "dim"))], styles=["dim"])
    _print("")

    dim = lambda *cells: tuple((c, "dim") for c in cells)  # noqa: E731
    rows = [dim("#", "tier", "model", "weights", "window", "need", "", "", "")]
    n = 0
    for key in model_family.classes():
        n += 1
        declared, running = _tier_binding(cfg, key)
        if key not in defined:
            status = ("not in config.yaml", "yellow")
        elif key == rec.size_class:
            status = ("▸ recommended", "accent")
        elif rec.fits.get(key):
            status = ("fits", "dim")
        else:
            status = ("too big", "yellow")
        rows.append((
            (str(n), "dim"),
            ("* " if key == active else "  ") + key,
            running,
            (f"{CLASS_COSTS[rec.cost_classes.get(key, key)].weights_gb:>5.1f} GB", "dim"),
            (f"{_k(rec.windows[key]):>4} ctx", "dim"),
            f"{rec.needs[key]:>5.1f} GB",
            _pulled_cell([running], up, have),
            status,
        ))
    rows.append(("",))
    rows.append(dim("", "embedder", "model", "weights", "", "need", "", ""))
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
            _pulled_cell([tag], up, have),
            status,
        ))
    ui.table(rows)

    override = cfg.num_ctx_override
    src = (f"runtime.num_ctx = {override} overrides every window" if override
           else "windows from config.yaml context_window (/config context to change)")
    ui.note(f"* active · ✓ pulled · need = weights + KV cache at that window + {HEADROOM_GB:g} GB headroom")
    ui.note(src)
    if not up:
        ui.warn("ollama daemon not reachable — start it with `ollama serve` (pulled state unknown)")
    if prof.gpu_error:
        ui.warn(f"GPU probe failed ({prof.gpu_error}) — budgeted as if there were no accelerator; "
                "`/models rescan` once the driver is up")
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
    legacy = _legacy_tiers(cfg)
    if legacy:
        ui.warn(f"legacy tiers in config.yaml: {', '.join(legacy)} — the names predate the size-class "
                "ladder; rebind one in place with `/models all <tag>`")
    _print_migration_notes(cfg)


def _pick(rec, active: str) -> "list[tuple[str, str]] | None":
    """The human's say after the readout: Enter takes the recommended TIER (the embedder, whose
    switch re-embeds the whole knowledge base, is confirmed separately — see _confirm_embedder),
    a row number picks that one tier or embedder, n/q/cancel keeps things as they are. Anything
    unparseable is treated as cancel — an auto-select must never land on a row nobody chose —
    and so is Ctrl-C / Ctrl-D: ui.ask would otherwise hand back the empty reply Enter produces,
    and an interrupt is the one keypress that must never select. Returns [(kind, class), ...]
    to apply in order, or None."""
    from tui import ui

    tiers = model_family.classes()
    embs = model_family.embedder_classes()
    total = len(tiers) + len(embs)
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
    return [("embedder", embs[idx - len(tiers) - 1])]


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

    tiers = cfg.get("tiers", {}) or {}
    if kind == "tier":
        if key not in tiers:
            ui.warn(f"tier '{key}' is not defined in this config.yaml — add it from "
                    "config.default.yaml, then re-run /models")
            return
        if not rec.fits.get(key, True):
            ui.warn(f"by the numbers tier '{key}' wants {rec.needs[key]:.1f} GB against a "
                    f"{rec.budget_gb:g} GB budget — it may fail to load, or run slowly, on this machine")
        models = _tier_running_models(cfg, key)
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


def _models_page(cfg, *, prompt: bool, session: bool = False, rescan: bool = False) -> None:
    """Render the page; with `prompt`, ask and apply."""
    from core.hardware import recommend
    from core.llms import list_local_models, ollama_reachable
    from tui import ui

    prof = _rescan() if rescan else _probe()
    rec = recommend(prof, _class_windows(cfg), _cost_classes(cfg))
    # One daemon round trip: the model list answers reachability too; the probe only has to
    # tell "daemon down" from "nothing pulled" when the list came back empty (check_models'
    # shape).
    local = list_local_models()
    up = bool(local) or ollama_reachable()
    have = {m.name for m in local}
    _render_page(cfg, prof, rec, up=up, have=have)
    if not prompt or not up:
        # With no daemon there is nothing a pick could be applied to (the models can't even be
        # listed).
        return
    from commands.config import _stdin_is_tty

    if not _stdin_is_tty():
        return
    picks = _pick(rec, cfg.active_tier)
    if picks is None:
        ui.note(f"staying on '{cfg.active_tier}'")
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


def _print_migration_notes(cfg=None) -> None:
    """Name the family substitutions the page shows, so a listing never claims the file's value
    is what is running: this session's ledger (config.migrated_bindings — the ACTIVE tier's
    roles, the only ones ever resolved) plus, given the config, every ladder tier whose declared
    model the row substitutes (a non-active tier's binding is never resolved, so the ledger alone
    would leave that row contradicting config.yaml with nothing said)."""
    notes = dict(_config_migrations())
    if cfg is not None:
        for key in model_family.classes():
            declared, running = _tier_binding(cfg, key)
            if declared and declared != running:
                notes.setdefault(declared, running)
    for original, replacement in notes.items():
        _print(f"  note: '{original}' in config.yaml is running as '{replacement}'.")


def _tier_model(cfg, key: str) -> str:
    """What a tier actually binds, read straight off the tiers mapping (dict access, never the
    dotted cfg.get path — a legacy tier name may contain a dot)."""
    tier = (cfg.get("tiers", {}) or {}).get(key) or {}
    roles = tier.get("roles", {}) or {}
    entry = roles.get("synthesizer") or next(iter(roles.values()), None)
    if isinstance(entry, dict):
        entry = entry.get("model", "")
    return str(entry or "")


def _config_migrations() -> dict:
    """This session's family substitutions, so the listing never claims the file's value is
    what is running."""
    import config as _config

    return _config.migrated_bindings()


@command(
    "models",
    "The model page: your hardware, the qwen ladder priced against it, pick a tier / embedder.",
    aliases=("model", "scan"),
    usage="/models [list|rescan] [--session] | /models tier <name> | /models <role|all|embedder> <id> [--session]",
    details="""
Shows the machine (chip, cores, memory, VRAM), the memory budget the model runner can address,
and the two ladders priced against it: the six chat tiers (one tag per size, the most advanced
of the qwen3.5-3.8 family) and the three qwen3-embedding sizes. Each row carries its weights,
the context window this config gives it, the memory it needs at that window, whether it is
pulled (✓), and whether it fits — with the recommendation marked ▸.

  budget     Apple silicon: ~75% of unified memory · NVIDIA: the card's VRAM ·
             CPU only: 50% of RAM, capped at the 9b class
  need       weights + KV cache at the window + 1.5 GB headroom (only 1 in 4 layers of these
             hybrid models keeps a cache, which is why the numbers are small)
  embedder   the largest that fits BESIDE the recommended tier, so lookups never evict it

The rows are numbered. Enter takes the recommended tier; a number picks that one row (a "too
big" pick is honored with a warning); n keeps things as they are. When the recommended embedder
differs from the active one, Enter then asks — y/N, default no — whether to switch it too,
because an embedder switch re-embeds the whole corpus. A pick whose model isn't pulled asks
first — y/N, default no — and only switches after the pull succeeds. An embedder pick is set
on every tier (it is a machine choice).

  /models                    the page, then the prompt
  /models list               the page only (`ls` / --check work too)
  /models rescan             probe the hardware again (it is read once at startup and cached)
  /models tier <name>        switch the tier directly
  /models all <id>           point every role at one family tag (a hidden/superseded tag works)
  /models <role> <id>        re-point one role — planner, tool_caller, synthesizer, utility, judge
  /models embedder <id>      switch the embedding model by name (re-embeds the corpus)

Every switch PERSISTS to config.yaml by default; --session applies it live only. Runs on the
very first launch (right before /config setup). Models are local Ollama ids only — cloud
support is shelved (2026-07-03).
""",
)
def _models(ctx, args):
    from config import get_config

    cfg = get_config()
    args, session, save = split_persist_flags(args)

    # The old cross-provider grammar (--provider <p> / a bare provider as 3rd arg) left with the
    # cloud-model shelve (2026-07-03): refuse it loudly rather than binding something surprising.
    if any(a.lower() == "--provider" for a in args):
        _print("  --provider was removed with the cloud-model shelve — models are local Ollama "
               "ids only; usage: /models <role|all> <model_id> [--save]")
        return

    if not args:
        _models_page(cfg, prompt=True, session=session)
        return

    sub = args[0].lower()

    if is_list_verb(sub) or sub in ("--check", "check"):
        _models_page(cfg, prompt=False)
        return

    if sub == "rescan":
        _models_page(cfg, prompt=True, session=session, rescan=True)
        return

    if sub == "tier":
        if len(args) < 2:
            _models_page(cfg, prompt=False)
            return
        tier = args[1]
        # Dict membership, not the dotted cfg.get("tiers.<tier>") path lookup — a tier key
        # containing a literal dot would otherwise be misread as two path segments and always
        # report unknown (hit this with the size-class key "0.8b", renamed to "800m" for the
        # same reason — see core/model_family.py).
        if tier not in cfg.get("tiers", {}):
            _print(f"  unknown tier: {tier} (defined: {list(cfg.get('tiers', {}))})")
            return
        _switch_tier(cfg, tier, session=session)
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
        _print(f"  unknown target: {role} (roles: {', '.join(_ROLES)}; or 'all'/'embedder'/'tier'/'list'/'rescan')")
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
