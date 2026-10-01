"""
/policy — the ONE trust front door: what runs without asking, and what can leave this machine.

Two halves, one command:

- the gate: trust/policy.py consolidated the gate-relaxation mechanisms into one object, and
  `risk` (a tool's tier), `allow` (the run_shell prefix allowlist) and `open` (the gate-off view)
  are its levers;
- the network boundary: `egress` renders the session ledger (trust/egress.py — what actually
  left) and `airgap` is the seal.

Bare /policy is ONE posture readout over both halves, calm by default and loud on deviation (the
trust/receipt.py principle): its header repeats the startup posture line's own words
(`receipt.posture_spans`, the one source) or says the defaults hold, and each row is styled only
when it deviates from the safe default.
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import is_list_verb, is_remove_verb, parse_toggle_status, split_save_flag

# One byte formatter for every trust surface (textutil.human_bytes) — the per-answer receipt and
# the egress ledger must render the same byte count identically, or the "receipt echoes the
# ledger" story quietly stops being true.
from textutil import human_bytes as _human_bytes


# ── /policy risk — one tool's tier ───────────────────────────────────────────────────────────


def risk_handler(ctx, args):
    """`/policy risk` — override one tool's approval tier live; --save persists it to
    the policy file; `reset` restores the declared tier."""
    from trust import policy
    from tools import registry
    from tools.registry import tool as TOOLS, risk_of
    from tools.toolspec import RISK_TIERS as _RISK_TIERS

    # One --save grammar (case-insensitive, any position) — same flag as every other command.
    args, save = split_save_flag(args)

    if not args:
        from tui import ui

        overrides = policy.risk_overrides()
        # Through the shared listing vocabulary, like /tools and /mcp: the risk tier is the
        # semantic fact here, so it must carry the same green/yellow/red it wears at the gate —
        # and a long mcp_<server>_<tool> name must not push the column out of alignment.
        ui.section("risk tiers", "* = persisted override")
        rows = []
        for t in TOOLS:
            risk = risk_of(t.name)
            mark = "*" if t.name in overrides else ""
            rows.append(((risk, ui.risk_style(risk)), (mark, "accent"), t.name))
        ui.table(rows)
        _print("  set: /policy risk <tool> <tier> [--save]   restore: /policy risk <tool> reset")
        return

    if len(args) < 2:
        _print(f"  usage: /policy risk <tool> {'|'.join(_RISK_TIERS)} [--save]  ·  "
               "/policy risk <tool> reset")
        return

    name = args[0]
    if name not in registry.tools_by_name:
        import difflib

        hint = difflib.get_close_matches(name, registry.tools_by_name, n=1)
        suggest = f" — did you mean {hint[0]}?" if hint else ""
        _print(f"  unknown tool: {name} (see /tools){suggest}")
        return

    tier = args[1].lower()

    if tier == "reset":
        declared = registry.DECLARED_RISK.get(name, "destructive")
        old = risk_of(name)
        registry.TOOL_RISK[name] = declared
        registry.refresh_trust_classifications()  # the coercion pattern tracks live tiers
        had_override = policy.clear_risk_override(name)
        forgot = " (persisted override removed)" if had_override else ""
        _print(f"  {name}: {old} -> {declared} (declared tier){forgot}.")
        return

    # Tiers prefix-match, so `/policy risk web_search side` (or just `s`) works.
    tier_matches = [t for t in _RISK_TIERS if t.startswith(tier)]
    if len(tier_matches) != 1:
        _print(f"  unknown tier: {tier} (choose one of {', '.join(_RISK_TIERS)}, or reset)")
        return
    tier = tier_matches[0]

    old = risk_of(name)
    registry.TOOL_RISK[name] = tier
    registry.refresh_trust_classifications()  # the coercion pattern tracks live tiers
    if save:
        policy.set_risk_override(name, tier)
        _print(f"  {name}: {old} -> {tier} (saved — survives restarts; undo with "
               f"/policy risk {name} reset).")
    else:
        _print(f"  {name}: {old} -> {tier} (session only; add --save to persist).")


# ── /policy allow — the run_shell prefix allowlist ───────────────────────────────────────────


def allow_handler(ctx, args):
    """`/policy allow` — the persisted run_shell prefix allowlist. `add <prefix>` is
    the unambiguous escape hatch (mirrors /docs add) because the shared REMOVE_VERBS vocabulary
    (remove/rm/delete/del/forget/drop) is also a set of common SHELL words: a removal verb routes
    to removal ONLY when the target resolves to a stored entry — never a silent guess. The shared
    LIST_VERBS get the same care: a LONE `list`/`ls` is the listing (it must never silently
    CREATE a gate exemption for the prefix `list`), while `ls -la`-style verb-plus-words stays an
    add — `ls` is a real shell command, and listing never takes arguments."""
    from trust import policy

    if not args or (len(args) == 1 and is_list_verb(args[0])):
        return _allow_list(policy)

    verb = args[0].lower()

    # Explicit add — the only spelling that can allowlist a prefix whose first word is itself a
    # removal verb (`/policy allow add del *.tmp` allowlists `del *.tmp`).
    if verb == "add":
        if len(args) < 2:
            _print("  usage: /policy allow add <prefix words…>")
            return
        return _allow_add(policy, " ".join(args[1:]))

    if is_remove_verb(verb):
        if len(args) < 2:
            _print("  usage: /policy allow remove <n|prefix>   (to allowlist the word itself: "
                   f"/policy allow add {verb})")
            return
        # remove_shell_allow resolves a 1-based index or the exact (case-insensitive) text of a
        # stored prefix — an unresolved target is reported with the add escape hatch, because
        # silently treating `/policy allow del *.tmp` as a failed removal would leave NO spelling
        # that allowlists such a prefix.
        target = " ".join(args[1:])
        removed = policy.remove_shell_allow(target)
        if removed is None:
            _print(f"  no such allowlisted prefix: {target!r} — /policy allow lists them with "
                   "their numbers.")
            _print(f"  to add a prefix starting with `{verb}`, use: /policy allow add "
                   + " ".join(args))
        else:
            _print(f"  removed: {removed} (commands like this face the gate again).")
        return

    _allow_add(policy, " ".join(args))


def _allow_list(policy) -> None:
    """The allowlist readout — the bare `/policy allow` view and its explicit `list`/`ls`
    spellings (one renderer, so the spellings can't drift)."""
    prefixes = policy.shell_allow()
    if not prefixes:
        _print("  no allowlisted shell prefixes — add one with /policy allow <prefix words…>")
        _print("  e.g. /policy allow git status")
        return
    by_scope = policy.shell_allow_by_scope()
    lifetime = {"task": "this turn", "session": "this session", "persist": "persisted"}
    scope_of = {}
    for scope, items in by_scope.items():
        for pfx in items:
            scope_of.setdefault(pfx, scope)
    _print("  run_shell commands starting with these run WITHOUT the approval gate:")
    for i, p in enumerate(prefixes, 1):
        _print(f"    {i}. {p}   ({lifetime.get(scope_of.get(p, 'persist'), 'persisted')})")
    _print("  remove: /policy allow remove <n|prefix>   add: /policy allow add <prefix>")


def _allow_add(policy, prefix: str) -> None:
    """Store one allowlist prefix + the shared confirmation copy (bare and `add` forms agree).
    add_shell_allow refuses text the matcher could never honor (ValueError: empty, or carrying a
    shell metacharacter) — rendered here as the command's own refusal, never the dispatcher's
    generic '/policy failed' catch-all."""
    try:
        added = policy.add_shell_allow(prefix, scope="persist")  # the command IS the durable edit
    except ValueError as exc:
        _print(f"  cannot allowlist `{prefix}`: {exc}.")
        _print("  such a command always faces the gate — nothing was stored.")
        return
    if added:
        _print(f"  allowed: run_shell commands starting with `{prefix}` now skip the gate.")
        _print("  (persisted; undo with /policy allow remove. Chained/redirected commands "
               "still prompt.)")
    else:
        _print(f"  `{prefix}` is already allowlisted.")


# ── /policy open — the gate-off view ─────────────────────────────────────────────────────────


def _gate_status() -> str:
    """The one gate status line for bare `/policy open` — a pure readout, never a flip."""
    from trust import policy

    if policy.gate_off():
        return "⚠ gate OFF — /policy open off to restore"
    return f"gate: prompting above {policy.tier()}"


def open_handler(ctx, args):
    """`/policy open` — the gate-off view of the threshold. Bare is a
    STATUS readout; opening the gate is ALWAYS an explicit verb (a habit-typed bare command must
    never silently drop the main safety check)."""
    from trust import policy

    new = parse_toggle_status(args)
    if new is None:
        _print(f"  {_gate_status()}")
        return
    if new == "invalid":
        _print(f"  usage: /policy open on|off   ({_gate_status()})")
        return
    policy.set_gate_off(new)
    if new:
        # Loud but compact: one ⚠ line + one pointer — the heavy frame is reserved for the
        # approval gate, and the status bar carries ⚠ GATE OFF while the threshold sits open.
        _print("  ⚠ AUTO-APPROVE ON — every tool call, including destructive ones, runs "
               "WITHOUT asking.")
        _print("  (/policy open off restores the previous threshold; the status bar shows "
               "⚠ GATE OFF until then)")
    else:
        _print(f"  auto-approve off — gate threshold restored to `{policy.tier()}`.")


# ── bare /policy — the one posture readout ───────────────────────────────────────────────────


def _asks_cell(ui):
    """What the gate asks before, from the live threshold: calm at the read_only default, loud
    once loosened (the same colors the status bar and the startup posture line use)."""
    from trust import policy
    from tools.toolspec import RISK_TIERS

    threshold = policy.tier()
    if threshold == "destructive":
        return ("nothing — ⚠ GATE OFF  (/policy open off restores it)",
                ui.risk_style("destructive"))
    above = RISK_TIERS[RISK_TIERS.index(threshold) + 1:] if threshold in RISK_TIERS else RISK_TIERS
    text = " and ".join(above) + " calls"
    if threshold == "read_only":
        return (text, None)
    return (f"{text} only (auto-approves up to {threshold})", ui.risk_style("side_effecting"))


def _models_cell(ui, inf: dict, airgap: bool):
    """Where the words are computed: the chat model and the embedder, local or behind a remote
    OLLAMA_HOST (egress._inference — THE locality classifier; the endpoint spelled by
    remote_ollama_label — never re-rolled)."""
    from trust.egress import remote_ollama_label

    names = " · ".join(b["model"] if b["role"] == "chat" else f"{b['role']} {b['model']}"
                       for b in inf["bindings"])
    if inf["all_local"]:
        return (f"local — {names}", ui.risk_style("read_only"))
    where = remote_ollama_label(inf)
    if airgap:
        return (f"BLOCKED by air-gap — {where}  ({names})", ui.risk_style("destructive"))
    return (f"OFF this machine — {where}  ({names})", ui.risk_style("side_effecting"))


def _mcp_rows(ui, airgap: bool) -> list:
    """One row per MCP server: a stdio server is a local process (its egress is its own); a
    remote one gets the call's arguments."""
    from tools import mcp_client

    statuses = mcp_client.status()
    if not statuses:
        if mcp_client.configured():
            return [("mcp servers", ("configured but not loaded — /mcp reload", "dim"))]
        return [("mcp servers", ("none", "dim"))]
    rows = []
    for s in statuses:
        if s.transport == "stdio":
            text, style = f"local process ({s.state})", "dim"
        elif airgap:
            text, style = f"remote → {s.target} — sealed by air-gap", "dim"
        else:
            text, style = f"remote → {s.target} ({s.state})", ui.risk_style("side_effecting")
        rows.append((f"mcp {s.name}", (text, style)))
    return rows


def _data_rows(ui) -> list:
    """Where Saturn keeps what it stores. One row when every store shares a folder (the calm
    case); a store outside it gets its own row, so nothing is hidden behind the summary."""
    import os
    from pathlib import Path
    from config import get_config

    cfg = get_config()
    stores = []
    for label, name in (("documents", "documents"), ("memory", "memory"),
                        ("traces", "db_sqlite"), ("sessions", "sessions"),
                        ("exports", "exports"), ("gate policy", "permissions")):
        try:
            stores.append((label, str(cfg.path(name))))
        except KeyError:
            continue
    if not stores:
        return []
    try:
        root = os.path.commonpath([p for _, p in stores])
    except ValueError:
        root = ""
    # A root as broad as / or the home folder names nothing — list each store instead.
    if root and root not in (os.sep, str(Path.home())):
        return [("your data", (f"on this machine — {root}", "dim"))]
    return [(f"your data: {label}", (path, "dim")) for label, path in stores]


# How long a gate "always allow" answer lasts, per runtime.grant_scope — the same words the
# allowlist listing uses for each prefix's lifetime.
_GRANT_LIFETIME = {"task": "this turn", "session": "this session", "persist": "until revoked"}


def _posture(ctx) -> None:
    """Bare /policy — the whole trust posture in one readout: the gate (what runs without
    asking), the boundary (where the words are computed, what can leave, the seal, the
    quarantine), and where the data lives. Every facet is the EFFECTIVE value from its owning
    module (policy.tier(), egress.airgap_on(), quarantine.mode(), egress._inference()), never
    a raw config string — this readout must not assert a posture nothing enforces."""
    from trust import egress, policy, quarantine
    from trust.egress import _inference
    from tui import ui

    airgap = egress.airgap_on()
    inf = _inference()

    # The header speaks the startup posture line's words (deviation-only, the one source) — or,
    # when every facet sits at its safe default, one calm sentence.
    try:
        from trust import receipt

        spans = receipt.posture_spans()
    except Exception:
        spans = []
    verdict = (" · ".join(text for text, _ in spans) if spans
               else "defaults hold — changes ask first; nothing leaves unless a web tool runs")
    ui.section("policy", verdict)

    overrides = policy.risk_overrides()
    allow = policy.shell_allow()
    scope = policy.default_grant_scope()
    if allow:
        by = policy.shell_allow_by_scope()
        scoped = " · ".join(f"{k}: {len(v)}" for k, v in by.items() if v)
        allow_cell = (f"{len(allow)} prefix(es) ({scoped}) — " + " · ".join(allow),
                      ui.risk_style("side_effecting"))
    else:
        allow_cell = ("none", "dim")
    gate_rows = [
        ("asks before", _asks_cell(ui)),
        ("risk overrides",
         (", ".join(f"{k}→{v}" for k, v in sorted(overrides.items())), ui.risk_style("side_effecting"))
         if overrides else ("none", "dim")),
        ("shell allowlist", allow_cell),
        ("always-allow", (f"a yes lasts {_GRANT_LIFETIME.get(scope, scope)}  "
                          f"(runtime.grant_scope: {scope})", "dim")),
    ]

    q = quarantine.mode()
    web = (("sealed by air-gap", "dim") if airgap
           else ("search query → DuckDuckGo; pages fetched directly", None))
    rows = [
        ("models", _models_cell(ui, inf, airgap)),
        ("web tools", web),
        *_mcp_rows(ui, airgap),
        ("air-gap", ("ON — web, remote MCP, and remote models are blocked", "accent") if airgap
         else ("off — /policy airgap on seals the boundary", "dim")),
        ("quarantine", (f"{q} — untrusted tool output is screened", "dim") if q == "gate"
         else (f"{q} — untrusted tool output is {'not screened' if q == 'off' else 'only flagged'}",
               ui.risk_style("side_effecting"))),
        ("telemetry", ("none — nothing phones home", "dim")),
        *_data_rows(ui),
    ]
    # One label column across both halves, so the readout scans as one block.
    width = max(len(label) for label, _ in gate_rows + rows)
    _print("  the gate — what runs without asking")
    ui.table([(label.ljust(width), cell) for label, cell in gate_rows])
    _print("  the boundary — what can leave this machine")
    ui.table([(label.ljust(width), cell) for label, cell in rows])
    _print("")
    _print("  more: /policy egress (what left) · airgap · risk · allow · open · --help")


# ── /policy egress — the session ledger ──────────────────────────────────────────────────────


def _egress(ctx, args):
    from trust import egress
    from tui import ui

    if args and args[0].lower() in ("clear", "reset"):
        egress.clear()
        _print("  egress ledger cleared for this session.")
        return

    limit = None
    for a in args:
        if a.lstrip("+-").isdigit():
            limit = max(1, int(a))

    evs = egress.events()
    s = egress.summary()

    airgap = ""
    try:
        if egress.airgap_on():
            airgap = "  ·  air-gap ON"
    except Exception:
        pass

    # A clear-emptied ledger must never read as "nothing left this machine" — the counts below
    # are since the clear. Same unknown-over-local-only contract the receipt applies.
    cleared = bool(s.get("cleared"))
    if not evs:
        if cleared:
            ui.section("egress", "no events since the ledger was cleared" + airgap)
            _print("  the in-memory ledger was cleared this session — earlier egress is not")
            _print("  shown here.")
        else:
            ui.section("egress", "nothing has left this machine this session" + airgap)
            _print("  the boundary has stayed closed — no web, http, MCP, or remote-model egress.")
        return

    hosts = s["hosts"]
    headline = (
        f"{s['sent']} egress event(s), {_human_bytes(s['bytes'])} sent to "
        f"{len(hosts)} host(s)"
        + (f", {s['blocked']} blocked" if s["blocked"] else "")
        + (f", {s['untracked']} untracked" if s["untracked"] else "")
        + airgap
    )
    ui.section("egress", headline)
    if cleared:
        _print("  (ledger cleared this session — counts are since the clear)")
    if s["untracked"]:
        _print("  untracked = a shell command or stdio MCP server ran; Saturn cannot see whether")
        _print("  it used the network (air-gap holds these for your approval).")

    shown = evs[-limit:] if limit else evs
    if limit and len(evs) > limit:
        _print(f"  last {limit} of {len(evs)} event(s) — newest last:")

    rows = []
    for e in shown:
        when = (e.ts or "")[11:19]
        status = (
            ("BLOCKED", ui.risk_style("destructive")) if e.status == egress.BLOCKED
            else ("UNTRACKED", "yellow") if e.status == egress.UNTRACKED
            else (_human_bytes(e.n_bytes), "dim")
        )
        rows.append((when, e.channel, e.host, e.detail, status))
    ui.table(rows, styles=["dim", "accent", None, None, None])

    if s["by_channel"]:
        mix = " · ".join(f"{v} {k}" for k, v in sorted(s["by_channel"].items()))
        _print(f"  by channel: {mix}")


# ── /policy airgap — seal the boundary ───────────────────────────────────────────────────────


def _airgap(ctx, args):
    from trust import egress
    from config import get_config, persist
    from tui import ui

    cfg = get_config()
    toggle_args, save = split_save_flag(args)
    new = parse_toggle_status(toggle_args)

    # No on/off -> status; `--save` alone persists the CURRENT value (the shared convention —
    # it mutates nothing live, so the seal can't silently flip).
    if new is None:
        if save:
            cur = "on" if egress.airgap_on() else "off"
            try:
                persist("runtime.airgap")
                _print(f"  airgap is {cur} — saved runtime.airgap to config.yaml "
                       "(no change made; /policy airgap on|off changes it).")
            except Exception as exc:
                _print(f"  (could not persist to config.yaml: {exc})")
            return
        _show_airgap(ctx, cfg, ui, egress)
        return
    if new == "invalid":
        _print(f"  usage: /policy airgap on|off [--save]   (currently "
               f"{'on' if cfg.get('runtime.airgap', False) else 'off'})")
        return

    cfg.set("runtime.airgap", new)
    # Drop the model cache so a remote model built while air-gap was OFF can't keep serving calls —
    # the next get_model rebuild re-checks the gate and refuses. Web tools / MCP check the gate live
    # on every call, so they need nothing here.
    try:
        from core.llms import reset_models
        reset_models()
    except Exception:
        pass

    if save:
        try:
            persist("runtime.airgap")
        except Exception as exc:
            _print(f"  (could not persist to config.yaml: {exc})")

    if new:
        offmachine = _offmachine_models()
        # Loud but compact: one ⛓ line + one pointer — the heavy frame is reserved for the
        # approval gate; the status bar carries ⛓ AIRGAP while the seal holds.
        _print("  ⛓ AIR-GAP ON — web tools, remote MCP calls, and off-machine models are "
               "blocked + logged.")
        _print("  (/policy airgap off re-opens · /policy egress lists any blocked attempts)")
        if offmachine:
            models = ", ".join(f"{kind} {m} ({where})" for kind, where, m in offmachine)
            _print(f"  ⚠  off-machine model(s) will now FAIL: {models}")
            _print("     point OLLAMA_HOST at this machine (or unset it) and restart to run "
                   "local.")
    else:
        _print("  air-gap off — network access restored.")
    if save:
        _print("  saved runtime.airgap to config.yaml (survives restart).")


def _offmachine_models():
    """(kind, where, model) for every model (chat / embedder) whose inference LEAVES this
    machine — one behind a remote OLLAMA_HOST (egress._inference, the one locality classifier;
    the endpoint label via remote_ollama_label, the one spelling)."""
    from trust.egress import _inference, remote_ollama_label

    inf = _inference()
    return [(b["role"], remote_ollama_label(inf), b["model"])
            for b in inf["bindings"] if b["locality"] == "remote"]


def _show_airgap(ctx, cfg, ui, egress):
    """Bare `/policy airgap` — the enforcement posture: what is open vs sealed right now."""
    from trust.egress import _inference, remote_ollama_label

    on = egress.airgap_on()
    inf = _inference()
    offmachine = not inf["all_local"]
    if on:
        verdict = "SEALED — web, remote MCP, and off-machine models are blocked"
    elif offmachine:
        verdict = "open — and off-machine model(s) are sending prompts off this machine right now"
    else:
        verdict = "open — but every model is local, so nothing leaves unless a web tool is used"
    ui.section("air-gap", verdict)

    sealed = lambda: ("sealed", ui.risk_style("read_only")) if on else ("open", ui.risk_style("destructive"))

    rows = []
    for b in inf["bindings"]:
        if b["locality"] == "local":
            rows.append((b["role"], b["model"], ("local", ui.risk_style("read_only"))))
        else:
            where = f"remote — {remote_ollama_label(inf)}"
            label = f"BLOCKED — {where}" if on else where
            rows.append((b["role"], b["model"], (label, ui.risk_style("destructive"))))
    _print("  inference (off-machine models refuse to run under air-gap)")
    ui.table(rows)

    _print("  egress paths")
    ui.table(
        [
            ("web tools", "web_search / web_extract", sealed()),
            ("remote MCP", "http/sse server calls (stdio = local process)", sealed()),
            ("off-machine models", "prompts + context to a remote Ollama",
             ("sealed", ui.risk_style("read_only")) if (on or not offmachine)
             else ("open", ui.risk_style("destructive"))),
        ]
    )

    s = egress.summary()
    _print(f"  this session: {s['sent']} egress event(s), {s['blocked']} blocked "
           f"— full ledger in /policy egress")
    if not on:
        _print("  seal it with  /policy airgap on   (then re-run /policy airgap to verify).")


# ── the front door ───────────────────────────────────────────────────────────────────────────


@command(
    "policy",
    "Your trust settings in one place: what runs without asking, and what can leave this machine.",
    usage="/policy [risk <tool> [<tier>|reset] [--save] | "
          "allow [list | <prefix> | add <prefix> | remove <n|prefix>] | "
          "open [on|off] | egress [clear|n] | airgap [on|off] [--save]]",
    details="""
One front door for the whole trust posture. Bare /policy answers "what runs without asking me,
and what can leave this machine?" — the subcommands are the levers and the proof.

  /policy                     the posture: what the gate asks before, overrides and the shell
                              allowlist, where the model and the embedder run (local vs a remote
                              OLLAMA_HOST), what the web tools and MCP servers send, the air-gap,
                              the injection quarantine, and where your data lives. Quiet when the
                              defaults hold; anything loosened or leaving the machine is colored.

The gate — what runs without asking:
  /policy risk <tool> <tier> [--save]
                              override one tool's tier live (tiers prefix-match: read/side/dest);
                              --save persists to permissions.json; `<tool> reset` restores the
                              declared tier. Bare `risk` lists every tool's tier.
  /policy allow [<prefix>]    allowlist a run_shell prefix that skips the gate (persisted;
                              token-boundary, case-insensitive, never with shell metacharacters —
                              chained/redirected commands always face the human); bare (or
                              `allow list` / `ls`) shows the stored prefixes;
                              `allow add <prefix>` always ADDS — the escape hatch when the prefix
                              itself starts with a removal word (`/policy allow add del *.tmp`)
                              or IS a lone reserved word (`/policy allow add ls`);
                              `allow remove <n|prefix>` revokes (rm/delete/del/forget/drop work
                              too, but only when the target is a stored number/prefix — anything
                              else is reported, never guessed). Allow narrow, read-only prefixes
                              (`git status`, `ls`) — not broad ones (`git`, `python`).
  /policy open [on|off]       the gate-off view: bare = STATUS only; `on` raises the threshold to
                              `destructive` (nothing prompts — the loud banner), `off` restores
                              the prior threshold. Opening is always an explicit verb.

The boundary — what can leave this machine:
  /policy egress              the ledger: what ACTUALLY left this session — every web search, page
                              fetch, remote MCP call, and remote-Ollama invocation, with
                              channel/host/bytes, plus every attempt BLOCKED by air-gap. Pair it
                              with a network monitor and the two agree.
    /policy egress 20           just the last 20 events
    /policy egress clear        reset the in-memory ledger for this session
  /policy airgap              seal the boundary. With no argument, prints the enforcement posture
                              (what is open vs sealed right now). When ON: web tools refuse,
                              remote MCP calls refuse, and a remote Ollama refuses to run.
    /policy airgap on|off [--save]
                                set; --save persists to config.yaml (`--save` with no value
                                persists the CURRENT setting without changing it)

Every relaxation (these levers, runtime.auto_approve, Shift+Tab cycling, the headless --yolo
flag) is a view of one policy object. Telemetry: none — nothing phones home.

Related: /trace export (a portable, replayable run record).
""",
)
def _policy_cmd(ctx, args):
    if not args:
        return _posture(ctx)

    sub = args[0].lower()
    rest = args[1:]

    if sub == "risk":
        return risk_handler(ctx, rest)
    if sub == "allow":
        return allow_handler(ctx, rest)
    if sub == "open":
        return open_handler(ctx, rest)
    if sub in ("egress", "ledger"):
        return _egress(ctx, rest)
    if sub in ("airgap", "air-gap", "seal"):
        return _airgap(ctx, rest)

    _print(f"  unknown /policy subcommand: {sub!r} — try: risk, allow, open, egress, airgap "
           "(or /policy --help)")
