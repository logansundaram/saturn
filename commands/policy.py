"""
/policy — the gate policy as ONE object: its levers, its blast radius, its shareable profile.

policy.py consolidated the five gate-relaxation mechanisms into one object; this command is that
object's front door. The levers live here as subcommands (`risk` moves a tool's tier, `allow`
edits the run_shell prefix allowlist, `open` is the gate-off view), `can` answers "what can this
thing actually do to me right now", and `export`/`import` turn the whole posture into a file you
can read, share, and apply — the same profile drives headless runs via `saturn --policy <file>`.
The historical top-level spellings (/risk, /allow, /autoapprove) stay registered and delegate to
the exact same handler functions, so the two spellings can never drift.
"""

from __future__ import annotations

from pathlib import Path

from commands._framework import command, _print
from commands._utils import is_list_verb, is_remove_verb, parse_toggle_status, split_save_flag

_PROFILE_HEADER = """\
# Saturn policy profile — the whole gate posture as one shareable document.
# Apply with `/policy import <this file>` or `saturn --policy <this file>`.
"""


def apply_policy_file(path_str: str, save: bool = False) -> str:
    """Load a profile YAML and apply it: policy.apply_profile (threshold + permissions.json +
    airgap/redaction) then sync the LIVE tool registry so the overrides bite immediately.
    Returns a one-line summary; raises on an unreadable/invalid profile (the caller reports)."""
    import yaml

    from trust import policy

    path = Path(path_str).expanduser()
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = policy.apply_profile(profile, save=save)

    # Sync the live registry: back to declared tiers, then the profile's overrides — an import
    # REPLACES the posture (mirrors apply_profile replacing permissions.json), never layers on
    # whatever /risk edits the session had. Stale names (tools this install doesn't have) are
    # reported, not applied — the declared fail-closed tier stays in effect.
    from tools import registry

    for name, tier in registry.DECLARED_RISK.items():
        registry.TOOL_RISK[name] = tier
    stale = []
    applied = 0
    for name, tier in overrides.items():
        if name in registry.tools_by_name:
            registry.TOOL_RISK[name] = tier
            applied += 1
        else:
            stale.append(name)

    # A profile can flip runtime.airgap — drop the model cache so a cloud model built while the
    # boundary was open can't keep serving calls from llms._DERIVED_CACHE (mirrors the
    # /privacy airgap toggle, which does exactly this for exactly this reason).
    from core import llms

    llms.reset_models()

    summary = (
        f"policy applied from {path.name}: threshold={policy.tier()}, "
        f"{applied} risk override(s), {len(policy.shell_allow())} shell prefix(es)"
    )
    if stale:
        summary += f"  (ignored overrides for unknown tools: {', '.join(stale)})"
    return summary


# ── /policy risk — one tool's tier (also the legacy /risk) ───────────────────────────────────


def risk_handler(ctx, args):
    """`/policy risk` · `/risk` — override one tool's approval tier live; --save persists it to
    the policy file; `reset` restores the declared tier. One handler behind both spellings."""
    from trust import policy
    from tools import registry
    from tools.registry import tool as TOOLS, risk_of
    from tools.toolspec import RISK_TIERS as _RISK_TIERS

    # One --save grammar (case-insensitive, any position) — same flag as every other command.
    args, save = split_save_flag(args)

    if not args:
        overrides = policy.risk_overrides()
        _print("  current risk tiers (* = persisted override):")
        for t in TOOLS:
            mark = "*" if t.name in overrides else " "
            _print(f"    {risk_of(t.name):<14}{mark} {t.name}")
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
    if save:
        policy.set_risk_override(name, tier)
        _print(f"  {name}: {old} -> {tier} (saved — survives restarts; undo with "
               f"/policy risk {name} reset).")
    else:
        _print(f"  {name}: {old} -> {tier} (session only; add --save to persist).")


# ── /policy allow — the run_shell prefix allowlist (also the legacy /allow) ──────────────────


def allow_handler(ctx, args):
    """`/policy allow` · `/allow` — the persisted run_shell prefix allowlist. `add <prefix>` is
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
    _print("  run_shell commands starting with these run WITHOUT the approval gate:")
    for i, p in enumerate(prefixes, 1):
        _print(f"    {i}. {p}")
    _print("  remove: /policy allow remove <n|prefix>   add: /policy allow add <prefix>")


def _allow_add(policy, prefix: str) -> None:
    """Store one allowlist prefix + the shared confirmation copy (bare and `add` forms agree).
    add_shell_allow refuses text the matcher could never honor (ValueError: empty, or carrying a
    shell metacharacter) — rendered here as the command's own refusal, never the dispatcher's
    generic '/policy failed' catch-all."""
    try:
        added = policy.add_shell_allow(prefix)
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


# ── /policy open — the gate-off view (also the legacy /autoapprove · /yolo) ──────────────────


def _gate_status() -> str:
    """The one gate status line, shared by bare `/policy open` and bare `/autoapprove` — a pure
    readout, never a flip."""
    from trust import policy

    if policy.gate_off():
        return "⚠ gate OFF — /policy open off to restore"
    return f"gate: prompting above {policy.tier()}"


def open_handler(ctx, args):
    """`/policy open` · `/autoapprove` · `/yolo` — the gate-off view of the threshold. Bare is a
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
        _print("  ┏━ ⚠  AUTO-APPROVE ON")
        _print("  ┃  the approval gate is OPEN: auto-approve threshold =")
        _print("  ┃  destructive, so every tool call — including side-")
        _print("  ┃  effecting and destructive ones — runs WITHOUT asking.")
        _print("  ┃  /policy open off to restore the previous threshold.")
        _print("  ┗━")
    else:
        _print(f"  auto-approve off — gate threshold restored to `{policy.tier()}`.")


# ── /policy can — the blast radius ───────────────────────────────────────────────────────────


def can_handler(ctx, args):
    """`/policy can` — the blast radius: what runs without asking, what faces the gate first, and
    what cannot happen at all right now. Everything derives live from the one policy object, the
    live tool registry, the MCP client, and the airgap/dry-run knobs — never a captured snapshot."""
    from trust import policy
    from tools import registry
    from config import get_config
    from tui import ui

    cfg = get_config()
    airgap = bool(cfg.get("runtime.airgap", False))
    dry = bool(cfg.get("runtime.dry_run", False))
    threshold = policy.tier()
    gate_open = policy.gate_off()

    headline = ("⚠ GATE OFF — every tool call runs without asking" if gate_open
                else f"gate prompts above `{threshold}`")
    if airgap:
        headline += "  ·  ⛓ AIRGAP sealed"
    if dry:
        headline += "  ·  DRY-RUN (nothing executes)"
    ui.section("blast radius", headline)

    free, gated = [], []
    for t in registry.tool:
        risk = registry.risk_of(t.name)
        (free if policy.auto_approves(risk) else gated).append((t.name, risk))

    def _rows(pairs):
        return [
            (name, (risk, ui.risk_style(risk)),
             ("remote (MCP) — untrusted, fails closed to destructive", "dim")
             if name.startswith("mcp_") else "")
            for name, risk in pairs
        ]

    note = " (dry-run: every call is stubbed — nothing executes)" if dry else ""
    _print(f"  WITHOUT ASKING — runs the moment the model calls it{note}")
    rows = _rows(free)
    for p in policy.shell_allow():
        rows.append((f"run_shell `{p} …`", ("allow prefix", ui.risk_style("side_effecting")),
                     ("persisted /policy allow grant — token-boundary, no metacharacters",
                      "dim")))
    if rows:
        ui.table(rows)
    else:
        _print("    (nothing — every tool faces the gate first)")

    _print("  WITH YOUR APPROVAL — faces you at the gate first")
    if gated:
        ui.table(_rows(gated))
    elif gate_open:
        _print("    (nothing — the gate is OFF; /policy open off to restore prompting)")
    else:
        _print("    (nothing)")

    _print("  CANNOT (right now)")
    cannot = []
    if airgap:
        cannot.append(("web egress",
                       "air-gap is ON — web_search / web_extract / http_request refuse"))
        try:
            from tools import mcp_client

            remote = [t.name for s in mcp_client.status() if s.transport != "stdio"
                      for t in s.tools]
        except Exception:
            remote = []
        if remote:
            cannot.append(("remote MCP", "air-gap is ON — " + ", ".join(remote) + " refuse"))
        from commands.privacy import _offmachine_roles

        offmachine = _offmachine_roles(cfg)
        if offmachine:
            roles = ", ".join(f"{r} ({p}:{m})" for r, p, m in offmachine)
            cannot.append(("off-machine inference",
                           f"air-gap is ON — off-machine role(s) refuse to run: {roles}"))
    if not gate_open:
        cannot.append(("anything you reject",
                       f"every call above `{threshold}` faces you first — `n` at the gate "
                       "stops it"))
    if cannot:
        ui.table(cannot)
    else:
        _print("    (nothing is structurally blocked — the gate is OFF and the boundary open;")
        _print("     /policy open off and /privacy airgap on restore the walls)")

    # Effective quarantine mode (quarantine.mode() normalizes case + invalid values to "gate") —
    # the blast-radius footer must state the mode in force, not echo a raw config string.
    from trust import quarantine

    qmode = quarantine.mode()
    rmode = str(cfg.get("runtime.redaction", "off") or "off")
    try:
        from trust import signing

        signed = bool(cfg.get("runtime.sign_exports", True)) and signing.available()
    except Exception:
        signed = False
    _print(f"  posture: quarantine {qmode} · redaction {rmode} · export signing "
           f"{'on' if signed else 'off'}")


# ── /policy export · import — the shareable profile ─────────────────────────────────────────


def _export(ctx, args):
    import yaml

    from trust import policy
    from config import get_config

    dest_str = None
    positional: list[str] = []
    it = iter(args)
    for a in it:
        if a.lower() in ("-o", "--out", "--output"):
            dest_str = next(it, None)
            if dest_str is None:
                _print("  usage: /policy export [<path> | -o <path>] — -o needs a path; "
                       "nothing written")
                return
        else:
            positional.append(a)
    if dest_str is None and positional:
        dest_str = " ".join(positional)
    if dest_str:
        dest = Path(dest_str.strip('"')).expanduser()
    else:
        dest = get_config().path("exports") / "policy.yaml"
    profile = policy.export_profile()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            _PROFILE_HEADER + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as exc:
        _print(f"  could not write {dest}: {exc}")
        return
    _print(f"  policy profile exported -> {dest}")
    _print("  apply it anywhere: /policy import <file>  ·  saturn --policy <file>")


def _import(ctx, args):
    rest, save = split_save_flag(args)
    if not rest:
        _print("  usage: /policy import <file> [--save]")
        return
    path = " ".join(rest).strip('"')
    try:
        summary = apply_policy_file(path, save=save)
    except FileNotFoundError:
        _print(f"  no such file: {path}")
        return
    except Exception as exc:
        _print(f"  could not apply {path}: {exc}")
        return
    _print(f"  {summary}")
    if not save:
        _print("  (threshold + airgap/redaction set for this session; --save persists them)")


# ── the front door ───────────────────────────────────────────────────────────────────────────


@command(
    "policy",
    "The gate policy as one object: its levers, the blast radius, shareable profiles.",
    usage="/policy [risk <tool> [<tier>|reset] [--save] | "
          "allow [list | <prefix> | add <prefix> | remove <n|prefix>] | "
          "open [on|off] | can | export [<path> | -o <path>] | import <path> [--save]]",
    details="""
Every gate-relaxation mechanism (/risk, /allow, /autoapprove, runtime.auto_approve, --yolo) is a
view of one policy object. This command IS that object's front door — its levers, its blast
radius, and its file form:

  /policy                     the live posture: auto-approve threshold, persisted risk overrides,
                              the shell allowlist (with count), airgap + redaction + quarantine.
  /policy risk <tool> <tier> [--save]
                              override one tool's tier live (tiers prefix-match: read/side/dest);
                              --save persists to permissions.json; `<tool> reset` restores the
                              declared tier. (/risk is the same handler.)
  /policy allow [<prefix>]    allowlist a run_shell prefix that skips the gate (persisted;
                              token-boundary, case-insensitive, never with shell metacharacters);
                              bare (or `allow list` / `ls`) shows the stored prefixes;
                              `allow add <prefix>` always ADDS — the escape hatch when the prefix
                              itself starts with a removal word (`/policy allow add del *.tmp`)
                              or IS a lone reserved word (`/policy allow add ls`);
                              `allow remove <n|prefix>` revokes (rm/delete/del/forget/drop work
                              too, but only when the target is a stored number/prefix — anything
                              else is reported, never guessed). (/allow is the same handler.)
  /policy open [on|off]       the gate-off view: bare = STATUS only; `on` raises the threshold to
                              `destructive` (nothing prompts — the loud banner), `off` restores
                              the prior threshold. Opening is always an explicit verb.
                              (/autoapprove · /yolo are the same handler.)
  /policy can                 the blast radius: what runs WITHOUT asking, what needs YOUR
                              approval, what CANNOT happen right now (air-gap, gate) — derived
                              live from the policy, registry, MCP client, and runtime knobs.
  /policy export [<path> | -o <path>]
                              write the posture as a YAML profile (default:
                              logging/exports/policy.yaml). Shareable + versionable: commit it,
                              hand it to a teammate, keep a `paranoid.yaml` next to a `ci.yaml`.
  /policy import <path>       apply a profile. REPLACES the durable policy (permissions.json:
                              risk overrides + shell allowlist) and sets the threshold and the
                              airgap/redaction knobs for the session; --save also persists those
                              knobs to config.yaml. Applies to the live registry immediately.

Headless: `saturn -p "query" --policy <file>` applies a profile at process start — pin the exact
safety posture a script or CI job runs under instead of choosing between deny-all and --yolo.
""",
)
def _policy_cmd(ctx, args):
    from trust import policy
    from config import get_config

    if not args:
        cfg = get_config()
        overrides = policy.risk_overrides()
        allow = policy.shell_allow()
        _print("  gate policy (one object — /policy risk · allow · open are its levers):")
        threshold = policy.tier()
        label = "⚠ GATE OFF (everything auto-approved)" if threshold == "destructive" else threshold
        _print(f"    auto-approve threshold : {label}")
        if overrides:
            _print("    risk overrides         : " + ", ".join(
                f"{k}→{v}" for k, v in sorted(overrides.items())))
        else:
            _print("    risk overrides         : (none)")
        if allow:
            _print(f"    shell allowlist        : {len(allow)} prefix(es) — " + " · ".join(allow))
        else:
            _print("    shell allowlist        : (none)")
        from trust import quarantine

        _print(f"    airgap                 : {'on' if cfg.get('runtime.airgap', False) else 'off'}")
        _print(f"    redaction              : {cfg.get('runtime.redaction', 'off') or 'off'}")
        # Effective mode, not the raw string — an invalid value runs as "gate" (quarantine.mode).
        _print(f"    quarantine             : {quarantine.mode()}")
        _print("  blast radius: /policy can  ·  export: /policy export  ·  apply: /policy import <path>")
        return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "risk":
        return risk_handler(ctx, rest)
    if sub == "allow":
        return allow_handler(ctx, rest)
    if sub == "open":
        return open_handler(ctx, rest)
    if sub == "can":
        return can_handler(ctx, rest)
    if sub == "export":
        return _export(ctx, rest)
    if sub == "import":
        return _import(ctx, rest)

    _print(f"  unknown /policy subcommand: {sub!r} — try: risk, allow, open, can, export, "
           "import (or /policy --help)")


# ── top-level muscle-memory spellings ─────────────────────────────────────────────────────────
# /risk, /allow and /autoapprove stay first-class commands but are registered HERE, next to the
# handlers they delegate to: one file owns every view of the gate policy, so there is no parallel
# logic to audit and nothing to drift.


@command(
    "risk",
    "Override a tool's approval risk tier (live; --save persists it).",
    usage="/risk <tool> read_only|side_effecting|destructive [--save]  ·  /risk <tool> reset",
    details="""
Changes the approval risk tier of a single tool. The approval gate reads the tier live, so the
change takes effect on the next turn. With no args, lists every tool's current tier (a * marks a
persisted override).

Tiers:
  read_only       no side effects; runs freely, never prompts
  side_effecting  writes / external actions; prompts for approval
  destructive     irreversible / dangerous; prompts for approval

Persistence: by default a change lasts only this session. Add --save to write it to the policy
file (database/permissions.json) so it survives a restart. `/risk <tool> reset` restores the
tool's declared tier and removes any persisted override.

Canonical home: /policy risk — this top-level spelling is kept for muscle memory and delegates
to the exact same handler (zero drift between the two). /policy is the one gate-policy object;
its other levers are /policy allow (shell prefixes) and /policy open (the threshold itself —
the same `runtime.auto_approve` knob Shift+Tab cycles).

Tier names prefix-match: read / side / dest (or just r / s / d) all work.

Examples:
  /risk                                  list current tiers
  /risk write_file destructive           tighten one tool for this session
  /risk web_search side --save           require approval for web_search, permanently
  /risk web_search reset                 back to its declared tier (and forget the override)
""",
)
def _risk(ctx, args):
    risk_handler(ctx, args)


@command(
    "allow",
    "Allowlist shell command prefixes that skip the approval gate (persisted).",
    usage="/allow [list | <prefix words…> | add <prefix words…> | remove <n|prefix>]",
    details="""
run_shell is `destructive`, so every command normally faces the approval gate. /allow stores
command PREFIXES that you trust — a run_shell call whose command starts with one runs without
prompting, so the gate stops training you to mash `y` on `git status` while still guarding
everything else.

  /allow                       list the stored prefixes (also: list, ls)
  /allow git status            allow `git status`, `git status --short`, …
  /allow add del *.tmp         explicit add — the spelling for a prefix whose FIRST word is
                               itself a removal verb (rm/del/delete/drop/forget) or a lone
                               reserved word (`/allow add ls` allowlists `ls` itself; a lone
                               `list`/`ls` shows the list instead of granting it)
  /allow remove 2              remove a prefix by its list number
  /allow remove git status     …or by its exact text
                               (rm / delete / del / forget / drop work too — the shared
                               removal vocabulary; a removal verb only removes when the
                               target resolves, otherwise it points you at `add`)

Matching is strict on purpose:
  - token-boundary: `git status` does NOT match `git statusx`
  - case-insensitive
  - a command containing ; | & < > ` $ or a newline is NEVER exempt, even if its start
    matches — chaining/redirection can smuggle anything behind a trusted prefix, so those
    always face the human.
  - a background run (run_shell with background=true — detached, no timeout) is NEVER exempt
    either; the prefix covers bounded foreground runs only.

Persisted to the policy file database/permissions.json (alongside /policy risk --save
overrides), so it survives restarts.

Canonical home: /policy allow — this top-level spelling is kept for muscle memory and delegates
to the exact same handler (zero drift between the two). Allow narrow, read-only prefixes
(`git status`, `git log`, `ls`) — not broad ones (`git`, `python`).
""",
)
def _allow(ctx, args):
    allow_handler(ctx, args)


@command(
    "autoapprove",
    "The gate-off view: show the gate status, or open/close it explicitly.",
    aliases=("yolo",),
    usage="/autoapprove [on|off]   (bare = status readout, never a flip)",
    details="""
Shows or moves the gate-off view of the ONE gate policy (policy.py): `on` raises the
`runtime.auto_approve` threshold to `destructive` so every tier auto-approves; `off` restores
the threshold that was in effect before. Bare /autoapprove is a STATUS readout — opening the
gate is ALWAYS an explicit verb, so a habit-typed bare command can never silently drop the main
safety check. The status bar shows ⚠ GATE OFF for as long as the threshold sits there, however
it got there (this command, Shift+Tab cycling, or config.yaml).

⚠  `on` removes the main safety check. Use it only when you trust the task and the tools.
Prefer /policy risk to relax a single tool, or /policy allow to exempt specific shell commands,
while keeping the gate on. Session-only — it never persists.

Canonical home: /policy open — this top-level spelling is kept for muscle memory and delegates
to the exact same handler (zero drift between the two).

Examples:
  /autoapprove      status: the prompting threshold, or ⚠ gate OFF
  /autoapprove on
  /autoapprove off
  /yolo on          alias — same thing (mirrors the headless --yolo flag)
""",
)
def _autoapprove(ctx, args):
    open_handler(ctx, args)


# ── /dryrun — execution control, registered here with the other control surfaces ─────────────
# About execution, not egress (so it lives with the trust-&-control commands, not /privacy);
# the trajectory-level control proof point above the reactive per-call gate.
@command(
    "dryrun",
    "Counterfactual mode — plan and decide everything, execute nothing.",
    aliases=("dry",),
    usage="/dryrun [on|off]   (bare = status readout, never a flip)",
    details="""
When ON, the agent grounds, plans, and decides its tool calls exactly as it normally would — but
NOTHING actually runs. Every tool call is stubbed (`[DRY RUN] would execute …`): no files touched,
no shell, no network, no side effects. The final answer summarizes the whole intended arc — the
plan plus every tool call it meant to make, with its exact arguments — so you can see what the
agent WOULD do before approving any of it.

This is the control proof point at the trajectory level: the approval gate decides one call at a
time, reactively; a dry-run lets you inspect the entire plan up front. Run it on something you'd
never let execute blind — "delete every log and email me the result" — and watch the plan + the
exact `run_shell` / `http_request` it intended, with zero risk.

  /dryrun on     enter dry-run (stays on until you turn it off — the status bar shows DRY-RUN)
  /dryrun off    back to real execution
  /dryrun        status readout — never flips (like every trust toggle, mutation is explicit)
""",
)
def _dryrun(ctx, args):
    from config import get_config
    from tui import ui

    cfg = get_config()
    current = bool(cfg.get("runtime.dry_run", False))
    new = parse_toggle_status(args)
    if new is None:
        _print(f"  dry-run: {'ON — nothing executes' if current else 'off'}   "
               "(/dryrun on|off to change)")
        return
    if new == "invalid":
        _print(f"  usage: /dryrun on|off   (currently {'on' if current else 'off'})")
        return

    cfg.set("runtime.dry_run", new)
    try:
        ui.set_input_preview  # noqa: B018 — ensure tui imports; the bar reads runtime.dry_run live
    except Exception:
        pass

    if new:
        _print("  ┏━ ◊  DRY-RUN ON")
        _print("  ┃  the agent will PLAN and DECIDE but execute nothing.")
        _print("  ┃  every tool call is stubbed — no files, shell, or")
        _print("  ┃  network. the answer reports what it WOULD do.")
        _print("  ┗━")
    else:
        _print("  dry-run off — tools execute for real again.")
