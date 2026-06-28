"""
/policy — the gate policy as ONE object: its levers, in one place.

policy.py consolidated the five gate-relaxation mechanisms into one object; this command is that
object's front door. The levers live here as subcommands (`risk` moves a tool's tier, `allow`
edits the run_shell prefix allowlist, `open` is the gate-off view). The historical top-level
spellings (/risk, /allow, /autoapprove) stay registered and delegate to the exact same handler
functions, so the two spellings can never drift.
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import is_list_verb, is_remove_verb, parse_toggle_status, split_save_flag


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


# ── the front door ───────────────────────────────────────────────────────────────────────────


@command(
    "policy",
    "The gate policy as one object: its levers, in one place.",
    usage="/policy [risk <tool> [<tier>|reset] [--save] | "
          "allow [list | <prefix> | add <prefix> | remove <n|prefix>] | "
          "open [on|off]]",
    details="""
Every gate-relaxation mechanism (/risk, /allow, /autoapprove, runtime.auto_approve, --yolo) is a
view of one policy object. This command IS that object's front door — its levers:

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
        return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "risk":
        return risk_handler(ctx, rest)
    if sub == "allow":
        return allow_handler(ctx, rest)
    if sub == "open":
        return open_handler(ctx, rest)

    _print(f"  unknown /policy subcommand: {sub!r} — try: risk, allow, open "
           "(or /policy --help)")


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
