"""
/policy — the gate policy as ONE object: its levers, in one place.

policy.py consolidated the five gate-relaxation mechanisms into one object; this command is that
object's front door. The levers live here as subcommands (`risk` moves a tool's tier, `allow`
edits the run_shell prefix allowlist, `open` is the gate-off view). The historical top-level
spellings (/risk, /allow, /autoapprove, /yolo) were CUT in the 2026-07-06 surface trim — they
land on _RENAMED pointers to the /policy subcommands, so muscle memory still lands softly and
there is exactly ONE spelling to learn (and zero parallel registrations to audit).
"""

from __future__ import annotations

from commands._framework import command, _print
from commands._utils import is_list_verb, is_remove_verb, parse_toggle_status, split_save_flag


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
        # Loud but compact (2026-07-06 declutter): one ⚠ line + one pointer — the heavy frame
        # is reserved for the approval gate and plan review (the ui design vocabulary), and the
        # status bar carries ⚠ GATE OFF for as long as the threshold sits open.
        _print("  ⚠ AUTO-APPROVE ON — every tool call, including destructive ones, runs "
               "WITHOUT asking.")
        _print("  (/policy open off restores the previous threshold; the status bar shows "
               "⚠ GATE OFF until then)")
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
Every gate-relaxation mechanism (this command's levers, runtime.auto_approve, Shift+Tab cycling,
the headless --yolo flag) is a view of one policy object. This command IS that object's front
door — its levers:

  /policy                     the live posture: auto-approve threshold, persisted risk overrides,
                              the shell allowlist (with count), airgap + redaction + quarantine.
  /policy risk <tool> <tier> [--save]
                              override one tool's tier live (tiers prefix-match: read/side/dest);
                              --save persists to permissions.json; `<tool> reset` restores the
                              declared tier.
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

(The old top-level spellings /risk, /allow, /autoapprove, /yolo were folded in here 2026-07-06 —
typing one prints a pointer to its subcommand.)
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


# (The top-level muscle-memory spellings — /risk, /allow, /autoapprove(/yolo) — were CUT in the
# 2026-07-06 surface trim: three registered commands whose only job was to delegate to the
# subcommands above. _RENAMED points each at its /policy lever, same soft landing as /dryrun —
# which was CUT 2026-07-03: redundant twice over (/plan review shows intent before execution,
# the gate shows every call before it runs) and misleading on multi-step turns, where every
# decision past the first stubbed observation is fiction.)
