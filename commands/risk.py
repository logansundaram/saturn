from commands._framework import command, _print
from registry import tool as TOOLS, risk_of
from toolspec import RISK_TIERS as _RISK_TIERS


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

/risk, /allow and /autoapprove are three views of ONE gate policy (policy.py): /risk moves a
tool's tier, /allow exempts exact shell-command prefixes, /autoapprove moves the auto-approve
THRESHOLD itself (the same `runtime.auto_approve` knob Shift+Tab cycles).

Tier names prefix-match: read / side / dest (or just r / s / d) all work.

Examples:
  /risk                                  list current tiers
  /risk write_file destructive           tighten one tool for this session
  /risk web_search side --save           require approval for web_search, permanently
  /risk web_search reset                 back to its declared tier (and forget the override)
""",
)
def _risk(ctx, args):
    import policy
    import registry

    if not args:
        overrides = policy.risk_overrides()
        _print("  current risk tiers (* = persisted override):")
        for t in TOOLS:
            mark = "*" if t.name in overrides else " "
            _print(f"    {risk_of(t.name):<14}{mark} {t.name}")
        _print("  set: /risk <tool> <tier> [--save]   restore: /risk <tool> reset")
        return

    if len(args) < 2:
        _print(f"  usage: /risk <tool> {'|'.join(_RISK_TIERS)} [--save]  ·  /risk <tool> reset")
        return

    name = args[0]
    if name not in registry.tools_by_name:
        import difflib

        hint = difflib.get_close_matches(name, registry.tools_by_name, n=1)
        suggest = f" — did you mean {hint[0]}?" if hint else ""
        _print(f"  unknown tool: {name} (see /tools){suggest}")
        return

    tier = args[1].lower()
    save = any(a in ("--save", "-s") for a in args[2:])

    if tier == "reset":
        declared = registry.DECLARED_RISK.get(name, "destructive")
        old = risk_of(name)
        registry.TOOL_RISK[name] = declared
        had_override = policy.clear_risk_override(name)
        forgot = " (persisted override removed)" if had_override else ""
        _print(f"  {name}: {old} -> {declared} (declared tier){forgot}.")
        return

    # Tiers prefix-match, so `/risk web_search side` (or just `s`) works.
    tier_matches = [t for t in _RISK_TIERS if t.startswith(tier)]
    if len(tier_matches) != 1:
        _print(f"  unknown tier: {tier} (choose one of {', '.join(_RISK_TIERS)}, or reset)")
        return
    tier = tier_matches[0]

    old = risk_of(name)
    registry.TOOL_RISK[name] = tier
    if save:
        policy.set_risk_override(name, tier)
        _print(f"  {name}: {old} -> {tier} (saved — survives restarts; undo with /risk {name} reset).")
    else:
        _print(f"  {name}: {old} -> {tier} (session only; add --save to persist).")
