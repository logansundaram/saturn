from commands._framework import command, _print
from registry import tool as TOOLS, TOOL_RISK, risk_of

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
def _risk(ctx, args):
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
