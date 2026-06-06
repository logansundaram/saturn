from commands._framework import command, _print
from registry import tool as TOOLS, risk_of


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
def _tools(ctx, args):
    _print("  registered tools:")
    for t in TOOLS:
        risk = risk_of(t.name)
        desc = (t.description or "").strip().splitlines()
        first = desc[0] if desc else ""
        _print(f"    [{risk:<14}] {t.name:<22} {first}")
