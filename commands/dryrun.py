from commands._framework import command, _print
from commands._utils import parse_toggle_status


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
        _print("  ┏━ ◊  DRY-RUN ON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        _print("  ┃  the agent will PLAN and DECIDE but execute nothing.")
        _print("  ┃  every tool call is stubbed — no files, shell, or")
        _print("  ┃  network. the answer reports what it WOULD do.")
        _print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        _print("  dry-run off — tools execute for real again.")
