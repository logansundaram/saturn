from commands._framework import command, _print
from commands._utils import parse_toggle_status


@command(
    "plan",
    "Show the plan; control review mode and the mid-run pause.",
    usage="/plan | /plan review [on|off] | /plan pause",
    details="""
The plan is the agent's living checklist. With no args, renders the most recent one — every step
with its status glyph and intended tool (empty until you've run at least one turn).

Status glyphs:  · pending   ▸ active   ✓ done   ⨯ skipped   ⊘ blocked   ✗ error   − cancelled

Execution is always step-at-a-time: the engine works exactly the current step, records its
result on the plan, and reflects before continuing. This command controls the human-in-the-loop
plan-review architecture around that:

  /plan review [on|off]   Persistent review mode. When on, EVERY turn pauses at the first step
                          boundary so you can inspect and edit the plan before any tool runs.
                          Bare `/plan review` shows the current state; explicit on|off changes it.
                          Off by default.

  /plan pause             Arm a ONE-SHOT pause: the next turn pauses at its first step boundary for
                          review, then runs normally afterwards. While a turn is running, Esc on an
                          empty line pauses for review at the next step; type a correction first and
                          THEN press Esc to STEER the running turn — the remaining steps are
                          redrafted around your correction without losing the turn. (Plain typing +
                          Enter still queues a follow-up to run after the turn finishes.)

When a turn pauses, you get an interactive editor. Its verbs (also usable live):
  add <label> [::tool] · edit <id> <label> · tool <id> <name|none>
  status <id> <status> · move <id> <pos>   · drop <id>
  go / <enter> to run the edited plan, abort to stop the turn.

Examples:
  /plan                      show the current plan
  /plan review               is review mode on?
  /plan review on            vet every plan before it runs
  /plan pause                review just the next turn's plan
""",
)
def _plan(ctx, args):
    from tui import ui

    if not args:
        _print("  current plan:")
        ui.render_plan(ctx.state.get("plan", []))
        mode = "on" if ctx.review_plan else "off"
        _print(f"  review mode: {mode}   (see /plan --help)")
        return

    sub = args[0].lower()

    if sub == "review":
        new = parse_toggle_status(args[1:])
        if new is None:
            cur = "on" if ctx.review_plan else "off"
            _print(f"  plan review is {cur} — /plan review on|off to change.")
            return
        if new == "invalid":
            _print(f"  usage: /plan review [on|off]   (currently {'on' if ctx.review_plan else 'off'})")
            return
        ctx.review_plan = new
        if new:
            _print("  plan review ON — every turn pauses at the first step so you can edit the plan.")
        else:
            _print("  plan review off — turns run without the pre-execution review pause.")
        return

    if sub == "pause":
        from core.plan_ops import get_pause_controller
        get_pause_controller().request("user", "one-shot: review the plan before it runs")
        _print("  armed — the next turn will pause at its first step boundary for plan review.")
        _print("  (during a running turn: Esc on an empty line pauses for review at the next step;")
        _print("   type a correction first, then Esc, to steer the running turn instead.)")
        return

    _print(f"  unknown /plan subcommand: {sub!r} — try: review, pause (or /plan --help)")
