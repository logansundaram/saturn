from commands._framework import command, _print
from commands._utils import parse_toggle_status, split_save_flag

# Printed for the removed recipe subcommands (save/recipes/run) so muscle memory lands softly.
_RECIPES_REMOVED = (
    "  plan recipes were removed — save a reusable prompt as a user command instead: "
    "database/commands/<name>.md -> /<name>"
)


@command(
    "plan",
    "Show the plan; control review mode, mid-run pause, and lockstep.",
    usage="/plan | /plan review [on|off] | /plan pause | /plan lockstep [on|off] [--save]",
    details="""
The plan is the agent's living checklist. With no args, renders the most recent one — every step
with its status glyph and intended tool (empty until you've run at least one turn).

Status glyphs:  · pending   ▸ active   ✓ done   ⨯ skipped

This command also controls the human-in-the-loop plan-review architecture:

  /plan review [on|off]   Persistent review mode. When on, EVERY turn pauses at the first step
                          boundary so you can inspect and edit the plan before any tool runs.
                          Bare `/plan review` shows the current state; explicit on|off changes it.
                          Off by default.

  /plan pause             Arm a ONE-SHOT pause: the next turn pauses at its first step boundary for
                          review, then runs normally afterwards. While a turn is running, Esc on an
                          empty line pauses for review at the next step; type a correction first and
                          THEN press Esc to STEER the running turn — the text is injected at the next
                          step boundary so the agent adjusts course without losing the turn. (Plain
                          typing + Enter still queues a follow-up to run after the turn finishes.)

  /plan lockstep [on|off] [--save]
                          Lockstep execution. When on (the default), the agent works the plan one
                          step at a time, strongly directed to the current step, so the plan is
                          followed closely. When off, it free-runs with only a soft next-step
                          pointer. Bare `/plan lockstep` shows the current state. Sets
                          runtime.lockstep (session only); add --save to persist it to config.yaml.

When a turn pauses, you get an interactive editor. Its verbs (also usable live):
  add <label> [::tool] · edit <id> <label> · tool <id> <name|none>
  status <id> <status> · move <id> <pos>   · drop <id>
  go / <enter> to run the edited plan, abort to stop the turn.

(Plan recipes were removed — for a reusable prompt, drop a markdown template into
database/commands/<name>.md and it becomes /<name>.)

Examples:
  /plan                      show the current plan
  /plan review               is review mode on?
  /plan review on            vet every plan before it runs
  /plan pause                review just the next turn's plan
  /plan lockstep off         let the agent free-run the plan
  /plan lockstep off --save  …and persist it to config.yaml
  /plan lockstep --save      persist the CURRENT setting unchanged (like /config persist)
""",
)
def _plan(ctx, args):
    from tui import ui

    if not args:
        _print("  current plan:")
        ui.render_plan(ctx.state.get("plan", []))
        mode = "on" if ctx.review_plan else "off"
        from config import get_config
        lock = "on" if get_config().lockstep else "off"
        _print(f"  review mode: {mode}  ·  lockstep: {lock}   (see /plan --help)")
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

    if sub == "lockstep":
        from config import get_config
        from commands.config import _persist_key

        cfg = get_config()
        rest, save = split_save_flag(args[1:])
        new = parse_toggle_status(rest)
        if new is None:
            if save:  # `/plan lockstep --save` persists the current value (like /config persist)
                _persist_key(cfg, "runtime.lockstep")
                return
            cur = "on" if cfg.lockstep else "off"
            _print(f"  lockstep is {cur} — /plan lockstep on|off to change (add --save to persist).")
            return
        if new == "invalid":
            _print(f"  usage: /plan lockstep [on|off] [--save]   (currently {'on' if cfg.lockstep else 'off'})")
            return
        cfg.set("runtime.lockstep", new)
        _print(
            f"  lockstep {'on' if new else 'off'} — "
            + ("the agent follows the plan one step at a time." if new
               else "the agent free-runs the plan with a soft pointer.")
        )
        if save:
            _persist_key(cfg, "runtime.lockstep")
        else:
            _print("  (session only; add --save to persist to config.yaml.)")
        return

    if sub in ("save", "recipes", "run"):
        _print(_RECIPES_REMOVED)
        return

    _print(f"  unknown /plan subcommand: {sub!r} — try: review, pause, lockstep (or /plan --help)")
