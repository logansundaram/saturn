from commands._framework import command, _print
from commands._utils import is_remove_verb, parse_toggle_status


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

To write your OWN plan up front and have the next request execute it, see /draft.

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
        if ctx.pending_plan:
            _print("  drafted plan (pending — your next message runs it; /draft edits):")
            ui.render_plan(ctx.pending_plan)
        mode = "on" if ctx.review_plan else "off"
        _print(f"  review mode: {mode}   (see /plan --help)")
        return

    sub = args[0].lower()

    if sub == "draft":
        # The draft composer moved to its own front door (2026-07-16, same day it shipped) —
        # the same moved-pointer contract _RENAMED gives cut top-level spellings.
        _print("  /plan draft moved — use /draft (see /draft --help)")
        return

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


@command(
    "draft",
    "Compose your OWN plan — your next message executes YOUR steps instead of the agent's draft.",
    usage="/draft [clear]",
    details="""
Co-planning: instead of editing the agent's plan mid-turn (plan review), you author the whole
plan up front and hand it to the engine. /draft opens the same step editor you get at plan
review, on an empty plan (or your pending draft, to keep editing). Save it, then type your
request — the next turn executes YOUR steps instead of drafting its own. The engine still
reflects after each step, so a failing step can be repaired, and every gated call still asks.

  /draft            open the editor (on the pending draft, if one is waiting)
  /draft clear      discard the pending draft (also: remove/rm/delete/…)

Tool names are normalized (calc → calculate); an unrecognized tool is kept as written and fails
closed at execution rather than silently degrading into an ungrounded reasoning step.

One draft = one turn: the next message consumes it. /plan shows a pending draft alongside the
live plan. Editor verbs:
  add <label> [::tool] · edit <id> <label> · tool <id> <name|none>
  status <id> <status> · move <id> <pos>   · drop <id>
  go / <enter> saves the draft, abort discards the edits.
""",
)
def _draft(ctx, args):
    """`/draft` — compose a plan BY HAND and hand it to the engine (the co-planning counterpart
    of plan review: instead of editing the agent's draft mid-turn, the user authors the whole
    plan up front and the next request executes it; promoted from `/plan draft` to its own
    command 2026-07-16 — a headline surface gets a front door). The steps are composed in the
    SAME editor + grammar as the mid-turn review (ui.review_plan over plan_ops — one grammar,
    two moments), stored on ctx.pending_plan, and seeded into the next turn's state by the REPL;
    plan_node honors a pre-seeded plan verbatim and skips LLM drafting. Everything downstream is
    unchanged — per-step reflection, the approval gate, rectify's guards — so a hand-written plan
    gets exactly the engine's safety envelope, just not its planner."""
    from tui import ui

    if args and (is_remove_verb(args[0]) or args[0].lower() in ("clear", "cancel")):
        if ctx.pending_plan:
            ctx.pending_plan = None
            _print("  draft discarded.")
        else:
            _print("  no pending draft.")
        return
    if args:
        _print(f"  unknown /draft argument: {args[0]!r} — bare `/draft` edits, `clear` discards.")
        return

    decision = ui.review_plan(
        {
            "plan": ctx.pending_plan or [],
            "reason": "compose the steps yourself — your next message becomes the request they run for",
            "title": "plan draft",
            "subtitle": "nothing is executing",
            "enter_verb": "saves the draft",
            "abort_verb": "discards edits",
            "verbs": ("draft saved", "draft unchanged"),
        }
    )
    if not isinstance(decision, dict) or decision.get("action") != "continue":
        # Abort keeps whatever was pending before — an interrupted edit never destroys a draft.
        return

    plan, notes = _normalize_draft(decision.get("plan") or [])
    for note in notes:
        _print(f"  {note}")
    ctx.pending_plan = plan or None
    if not plan:
        _print("  empty draft — nothing pending.")
        return
    _print(f"  draft saved ({len(plan)} step(s)) — type your request to run it; /plan shows it,")
    _print("  /draft re-opens the editor, /draft clear discards it.")


def _normalize_draft(plan: list) -> tuple[list, list[str]]:
    """Map each drafted step's tool spelling onto the live registry, mirroring the planner path's
    rules (structured.to_steps): synonyms normalize (calc → calculate), a no-tool marker means a
    genuine reasoning step, and an UNRESOLVABLE spelling is kept raw — execute fails closed on it
    (an error incident) instead of silently answering the step from the model's priors. Returns
    the normalized plan plus human-readable notes for anything that changed or will fail."""
    from core.structured import _NO_TOOL_MARKERS, norm_tool  # the one tool-spelling authority

    out, notes = [], []
    for step in plan:
        step = dict(step)
        raw = str(step.get("intended_tool") or "").strip()
        if raw:
            tool = norm_tool(raw)
            if tool is not None:
                if tool != raw:
                    notes.append(f"step {step.get('step_id')}: tool '{raw}' → {tool}")
                step["intended_tool"] = tool
            elif raw.lower() in _NO_TOOL_MARKERS:
                step["intended_tool"] = None  # a spelled-out "reasoning" step
            else:
                notes.append(
                    f"step {step.get('step_id')}: '{raw}' is not a registered tool — the step "
                    "will fail closed at execution (fix it with /draft, or /tools lists names)"
                )
        out.append(step)
    return out, notes
