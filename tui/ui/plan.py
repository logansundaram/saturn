"""
Plan rendering + the plan-review editor (the human-in-the-loop pause). `render_plan` prints a plan
on demand (`/plan`); `show_plan` re-renders the full plan — status glyph + intended tool on every
row — each time it materially changes (the live trace's transparency surface); `review_plan` runs
the interactive edit loop reached when a turn pauses at the plan_gate. All three share the
`_plan_line` row format.
"""

import time

from . import _base
from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _FAINT, _PLAN, _RAIL, _RAIL_GLYPH,
    _emit, _rail, _term_width, _truncate,
)
from .statusbar import _live_start, _live_stop


def _plan_line_bare(step: dict, *, show_tool: bool) -> "Text | str":
    """One plan row WITHOUT the trace-rail prefix — for hosts that draw their own gutter (the
    review frame's `┃`), where the railed variant would render a doubled gutter."""
    status = step.get("status", "pending")
    glyph, style = _PLAN.get(status, _PLAN["pending"])
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")

    # Width-responsive: drop the ::tool tag on a very narrow terminal, then size the label to what's
    # left after the prefix (rail+nest+glyph+id ≈ 12) and the tag, so a step stays on one row.
    tw = _term_width()
    tag = f"  ::{tool}" if (show_tool and tool and tw >= 56) else ""
    label = _truncate(str(step.get("label", "")), max(20, tw - 14 - len(tag)))

    if _RICH:
        line = Text()
        line.append("  ", style=_RAIL)  # nest steps under the node / frame edge
        line.append(f"{glyph} ", style=style)
        line.append(f"{str(sid):>2}  ", style=_DIM)
        line.append(label, style=style if status in ("active", "skipped") else "default")
        if tag:
            line.append(tag, style=_FAINT)  # the most incidental annotation — faintest
        return line
    return f"  {glyph} {str(sid):>2}  {label}{tag}"


def _plan_line(step: dict, *, show_tool: bool) -> "Text | str":
    bare = _plan_line_bare(step, show_tool=show_tool)
    if _RICH:
        line = _rail()
        line.append_text(bare)
        return line
    return f"  {_RAIL_GLYPH} {bare}"


def render_plan(plan) -> None:
    """Print the full plan unconditionally — every step, with its intended tool. Unlike
    `show_plan` this does no diffing and touches no per-turn state, so it's the right call for the
    `/plan` command (inspect the last plan on demand, outside the live trace)."""
    if not plan:
        _emit("  (no plan yet — run a turn first)")
        return
    for step in plan:
        _emit(_plan_line(step, show_tool=True))


def _fingerprint(plan) -> list[tuple]:
    """What the display keys on, positionally: (label, status, intended_tool) per step. Any
    change to any of the three — or to the step set itself — is a material plan change."""
    return [
        (str(step.get("label", "")), step.get("status", "pending"),
         step.get("intended_tool") or None)
        for step in plan
    ]


def show_plan(plan) -> None:
    """Render the live plan — the FULL step list, every row carrying its status glyph and
    intended tool — each time it materially changes (2026-07-06 faithful-rendering rework):
    the first draft, each completed step (the execute → update_plan loop), a replan's redraft,
    a rectify cancellation, a review edit. Re-rendering the whole block (instead of the old
    one-line status diff, which hid tools after the first print and missed a redraft that kept
    ids/statuses) keeps the transparency surface showing the plan AS IT CURRENTLY STANDS.

    The one fold: a step flipping to `active` with nothing else changed — the execute rail line
    + reasoning leaf in the same delta already name the step being worked, so that flip rides
    silently into the next material render (where it lands as its terminal status)."""
    if not plan:
        return

    fp = _fingerprint(plan)
    seen = _base._plan_seen or None  # {} = the reset marker (reset_turn / show_run) — first render
    if seen == fp:
        return
    if isinstance(seen, list) and len(seen) == len(fp):
        active_only = all(
            (old[0], old[2]) == (new[0], new[2]) and (old[1] == new[1] or new[1] == "active")
            for old, new in zip(seen, fp)
        )
        if active_only:
            _base._plan_seen = fp  # record it so the terminal render still diffs as a change
            return
    _base._plan_seen = fp
    for step in plan:
        _emit(_plan_line(step, show_tool=True))


# ── plan-review editor (the human-in-the-loop pause) ─────────────────────────────
# Reached when a turn pauses at the plan_gate (keyboard pause, /plan pause|review, or an in-graph
# request). Renders the live plan with the current step marked, then runs a small edit loop on the
# shared plan_ops grammar until the user continues (resume with the edited plan) or aborts (end the
# turn). Mirrors ask_approval's Live teardown/restart so it composes with the bottom status bar.
def _review_emit(text) -> None:
    _emit(text)


def _review_header(reason: str) -> None:
    if _RICH:
        _console.print()  # the pause lands mid-trace — let the frame read as its own moment
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("plan review", style=f"bold {_ACCENT}")
        top.append(" — execution paused", style=_DIM)
        _console.print(top)
        if reason:
            r = Text()
            r.append("  ┃ ", style="bold")
            r.append(reason, style=_DIM)
            _console.print(r)
    else:
        print()
        print("  ┏━ plan review — execution paused")
        if reason:
            print(f"  ┃ {reason}")


def _render_review_plan(plan: list[dict]) -> None:
    """List the plan inside the review block: every step with its status NAMED (the glyph alone
    doesn't tell the user what word to type at `status <id> <…>`) + intended tool, and the step
    execution will resume at flagged. The resume pointer is recomputed from the plan being
    rendered (first step with no recorded result — the engine's own execution pointer), so it
    tracks the user's edits instead of going stale the moment a step is added/dropped/moved."""
    if not plan:
        _review_emit("  ┃   (empty plan — add steps with `add <label>`)")
        return
    current = next((s.get("step_id") for s in plan if s.get("result") is None), None)
    for step in plan:
        # Bare rows: the review frame's `┃` IS the gutter — the railed variant would double it.
        line = _plan_line_bare(step, show_tool=True)
        status = step.get("status", "pending")
        tag = f"  [{status}]" if status != "pending" else ""
        marker = "  ← next to run" if step.get("step_id") == current else ""
        if _RICH:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append_text(line if isinstance(line, Text) else Text(str(line)))
            if tag:
                row.append(tag, style=_DIM)
            if marker:
                row.append(marker, style=f"bold {_ACCENT}")
            _console.print(row)
        else:
            print(f"  ┃ {line}{tag}{marker}")


def _review_hint() -> None:
    """The one-line standing hint under the plan — enough to act (run, stop, or start editing)
    without reprinting the whole editor grammar at every pause; `help` opens the full version."""
    if _RICH:
        t = Text()
        t.append("  ┃ ", style="bold")
        t.append("enter", style=_ACCENT)
        t.append(" runs · ", style=_DIM)
        t.append("abort", style=_ACCENT)
        t.append(" stops · edit with ", style=_DIM)
        t.append("add/edit/tool/status/move/drop", style="default")
        t.append(" · ", style=_DIM)
        t.append("help", style=_ACCENT)
        t.append(" for the grammar", style=_DIM)
        _console.print(t)
    else:
        print("  ┃ enter runs · abort stops · edit with add/edit/tool/status/move/drop"
              " · help for the grammar")


def _review_help() -> None:
    from core import plan_ops

    _review_emit("  ┃ edit the plan, then `go` to run it (or `abort` to stop):")
    for h in plan_ops.COMMAND_HELP:
        _review_emit(f"  ┃     {h}")
    _review_emit("  ┃     go / <enter>          run the (edited) plan")
    _review_emit("  ┃     abort / Ctrl-C        stop this turn")
    _review_emit("  ┃     show · help           reprint the plan · this help")


def _review_note(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ┃   ", style="bold")
        t.append(msg, style=_DIM if not msg.startswith("!") else "yellow")
        _console.print(t)
    else:
        print(f"  ┃   {msg}")


def _review_input() -> str:
    if _RICH:
        return _console.input("  [bold]┗━[/] edit» ", markup=True)
    return input("  ┗━ edit» ")


def review_plan(value: dict) -> dict:
    """Handle a plan-review interrupt. Returns `{"action": "continue"|"abort", "plan": <edited>}`
    — the resume value the plan_gate node applies. The edited plan is normalized + renumbered by
    plan_ops, so step ids the user typed always match what's rendered. Bare Enter (or `go`)
    continues; Ctrl-C/EOF aborts — an interrupted review must never run the plan."""
    from core import plan_ops

    plan = plan_ops.normalize(value.get("plan") or [])
    reason = value.get("reason", "")

    _live_stop()  # the editor blocks on input(); the bar can't be live while it does

    _review_header(reason)
    _render_review_plan(plan)
    _review_hint()

    action = "continue"
    while True:
        try:
            raw = _review_input()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C / a closed stdin at a control point must fail closed: abort the turn,
            # never run a plan the user was interrupting their review of.
            action = "abort"
            break
        cmd = raw.strip()
        low = cmd.lower()
        if low in ("", "go", "c", "continue", "run", "resume"):
            action = "continue"
            break
        if low in ("abort", "q", "quit", "cancel", "stop"):
            action = "abort"
            break
        if low in ("help", "h", "?"):
            _review_help()
            continue
        if low in ("show", "ls", "plan"):
            _render_review_plan(plan)
            continue
        try:
            plan, note = plan_ops.apply_command(plan, cmd)
            _review_note(note)
            _render_review_plan(plan)
        except ValueError as exc:
            _review_note(f"! {exc}")

    if _RICH:
        tail = Text()
        verb = "running the plan" if action == "continue" else "aborting the turn"
        tail.append("  ┗━ ", style="bold")
        tail.append(verb, style=_ACCENT)
        _console.print(tail)
    else:
        print(f"  ┗━ {'running the plan' if action == 'continue' else 'aborting the turn'}")

    _base._t_last = time.perf_counter()  # don't bill the human's edit time to the next node
    _live_start()  # the turn continues; re-pin the bar
    return {"action": action, "plan": plan}
