"""
Plan rendering + the plan-review editor (the human-in-the-loop pause). `render_plan` prints a plan
on demand (`/plan`); `show_plan` prints it once as the intended route then one line per status
change (the live trace's transparency surface); `review_plan` runs the interactive edit loop reached
when a turn pauses at the plan_gate. All three share the `_plan_line` row format.
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


def show_plan(plan) -> None:
    """First call this turn: print the whole plan as the intended route (with tools).
    Later calls: print only the steps whose status changed — one line each, like a trace.
    This keeps the plan transparent without re-rendering a panel on every node update."""
    if not plan:
        return

    first_render = not _base._plan_seen
    for step in plan:
        sid = step.get("step_id")
        status = step.get("status", "pending")
        if first_render:
            _emit(_plan_line(step, show_tool=True))
            _base._plan_seen[sid] = status
        elif _base._plan_seen.get(sid) != status:
            _emit(_plan_line(step, show_tool=False))
            _base._plan_seen[sid] = status


# ── plan-review editor (the human-in-the-loop pause) ─────────────────────────────
# Reached when a turn pauses at the plan_gate (keyboard pause, /plan pause|review, or an in-graph
# request). Renders the live plan with the current step marked, then runs a small edit loop on the
# shared plan_ops grammar until the user continues (resume with the edited plan) or aborts (end the
# turn). Mirrors ask_approval's Live teardown/restart so it composes with the bottom status bar.
def _review_emit(text) -> None:
    _emit(text)


def _review_header(reason: str) -> None:
    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("plan review", style=f"bold {_ACCENT}")
        top.append(" — execution paused", style=_DIM)
        top.append(" " + "━" * 22, style="bold")
        _console.print(top)
        if reason:
            r = Text()
            r.append("  ┃ ", style="bold")
            r.append(reason, style=_DIM)
            _console.print(r)
    else:
        print("  ┏━ plan review — execution paused " + "━" * 16)
        if reason:
            print(f"  ┃ {reason}")


def _render_review_plan(plan: list[dict], active_id) -> None:
    """List the plan inside the review block: every step with status + intended tool, the current
    step flagged so the user sees where execution will resume."""
    if not plan:
        _review_emit("  ┃   (empty plan — add steps with `add <label>`)")
        return
    for step in plan:
        # Bare rows: the review frame's `┃` IS the gutter — the railed variant would double it.
        line = _plan_line_bare(step, show_tool=True)
        marker = "  ← current" if step.get("step_id") == active_id else ""
        if _RICH:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append_text(line if isinstance(line, Text) else Text(str(line)))
            if marker:
                row.append(marker, style=f"bold {_ACCENT}")
            _console.print(row)
        else:
            print(f"  ┃ {line}{marker}")


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
    active = value.get("active_step") or {}
    active_id = active.get("step_id")

    _live_stop()  # the editor blocks on input(); the bar can't be live while it does

    _review_header(reason)
    _render_review_plan(plan, active_id)
    _review_help()

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
            _render_review_plan(plan, active_id)
            continue
        try:
            plan, note = plan_ops.apply_command(plan, cmd)
            _review_note(note)
            _render_review_plan(plan, active_id)
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
