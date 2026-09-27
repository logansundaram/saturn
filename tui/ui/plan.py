"""
Plan rendering — the model's checklist (tools/planning.py) as the rail shows it. `render_plan`
prints a plan on demand (the pause prompt); `show_plan` re-renders the full plan — status glyph on
every row — each time it materially changes (the live trace's transparency surface). Both share
the `_plan_line` row format. (The plan-review editor left with the plan engine, 2026-09-27.)
"""

import time

from . import _base
from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _FAINT, _PLAN, _RAIL, _RAIL_GLYPH,
    _emit, _rail, _term_width, _truncate,
)


def _plan_line_bare(step: dict, *, show_tool: bool) -> "Text | str":
    """One plan row WITHOUT the trace-rail prefix — for hosts that draw their own gutter (the
    review frame's `┃`), where the railed variant would render a doubled gutter."""
    status = step.get("status", "pending")
    # An unknown status (a garbled/legacy record) renders as UNKNOWN — never guessed as pending
    # (transplanted from the visibility isolate: views over instrumentation, no defaults).
    glyph, style = _PLAN.get(status, ("?", "bold yellow"))
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")
    unknown = f"  ⟨unknown status: {status}⟩" if status not in _PLAN else ""

    # Width-responsive: drop the ::tool tag on a very narrow terminal, then size the label to what's
    # left after the prefix (rail+nest+glyph+id ≈ 12) and the tag, so a step stays on one row.
    tw = _term_width()
    tag = f"  ::{tool}" if (show_tool and tool and tw >= 56) else ""
    label = _truncate(str(step.get("label", "")), max(20, tw - 14 - len(tag) - len(unknown)))

    if _RICH:
        line = Text()
        line.append("  ", style=_RAIL)  # nest steps under the node / frame edge
        line.append(f"{glyph} ", style=style)
        line.append(f"{str(sid):>2}  ", style=_DIM)
        line.append(label, style=style if status in ("active", "skipped") else "default")
        if unknown:
            line.append(unknown, style="bold yellow")
        if tag:
            line.append(tag, style=_FAINT)  # the most incidental annotation — faintest
        return line
    return f"  {glyph} {str(sid):>2}  {label}{unknown}{tag}"


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


def _finished(plan) -> int:
    """How many steps have reached a terminal status. Reads the engine's own vocabulary
    (core.state.TERMINAL_STATUSES) rather than re-listing it here — a hand copy would drift the
    moment a status is added. Lazily imported so the TUI stays a leaf; any failure just costs
    the count, never the render."""
    try:
        from core.state import TERMINAL_STATUSES

        return sum(1 for s in plan if s.get("status") in TERMINAL_STATUSES)
    except Exception:
        return 0


def _plan_header(plan) -> None:
    """One dim `│ plan · 4/12` row above each full re-render. show_plan re-prints the whole plan
    on every material change — 8-12 times in a normal turn — and the rows were bare, interleaved
    with trace rows, with nothing marking where one rendering ended and the next began. The
    header is the delimiter, and the count doubles as the progress the repetition is there to
    show."""
    total = len(plan)
    label = f"plan · {_finished(plan)}/{total}"
    if _RICH:
        row = _rail()
        row.append(label, style=_DIM)
        _emit(row)
    else:
        _emit(f"  {_RAIL_GLYPH} {label}")


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
    _plan_header(plan)
    for step in plan:
        _emit(_plan_line(step, show_tool=True))
