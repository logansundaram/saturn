"""
Plan rendering — the model's checklist (tools/planning.py) as the rail shows it. `render_plan`
prints a plan on demand (the pause prompt); `show_plan` re-renders the full plan — status glyph on
every row — each time it materially changes (the live trace's transparency surface). Both share
the `_plan_line` row format.
"""

import time

from . import _base
from ._base import (
    _DIM, _FAINT, _PLAN, _RAIL,
    _emit, _rail, _term_width, _truncate,
)


def _plan_line(step: dict, *, show_tool: bool) -> "Text":
    """One `│ ▸ 2  label  ::tool` plan row on the trace rail."""
    status = step.get("status", "pending")
    # An unknown status (a garbled/legacy record) renders as UNKNOWN — never guessed as pending.
    glyph, style = _PLAN.get(status, ("?", "bold yellow"))
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")
    unknown = f"  ⟨unknown status: {status}⟩" if status not in _PLAN else ""

    # Width-responsive: drop the ::tool tag on a very narrow terminal, then size the label to what's
    # left after the prefix (rail+nest+glyph+id ≈ 12) and the tag, so a step stays on one row.
    tw = _term_width()
    tag = f"  ::{tool}" if (show_tool and tool and tw >= 56) else ""
    label = _truncate(str(step.get("label", "")), max(20, tw - 14 - len(tag) - len(unknown)))

    line = _rail()
    line.append("  ", style=_RAIL)  # nest steps under the node
    line.append(f"{glyph} ", style=style)
    line.append(f"{str(sid):>2}  ", style=_DIM)
    line.append(label, style=style if status in ("active", "skipped") else "default")
    if unknown:
        line.append(unknown, style="bold yellow")
    if tag:
        line.append(tag, style=_FAINT)  # the most incidental annotation — faintest
    return line


def render_plan(plan) -> None:
    """Print the full plan unconditionally — every step, with its intended tool. Unlike
    `show_plan` this does no diffing and touches no per-turn state (the pause prompt's view)."""
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
    """One dim `│ plan · 4/12` row above each full re-render: show_plan re-prints the whole plan
    on every material change, interleaved with trace rows, so the header delimits one rendering
    from the next — and the count is the progress the repetition is there to show."""
    row = _rail()
    row.append(f"plan · {_finished(plan)}/{len(plan)}", style=_DIM)
    _emit(row)


def show_plan(plan) -> None:
    """Render the live plan — the FULL step list, every row carrying its status glyph and
    intended tool — each time it materially changes, i.e. on each `plan` call the model makes
    (tools/planning.py), so the transparency surface shows the plan AS IT CURRENTLY STANDS.

    The one fold: a step flipping to `active` with nothing else changed rides silently into the
    next material render (where it lands as its terminal status)."""
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
