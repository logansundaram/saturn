"""
Plan-editing operations + a tiny command grammar, shared by every surface that mutates a living
plan: the interactive plan-review editor (`ui.review_plan`, reached when a turn pauses at the
`plan_gate`) and the `/plan` command. Keeping the verbs in one place means the two surfaces can't
drift — the prompt you get mid-turn understands exactly the same `add`/`edit`/`drop`/`move`/
`tool`/`status` words you'd document for `/plan`.

The plan is the same plain-dict shape used everywhere else in state:
`{step_id, label, status, intended_tool}` (see state.py). These functions
are pure: they take a plan list and return a NEW edited list (the caller decides whether to commit
it), so they're trivially testable and never mutate the live plan in place by surprise.

`step_id` is treated as a 1-based position and renumbered after any structural change, so the ids a
user sees in the rendered plan always match what they type. Editing a step preserves its `status`
(so a `done` step the user merely relabels stays done and the loop continues from where it was);
newly added steps start `pending`. `intended_tool` values are validated against the registered
tools — an unknown tool is allowed (it just won't auto-advance the mechanical `update_plan`) but
the caller is told, so typos surface.
"""

from __future__ import annotations

from typing import Optional

_VALID_STATUS = ("pending", "active", "done", "skipped")


def normalize(plan: Optional[list[dict]]) -> list[dict]:
    """Return a clean copy of the plan with every field present and `step_id`s renumbered 1..N."""
    out: list[dict] = []
    for i, step in enumerate(plan or [], start=1):
        status = step.get("status")
        out.append(
            {
                "step_id": i,
                "label": str(step.get("label", "")).strip() or f"Step {i}",
                "status": status if status in _VALID_STATUS else "pending",
                "intended_tool": step.get("intended_tool") or None,
            }
        )
    return out


def _renumber(plan: list[dict]) -> list[dict]:
    for i, step in enumerate(plan, start=1):
        step["step_id"] = i
    return plan


def _index_of(plan: list[dict], step_id: int) -> int:
    for i, step in enumerate(plan):
        if step.get("step_id") == step_id:
            return i
    raise ValueError(f"no step #{step_id} (plan has {len(plan)} step(s))")


def known_tools() -> set[str]:
    """Registered tool names, for validating `intended_tool`. Imported lazily so this module has
    no import-time dependency on the registry (which pulls in the tool implementations)."""
    try:
        from tools.registry import tools_by_name

        return set(tools_by_name)
    except Exception:
        return set()


def add_step(
    plan: list[dict], label: str, intended_tool: Optional[str] = None, at: Optional[int] = None
) -> list[dict]:
    """Insert a new pending step. `at` is a 1-based position (default: append)."""
    plan = [dict(s) for s in plan]
    step = {"step_id": 0, "label": label.strip(), "status": "pending",
            "intended_tool": intended_tool or None}
    if at is None or at > len(plan):
        plan.append(step)
    else:
        plan.insert(max(0, at - 1), step)
    return _renumber(plan)


def edit_step(plan: list[dict], step_id: int, label: str) -> list[dict]:
    """Relabel a step, preserving its status and intended tool."""
    plan = [dict(s) for s in plan]
    plan[_index_of(plan, step_id)]["label"] = label.strip()
    return plan


def set_tool(plan: list[dict], step_id: int, tool: Optional[str]) -> list[dict]:
    """Set (or clear, with tool=None) a step's intended tool."""
    plan = [dict(s) for s in plan]
    plan[_index_of(plan, step_id)]["intended_tool"] = tool or None
    return plan


def set_status(plan: list[dict], step_id: int, status: str) -> list[dict]:
    if status not in _VALID_STATUS:
        raise ValueError(f"status must be one of {', '.join(_VALID_STATUS)}")
    plan = [dict(s) for s in plan]
    plan[_index_of(plan, step_id)]["status"] = status
    return plan


def drop_step(plan: list[dict], step_id: int) -> list[dict]:
    plan = [dict(s) for s in plan]
    del plan[_index_of(plan, step_id)]
    return _renumber(plan)


def move_step(plan: list[dict], step_id: int, to_pos: int) -> list[dict]:
    """Move a step to a new 1-based position, shifting the rest."""
    plan = [dict(s) for s in plan]
    i = _index_of(plan, step_id)
    step = plan.pop(i)
    to_pos = max(1, min(to_pos, len(plan) + 1))
    plan.insert(to_pos - 1, step)
    return _renumber(plan)


# Verbs accepted by apply_command, surfaced in the editor's `help`.
COMMAND_HELP = (
    "add <label> [::tool]    append a step (optionally with an intended tool)",
    "edit <id> <label>       relabel step #id",
    "tool <id> <name|none>   set or clear step #id's intended tool",
    "status <id> <status>    set step status (pending|active|done|skipped)",
    "move <id> <pos>         move step #id to position pos",
    "drop <id>               remove step #id",
)


def _parse_id(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f"expected a step number, got {token!r}")


def apply_command(plan: list[dict], line: str) -> tuple[list[dict], str]:
    """Parse one edit line against the shared grammar and apply it, returning `(new_plan, note)`.

    Raises `ValueError` (with a readable message) on a malformed command or an out-of-range step —
    callers report it and keep the prompt open rather than crashing. The label/tool grammar lets a
    step carry an intended tool via a trailing `::tool` on `add`."""
    parts = line.split()
    if not parts:
        raise ValueError("empty command")
    verb = parts[0].lower()
    rest = parts[1:]

    if verb == "add":
        if not rest:
            raise ValueError("usage: add <label> [::tool]")
        text = " ".join(rest)
        tool = None
        if "::" in text:
            text, _, tool_part = text.rpartition("::")
            tool = tool_part.strip() or None
        label = text.strip()
        if not label:
            raise ValueError("a step needs a label")
        note = _tool_note(tool)
        return add_step(plan, label, tool), f"added: {label}" + note

    if verb == "edit":
        if len(rest) < 2:
            raise ValueError("usage: edit <id> <label>")
        sid = _parse_id(rest[0])
        return edit_step(plan, sid, " ".join(rest[1:])), f"edited step #{sid}"

    if verb == "tool":
        if len(rest) < 2:
            raise ValueError("usage: tool <id> <name|none>")
        sid = _parse_id(rest[0])
        raw = rest[1]
        tool = None if raw.lower() in ("none", "null", "-", "clear") else raw
        note = _tool_note(tool)
        return set_tool(plan, sid, tool), f"step #{sid} tool -> {tool or 'none'}" + note

    if verb == "status":
        if len(rest) < 2:
            raise ValueError("usage: status <id> <pending|active|done|skipped>")
        sid = _parse_id(rest[0])
        return set_status(plan, sid, rest[1].lower()), f"step #{sid} status -> {rest[1].lower()}"

    if verb == "move":
        if len(rest) < 2:
            raise ValueError("usage: move <id> <pos>")
        sid = _parse_id(rest[0])
        pos = _parse_id(rest[1])
        return move_step(plan, sid, pos), f"moved step #{sid} -> position {pos}"

    if verb == "drop":
        if not rest:
            raise ValueError("usage: drop <id>")
        sid = _parse_id(rest[0])
        return drop_step(plan, sid), f"dropped step #{sid}"

    raise ValueError(f"unknown edit verb {verb!r} — try: add, edit, tool, status, move, drop")


def _tool_note(tool: Optional[str]) -> str:
    """A trailing warning if a named tool isn't registered (allowed, but it won't auto-advance)."""
    if tool and tool not in known_tools():
        return f"  (note: '{tool}' is not a registered tool — it won't auto-advance the plan)"
    return ""
