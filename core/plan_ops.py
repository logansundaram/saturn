"""
Plan-editing operations + a tiny command grammar, shared by every surface that mutates a living
plan: the interactive plan-review editor (`ui.review_plan`, reached when a turn pauses at the
`plan_gate`) and the `/plan` command. Keeping the verbs in one place means the two surfaces can't
drift — the prompt you get mid-turn understands exactly the same `add`/`edit`/`drop`/`move`/
`tool`/`status` words you'd document for `/plan`.

The plan is the same plain-dict shape used everywhere else in state:
`{step_id, label, status, intended_tool, result, needs_resolution}` (see state.py — the plan is
the engine's data bus; `result` is None until a step runs). These functions are pure: they take a
plan list and return a NEW edited list (the caller decides whether to commit it), so they're
trivially testable and never mutate the live plan in place by surprise. Editing a completed
step's label keeps its recorded outcome; the ONE editor verb that touches `result` is
`set_status`, which keeps the status/result pairing intact (see its docstring).

`step_id` is treated as a 1-based position and renumbered after any structural change, so the ids a
user sees in the rendered plan always match what they type. Editing a step preserves its `status`
(so a `done` step the user merely relabels stays done and the loop continues from where it was);
newly added steps start `pending`. `intended_tool` values are validated against the registered
tools — an unknown tool is allowed (it just won't auto-advance the mechanical `update_plan`) but
the caller is told, so typos surface.

This module is the whole plan-review seam: alongside the editor verbs it owns the
`PauseController` (bottom of the file) — the latch that gets execution TO the review prompt in
the first place. (It lived in core/interrupts.py until the 2026-06-11 leaf consolidation.)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from core.state import TERMINAL_STATUSES

# The one status vocabulary, derived from core/state.py's declaration — a hand-copied tuple here
# would silently launder any status added there back to "pending" on the next review edit.
_VALID_STATUS = ("pending", "active") + tuple(TERMINAL_STATUSES)


def retarget_knowledge_base_reads(steps: list[dict]) -> list[dict]:
    """The namespace guard (2026-09-02): a `read_file` step whose label names an INGESTED
    document that has no workspace file of that name is retargeted to `search_knowledge_base`,
    the one tool that can read it. The engine knows both namespaces exactly (the two manifests);
    a small planner reads "read X" as read_file even when the context lists X under the
    knowledge base (measured twice against qwen3.5:9b, prompt rule and all). Deterministic and
    safe: both tools are read_only, the label is untouched, the swap is logged. The match is
    the document's full manifest name (or its basename) appearing in the label,
    case-insensitively — never a stem, so 'notes' does not claim 'notes.md'. In place on the
    given dicts; returns the same list for chaining."""
    reads = [s for s in steps or [] if s.get("intended_tool") == "read_file"]
    if not reads:
        return steps
    try:
        from pathlib import PurePath

        from config import get_config
        from stores.document_registry import manifest_entries, read_documents_manifest

        names = [e["name"] for e in manifest_entries(read_documents_manifest()) if e.get("name")]
        workspace = get_config().path("workspace")
    except Exception:  # a manifest problem must never fail planning
        return steps
    if not names:
        return steps
    for s in reads:
        label = str(s.get("label") or "").lower()
        for name in names:
            base = PurePath(name).name
            if name.lower() in label or base.lower() in label:
                if (workspace / name).is_file() or (workspace / base).is_file():
                    break  # a same-named workspace file: read_file is right
                s["intended_tool"] = "search_knowledge_base"
                import diag

                diag.log(f"plan: retargeted read_file -> search_knowledge_base for "
                         f"knowledge-base document {name!r}")
                break
    return steps


def normalize(plan: Optional[list[dict]]) -> list[dict]:
    """Return a clean copy of the plan with every field present and `step_id`s renumbered 1..N.
    `result` survives untouched (it is the data bus — a completed step's recorded outcome), and
    a step whose status fell outside the vocabulary resets to pending WITHOUT clearing it."""
    out: list[dict] = []
    for i, step in enumerate(plan or [], start=1):
        status = step.get("status")
        out.append(
            {
                "step_id": i,
                "label": str(step.get("label", "")).strip() or f"Step {i}",
                "status": status if status in _VALID_STATUS else "pending",
                "intended_tool": step.get("intended_tool") or None,
                "result": step.get("result"),
                "needs_resolution": bool(step.get("needs_resolution")),
            }
        )
        # `origin` (replan's stamp — the effect-authorization rule keys on it) survives a review
        # edit: a human's edit never launders a redrafted step into an up-front one.
        if step.get("origin"):
            out[-1]["origin"] = step.get("origin")
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


def resolve_tool(raw: "Optional[str]") -> "tuple[Optional[str], str]":
    """Map a tool spelling typed in the plan editor onto the live registry, returning
    (tool, note). THE one tool-spelling authority is `structured.norm_tool` — the same one the
    planner path and `/draft` use — so a synonym the user learned in one surface (`calc`) means
    the same thing in the other. This editor used to test bare membership instead, which accepted
    `calc` with a warning and then failed the step closed at execute.

    Parity means BOTH halves of the rule: a no-tool marker (`none`, `reasoning`, `answer`, … —
    structured._NO_TOOL_MARKERS, the same set /draft consults) means a genuine reasoning step
    (intended_tool None), and an unresolvable spelling that is NOT a marker is KEPT RAW (never
    silently blanked): execute fails closed on it as an error incident rather than answering the
    step from the model's priors. Imported lazily so this module keeps no import-time dependency
    on the registry."""
    raw = (raw or "").strip() or None
    if raw is None:
        return None, ""
    try:
        from core.structured import _NO_TOOL_MARKERS, norm_tool
    except Exception:
        return raw, ""
    tool = norm_tool(raw)
    if tool is None:
        if raw.lower() in _NO_TOOL_MARKERS:
            return None, ""  # a spelled-out "no tool" — a genuine reasoning step, like /draft
        return raw, f"  (note: '{raw}' is not a registered tool — the step will fail closed)"
    if tool != raw:
        return tool, f"  (tool '{raw}' → {tool})"
    return tool, ""


def add_step(
    plan: list[dict], label: str, intended_tool: Optional[str] = None, at: Optional[int] = None
) -> list[dict]:
    """Insert a new pending step. `at` is a 1-based position (default: append)."""
    plan = [dict(s) for s in plan]
    step = {"step_id": 0, "label": label.strip(), "status": "pending",
            "intended_tool": intended_tool or None, "result": None,
            "needs_resolution": False}
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


# The review-retirement stamp: `set_status` writes it when the USER retires an un-run step at
# plan review, and `is_review_retirement` reads it back — one producer + one parser (the
# DECLINE_TEXT pattern), so `rectify` can tell a user's single-step veto apart from a gate /
# write-gate rejection: the former continues past the step, the latter cancels the run.
_REVIEW_STAMP_SUFFIX = "at plan review — the step did not run"


def retirement_text(status: str, reason: str = "") -> str:
    """THE result text for a step the USER retired — one producer for both ways it happens: the
    review editor's status verb (`set_status`/`retire_step`) and `nodes/execute`'s revocation
    refusal (the user removed the effect at review and a later redraft tried to perform it
    anyway). Both end with `_REVIEW_STAMP_SUFFIX`, which is what tells rectify this is a
    SINGLE-STEP veto — skip this one, continue the rest — rather than a guard rejection ending
    the run."""
    detail = f": {reason} —" if reason else ""
    return f"marked {status}{detail} {_REVIEW_STAMP_SUFFIX}"


def review_stamp(status: str) -> str:
    """The result text stamped onto a step the user retired at the plan-review editor."""
    return retirement_text(status)


def retire_step(step: dict, status: str = "skipped") -> dict:
    """A COPY of `step` retired at the user's request (status + the review stamp as result)."""
    out = dict(step)
    out["status"] = status
    out["result"] = retirement_text(status)
    return out


def is_review_retirement(step) -> bool:
    """True when this step was retired BY THE USER at plan review (set_status's stamp) rather
    than by an executed guard. Consumed by rectify's guarded branch and plan_gate's veto
    detection; tolerates garbage (absent-as-no)."""
    return isinstance(step, dict) and str(step.get("result") or "").endswith(_REVIEW_STAMP_SUFFIX)


def set_status(plan: list[dict], step_id: int, status: str) -> list[dict]:
    """Set a step's status, keeping the status/result pairing intact (gotcha #6): the execution
    pointer is `result is None`, so a TERMINAL status on an un-run step must also stamp a result
    — without one the engine re-selects the step, flips it back to `active`, and RUNS it,
    silently discarding the user's edit. Symmetrically, `pending`/`active` clears the result so
    a retired step becomes runnable again."""
    if status not in _VALID_STATUS:
        raise ValueError(f"status must be one of {', '.join(_VALID_STATUS)}")
    plan = [dict(s) for s in plan]
    step = plan[_index_of(plan, step_id)]
    step["status"] = status
    if status in TERMINAL_STATUSES:
        if step.get("result") is None:
            step["result"] = review_stamp(status)
    else:
        step["result"] = None
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
    "status <id> <status>    set step status (pending|active|done|skipped|blocked|error|cancelled)",
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
        tool, note = resolve_tool(tool)
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
        tool, note = resolve_tool(tool)
        return set_tool(plan, sid, tool), f"step #{sid} tool -> {tool or 'none'}" + note

    if verb == "status":
        if len(rest) < 2:
            raise ValueError("usage: status <id> <pending|active|done|skipped|blocked|error|cancelled>")
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


# The pause latch moved to core/pause.py (2026-09-27); re-exported here until this module goes.
from core.pause import PauseRequest, PauseController, get_pause_controller  # noqa: E402,F401
