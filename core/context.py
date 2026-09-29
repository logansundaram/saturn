"""The grounding halves and the observation normalizer — the two helpers every node shared out
of core/plan_context.py, kept after the plan engine's removal (2026-09-27)."""

from __future__ import annotations


# The filesystem write tools — the one classification the answer trailers and memory review
# still key on (which calls wrote files).
WRITE_TOOLS = ("write_file", "edit_file")


def grounding_parts(state) -> "tuple[str, str]":
    """The grounding context as (stable, per-turn) halves — the grounding node's split
    (`context_stable` / `context_dynamic`). A state carrying only the joined `context` (an
    older checkpoint, a test fixture) is all-stable."""
    stable = state.get("context_stable")
    if stable is None and state.get("context_dynamic") is None:
        return str(state.get("context") or "").strip(), ""
    return str(stable or "").strip(), str(state.get("context_dynamic") or "").strip()


def clean(text) -> str:
    """Normalize an observation: absolute paths under the working folder (run_shell output
    routinely embeds them) collapse to relative ones so prompts and the rail stay readable and
    machine-independent. Best-effort; unknown shapes pass through."""
    s = str(text)
    try:
        from core import workspace

        raw = str(workspace.root())
    except Exception:
        return s
    for form in {raw, raw.replace("\\", "/")}:
        if form:
            s = s.replace(form + "/", "").replace(form + "\\", "").replace(form, ".")
    return s
