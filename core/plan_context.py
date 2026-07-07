"""
Plan-as-data-bus context builders (transplanted from the agentic_benchmark harness, 2026-07-03).

The engine's LLM calls never see the raw message history — each one gets a CURATED context built
from the plan's own step results: the user's request, a compact results block, an explicit
"the previous step" callout, and the current step. This is what keeps small local models on
task: the context is exactly the data the step needs, in a stable shape, with nothing stale.

All helpers are pure over the plain step dicts ({step_id, label, status, intended_tool, result,
needs_resolution} — see core/state.py) so they are trivially testable offline.
"""

from __future__ import annotations

from config import get_config
from textutil import head_tail

# Cap each earlier result inside the results block. The full observation still lives on the step
# (and in tool_results for synthesize's numbered sections); this bound keeps a per-step context
# from re-sending every prior read in full.
_RESULT_CAP = 800

# The "previous step" callout carries more of its result than the block (it is the referent of
# "the previous step's result" in step labels), but still bounded — a ~12k clamped observation
# must not ride every per-step prompt in full.
_CALLOUT_CAP = 4000

# THE engine-wide tool classifications (one home — execute's write gate, rectify's resolution
# exemption, and synthesize's write verification all key off these; a copy per node is how a
# new tool silently escapes one of the three).
WRITE_TOOLS = ("write_file", "edit_file")
SEARCH_TOOLS = {"search_knowledge_base", "search_files", "find_files", "web_search"}


def original_request(state) -> str:
    """The user's request as the engine's prompts see it — the current turn's query."""
    return str(state.get("current_query") or "")


def vetoes_block(state) -> str:
    """The user's plan-review vetoes (state["plan_vetoes"], written by plan_gate) as a prompt
    block — '' when there are none. THE one framing both the rectify judge and the replanner
    receive: work the human explicitly removed at the plan-review editor is deliberately out of
    scope, and the plan must never be changed or extended to reinstate it — the human's edit
    outranks the engine's self-correction (the gate's guarded-outcome principle, applied to
    review edits)."""
    vetoes = [str(v).strip() for v in (state.get("plan_vetoes") or []) if str(v).strip()]
    if not vetoes:
        return ""
    return (
        "The user EDITED the plan at the plan-review prompt and REMOVED these steps — they are "
        "deliberately out of scope for this turn at the user's own request. Do NOT change or "
        "extend the plan to reinstate them (or equivalent work), and do not treat their absence "
        "as a gap in the plan:\n" + "\n".join(f"- {v}" for v in vetoes)
    )


def clean(text) -> str:
    """Normalize an observation before it lands on a step: absolute workspace paths (run_shell
    output routinely embeds them) collapse to workspace-relative so prompts and the rendered
    plan stay readable and machine-independent. Best-effort; unknown shapes pass through."""
    s = str(text)
    try:
        raw = str(get_config().path("workspace"))
    except Exception:
        return s
    for form in {raw, raw.replace("\\", "/")}:
        if form:
            s = s.replace(form + "/", "").replace(form + "\\", "").replace(form, "workspace")
    return s


def results_block(plan) -> str:
    """The 'Results from earlier steps' block: every completed step's label -> result (capped),
    numbered in plan order. Empty string when nothing has run."""
    done = [s for s in plan or [] if s.get("result") is not None]
    if not done:
        return ""
    lines = ["Results from earlier steps (use these exact values):"]
    for i, s in enumerate(done, 1):
        r = str(s.get("result") or "").strip()
        if len(r) > _RESULT_CAP:
            r = r[:_RESULT_CAP] + " …(truncated)"
        lines.append(f"{i}. {s.get('label')} -> {r}")
    return "\n".join(lines)


def exec_context(state, step) -> str:
    """The curated context for executing ONE step: request + earlier results + an explicit
    'the previous step' callout (the referent of 'the previous step's result' in step labels)
    + the current step. The grounding context rides along so workspace manifests / attachments /
    memory stay visible without the raw history."""
    parts = [f"User's overall request: {original_request(state)}"]
    grounding = str(state.get("context") or "").strip()
    if grounding:
        parts.append(grounding)
    plan = state.get("plan") or []
    prior = [s for s in plan if s is not step and s.get("result") is not None]
    block = results_block([s for s in plan if s is not step])
    if block:
        parts.append(block)
    if prior:
        last = prior[-1]
        parts.append(
            f'The immediately preceding step ("the previous step") was: '
            f"{last.get('label')}\n  its result: "
            f"{head_tail(str(last.get('result') or '').strip(), _CALLOUT_CAP)}"
        )
    parts.append(f"Your current step: {step.get('label')}")
    return "\n\n".join(parts)


def plan_txt(plan) -> str:
    """The whole plan as text for the rectify/replan prompts: DONE steps with their results
    (capped like the results block — several ~12k clamped observations would otherwise overflow
    a small model's window and front-truncate the very system prompt the call depends on),
    PENDING steps with their intended tool."""
    lines = []
    for i, s in enumerate(plan or [], 1):
        tool = s.get("intended_tool") or "none"
        if s.get("result") is None:
            lines.append(f"{i}. [PENDING] tool={tool} | {s.get('label')}")
        else:
            r = str(s.get("result") or "").strip()
            if len(r) > _RESULT_CAP:
                r = r[:_RESULT_CAP] + " …(truncated)"
            lines.append(
                f"{i}. [DONE] tool={tool} | {s.get('label')}\n   result: {r}"
            )
    return "\n".join(lines)
