import time
from state import AgentState

"""
update_plan node (repurposed from the old reflect node).

Runs after each tool round to keep the living plan's statuses in sync with reality, so the
plan stays a faithful, inspectable record of progress (and drives the live plan panel in
Phase 2).

This is intentionally MECHANICAL, not LLM-driven: the local model (gemma4:e4b) cannot reliably
emit structured JSON for plan revision — it leaks prose/markdown and fails to parse on nearly
every call, adding latency and errors for no benefit. Deterministic status advancement from the
tools actually called is both reliable and free. (An LLM-based reviser that can also INSERT
steps mid-loop is a post-MVP upgrade gated on a more capable model — see SATURDAY_MVP_PLAN.md §1.)
"""

_TERMINAL = ("done", "skipped")


def update_plan_node(state: AgentState):
    start = time.perf_counter()

    plan = state.get("plan", [])
    if not plan:
        return {}

    called = set(state.get("tools_called", []))

    # 1) Mark any not-yet-finished step whose intended tool has been called as done.
    marked = False
    for step in plan:
        if step["status"] in _TERMINAL:
            continue
        if step.get("intended_tool") and step["intended_tool"] in called:
            step["status"] = "done"
            marked = True

    # 2) If a tool round happened but matched no intended_tool (null/mismatched), still show
    #    progress by completing the earliest unfinished step.
    if called and not marked:
        for step in plan:
            if step["status"] not in _TERMINAL:
                step["status"] = "done"
                break

    # 3) Surface the next remaining step as active.
    for step in plan:
        if step["status"] == "pending":
            step["status"] = "active"
            break

    print(f"update_plan_node : {time.perf_counter() - start:.4f}s")
    return {"plan": plan}
