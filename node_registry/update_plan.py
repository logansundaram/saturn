import time
from state import AgentState

"""
update_plan node (formerly reflect.py).

Runs after each tool round to keep the living plan's statuses in sync with reality, so the
plan stays a faithful, inspectable record of progress (and drives the live plan panel in
Phase 2).

This is intentionally MECHANICAL, not LLM-driven: the local model (gemma4:e4b) cannot reliably
emit structured JSON for plan revision — it leaks prose/markdown and fails to parse on nearly
every call, adding latency and errors for no benefit. Deterministic status advancement from the
tools actually called is both reliable and free.

The LLM-driven counterpart that can also INSERT a step mid-loop lives in `node_registry/replan.py`
(the `judge` role): it runs only at the agent's apparently-complete finish to escalate an
ungrounded answer to a web_search. Keeping the per-tool-round status bookkeeping here mechanical
and the (rarer, more expensive) judgment call there keeps the common path fast and reliable.
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
