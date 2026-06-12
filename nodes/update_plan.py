import time
import diag
from collections import Counter
from core.state import AgentState, TERMINAL_STATUSES

"""
update_plan node (formerly reflect.py).

Runs after each tool round to keep the living plan's statuses in sync with reality, so the
plan stays a faithful, inspectable record of progress (and drives the live plan panel in
Phase 2).

This is intentionally MECHANICAL, not LLM-driven: the local model (gemma4:e4b) cannot reliably
emit structured JSON for plan revision — it leaks prose/markdown and fails to parse on nearly
every call, adding latency and errors for no benefit. Deterministic status advancement from the
tools actually called is both reliable and free.

The LLM-driven counterpart that can also INSERT a step mid-loop lives in `nodes/replan.py`
(the `judge` role): it runs only at the agent's apparently-complete finish to escalate an
ungrounded answer to a web_search. Keeping the per-tool-round status bookkeeping here mechanical
and the (rarer, more expensive) judgment call there keeps the common path fast and reliable.
"""

def update_plan_node(state: AgentState):
    start = time.perf_counter()

    plan = state.get("plan", [])
    if not plan:
        return {}

    # Work on a copy — never mutate the state object in place. Returning a fresh list keeps the
    # checkpointer's snapshot diffs honest and matches agent_node, which also copies before it
    # advances a step.
    plan = [dict(s) for s in plan]
    called = state.get("tools_called", [])

    # 1) Positional credit: consume each called tool against the steps that expect it, IN ORDER,
    #    as a multiset — so two steps sharing a tool (e.g. two web_search steps) each need their
    #    own call rather than both completing off the first one. (Set membership would mark both
    #    done on the first call and collapse the plan; this mirrors state.unrun_planned_tools so
    #    the two views of "what's run" never disagree.) Already-done steps consume their tool first
    #    so a later same-tool step lines up against the correct remaining call.
    remaining = Counter(called)
    newly_marked = False
    for step in plan:
        if step["status"] == "skipped":
            continue
        tool = step.get("intended_tool")
        if step["status"] == "done":
            if tool and remaining.get(tool, 0) > 0:
                remaining[tool] -= 1
            continue
        if tool and remaining.get(tool, 0) > 0:
            remaining[tool] -= 1
            step["status"] = "done"
            newly_marked = True

    # 2) Progress fallback: a tool round happened but credited no planned step (the planner's
    #    intended_tool guess didn't match the tool the agent actually used). Advance the current
    #    step so the plan still reflects movement. In lockstep the earliest non-terminal step is
    #    exactly the one being worked, so this completes the right step.
    if called and not newly_marked:
        for step in plan:
            if step["status"] not in TERMINAL_STATUSES:
                # Only advance a step that actually expected a tool (the intended_tool-mismatch
                # case this fallback is for). A no-intended_tool step — notably the generic
                # single-step fallback plan — must NOT be marked done off the first tool round, or
                # the plan reports complete mid-task and the gathering-floor nudge switches off. A
                # genuine no-tool step is retired by agent_node's lockstep advance when it finishes.
                if step.get("intended_tool"):
                    step["status"] = "done"
                break

    # 3) Surface the next remaining step as active.
    for step in plan:
        if step["status"] == "pending":
            step["status"] = "active"
            break

    diag.log(f"update_plan_node : {time.perf_counter() - start:.4f}s")
    return {"plan": plan}
