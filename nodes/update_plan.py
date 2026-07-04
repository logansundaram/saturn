"""
update_plan node — the mechanical recorder (the 2026-07-03 engine transplant).

Runs after each tool round (and after a fully-rejected approval batch) and does ONE thing: write
the observation onto the current plan step — the plan IS the data bus, so this is the moment a
step's result becomes available to every later step's curated context, to rectify, and to
synthesize.

Intentionally MECHANICAL, no LLM: the current step is the first with `result` None (execute only
ever emits a call for that step), the observation is the trailing ToolMessage(s) the tool node
(or the approval gate's decline) just appended, and the status derives from the observation text:

    decline ToolMessage (approval rejection)  -> skipped   (a guarded outcome; rectify cancels
                                                            the rest and the answer reports it)
    "BLOCKED..."                              -> blocked
    "error..." / "Error calling ..."          -> error
    anything else                             -> done

This replaced the positional-multiset status walkers (old gotcha #6): with the result recorded
on the step itself there is no cross-walker accounting to keep in sync.
"""

import time

import diag
from langchain.messages import ToolMessage

from core.plan_context import clean
from core.state import AgentState

# The approval gate's decline text (nodes/approval.py DECLINE_TEXT) starts with this — the
# recorder keys the `skipped` status off it so a rejection is an incident, never a "done".
_DECLINE_PREFIX = "Execution declined by the user"


def _status_for(observation: str) -> str:
    text = observation.strip()
    low = text.lower()
    if text.startswith(_DECLINE_PREFIX):
        return "skipped"
    if low.startswith("blocked"):
        return "blocked"
    if low.startswith("error"):
        return "error"
    return "done"


def update_plan_node(state: AgentState):
    start = time.perf_counter()
    plan = state.get("plan") or []
    idx = next((i for i, s in enumerate(plan) if s.get("result") is None), None)
    if idx is None:
        return {}

    # The observation: every trailing ToolMessage (normally exactly one — execute emits a single
    # call per step; a mixed approval decision can leave a decline + a result, joined in order).
    msgs = state.get("messages") or []
    parts: list[str] = []
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            parts.append(str(m.content))
            continue
        break
    if not parts:
        return {}  # nothing to record (defensive — the graph only routes here after a round)

    observation = "\n\n".join(reversed(parts))
    plan = [dict(s) for s in plan]  # work on a copy — never mutate state's plan in place
    step = plan[idx]
    step["result"] = clean(observation)
    step["status"] = _status_for(observation)

    diag.log(
        f"update_plan_node : {time.perf_counter() - start:.4f}s "
        f"(step {step.get('step_id')} -> {step['status']})"
    )
    return {"plan": plan}
