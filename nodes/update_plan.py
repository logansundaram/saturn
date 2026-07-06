"""
update_plan node — the mechanical recorder (the 2026-07-03 engine transplant).

Runs after each tool round (and after a fully-rejected approval batch) and does ONE thing: write
the observation onto the current plan step — the plan IS the data bus, so this is the moment a
step's result becomes available to every later step's curated context, to rectify, and to
synthesize.

Intentionally MECHANICAL, no LLM: the current step is the first with `result` None (execute only
ever emits a call for that step), the observation is the trailing ToolMessage(s) the tool node
(or the approval gate's decline) just appended, and the status is read off each message's
STRUCTURAL stamp (`additional_kwargs["saturn_status"]`, set by the producer at the moment the
outcome was known — nodes/tools.py for tool rounds, nodes/approval.py for declines):

    approval decline               -> skipped   (a guarded outcome; rectify cancels the rest
                                                 and the answer reports it)
    air-gap refusal (egress slice) -> blocked
    tool raised / unknown tool     -> error
    anything else                  -> done

Status is NEVER sniffed out of observation text: a successful read of a file whose content
happens to start with "ERROR:" or "Blocked …" must not fail its step (and the air-gap refusal
string never started with "blocked" anyway — the old text contract was dead). The one textual
fallback kept is the DECLINE_TEXT prefix, belt-and-braces for an unstamped decline.

This replaced the positional-multiset status walkers (old gotcha #6): with the result recorded
on the step itself there is no cross-walker accounting to keep in sync.
"""

import time

import diag
from langchain.messages import ToolMessage

from core.plan_context import clean
from core.state import AgentState
from nodes.approval import DECLINE_TEXT

# When several trailing ToolMessages record onto one step (a mixed approval decision leaves a
# decline + a result), the guarded/failed outcome wins: a rejection must never be averaged away
# by a sibling call's success.
_STATUS_RANK = {"done": 0, "error": 1, "blocked": 2, "skipped": 3}


def _status_of(msg: ToolMessage) -> str:
    """One message's outcome: the producer's structural stamp, else the decline prefix
    (belt-and-braces for an unstamped decline), else done."""
    stamped = (getattr(msg, "additional_kwargs", None) or {}).get("saturn_status")
    if stamped in _STATUS_RANK:
        return stamped
    if str(msg.content).strip().startswith(DECLINE_TEXT):
        return "skipped"
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
    trailing: list[ToolMessage] = []
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            trailing.append(m)
            continue
        break
    if not trailing:
        return {}  # nothing to record (defensive — the graph only routes here after a round)

    trailing.reverse()
    observation = "\n\n".join(str(m.content) for m in trailing)
    status = max((_status_of(m) for m in trailing), key=_STATUS_RANK.__getitem__)
    plan = [dict(s) for s in plan]  # work on a copy — never mutate state's plan in place
    step = plan[idx]
    step["result"] = clean(observation)
    step["status"] = status

    diag.log(
        f"update_plan_node : {time.perf_counter() - start:.4f}s "
        f"(step {step.get('step_id')} -> {step['status']})"
    )
    return {"plan": plan}
