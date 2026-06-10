"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Tool calls at or below the configured `runtime.auto_approve` risk tier pass straight through;
anything riskier pauses via a LangGraph `interrupt` so the user can approve or reject the whole
batch. The policy is read from config each call, so /config can loosen or tighten it live.
Resuming with the user's decision is handled in agent.run_turn.
"""

from collections import Counter
from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

from config import get_config
from registry import risk_of
from state import AgentState


def _skip_rejected_steps(plan: list[dict], rejected_tools: list[str]) -> list[dict]:
    """Mark the planned step(s) a rejected tool call was fulfilling as `skipped`.

    Without this, a rejected call leaves its planned step non-terminal: `active_step` keeps the
    lockstep directive pinned to it and `unrun_planned_tools` keeps `route_after_agent` nudging,
    so the agent re-issues the very call the user just declined and the approval prompt fires again
    and again (bounded only by max_iterations — the reject -> infinite re-approve loop). Skipping
    the step retires that planned work so the plan advances past it.

    Matching mirrors `update_plan`/`unrun_planned_tools`: each rejected tool name is consumed,
    positionally as a multiset, against the first non-terminal step that expects it (so two
    same-tool steps don't both get skipped off one rejection). A rejected call whose tool matched no
    planned step falls back to skipping the current active step — the planner's `intended_tool`
    guess didn't match what the agent actually called, but that active step is still what lockstep
    was driving, so it must advance too. Returns a fresh list; never mutates the input."""
    if not plan:
        return plan
    plan = [dict(s) for s in plan]
    remaining = Counter(rejected_tools)
    for step in plan:
        if step.get("status") in ("done", "skipped"):
            continue
        tool = step.get("intended_tool")
        if tool and remaining.get(tool, 0) > 0:
            remaining[tool] -= 1
            step["status"] = "skipped"
    # Fallback: a rejection didn't line up with any planned tool. Skip the current active step so
    # the lockstep directive stops re-pointing the agent at the work it was just told not to do.
    if sum(remaining.values()) > 0:
        for step in plan:
            if step.get("status") not in ("done", "skipped"):
                # Only skip a step that expected a tool. A no-intended_tool step (e.g. the generic
                # "answer the request" fallback plan) shouldn't be retired just because an unplanned
                # gated call was declined — the user declined one action, not the whole task.
                if step.get("intended_tool"):
                    step["status"] = "skipped"
                break
    return plan


def approval_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """Human-in-the-loop safety gate. Calls within the configured auto-approve tier pass
    straight through. If any pending call exceeds it, pause via `interrupt` and let the user
    decide per batch OR per call.

    The resume value is either a bool (True = approve the whole batch, False = reject it) or
    `{"approved_ids": [...]}` from the UI's per-call select mode. Rejected calls get a decline
    ToolMessage here (orphaned tool_calls break the next model turn); everything else in the
    batch — ungated calls and per-call-approved ones — still routes to `tools`, which executes
    only the calls that don't already have a ToolMessage. Only a fully-rejected batch goes back
    to the agent to respond without having acted."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    cfg = get_config()
    gated = [tc for tc in tool_calls if not cfg.auto_approves(risk_of(tc["name"]))]

    if not gated:
        return Command(goto="tools")

    decision = interrupt(
        {
            "type": "approval_request",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                    "risk": risk_of(tc["name"]),
                }
                for tc in gated
            ],
        }
    )

    # Resolve the decision into the set of approved gated-call ids.
    gated_ids = {tc["id"] for tc in gated}
    if isinstance(decision, dict):
        approved_ids = gated_ids & set(decision.get("approved_ids") or [])
    elif decision:
        approved_ids = set(gated_ids)
    else:
        approved_ids = set()

    if approved_ids == gated_ids:
        return Command(goto="tools")

    # Decline ONLY the rejected calls. (Previously a rejection abandoned the whole batch and a
    # read-only call riding along had to be re-issued next iteration; now it just runs.)
    rejected = [tc for tc in gated if tc["id"] not in approved_ids]
    decline = [
        ToolMessage(
            content=(
                "Execution declined by the user. Do not retry this action; tell the user you "
                "did not perform it."
            ),
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in rejected
    ]

    # Retire the planned step(s) the rejected calls were fulfilling, so the plan stops demanding
    # work the user declined (otherwise lockstep + the nudge re-issue the same call indefinitely).
    update = {"messages": decline}
    plan = state.get("plan", [])
    if plan:
        update["plan"] = _skip_rejected_steps(plan, [tc["name"] for tc in rejected])

    # Anything left to run (ungated or approved) still runs; a fully-rejected batch goes back to
    # the agent instead.
    if len(rejected) < len(tool_calls):
        return Command(goto="tools", update=update)
    return Command(goto="agent", update=update)
