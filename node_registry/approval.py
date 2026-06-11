"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Whether a call skips the human is ONE question asked of ONE object: `policy.approves(name,
risk, args)` (the tier threshold + the /allow shell allowlist — see policy.py). Anything it
doesn't approve pauses via a LangGraph `interrupt` so the user can decide per batch or per
call. The policy is read live each call, so /config, /risk, /allow, /autoapprove and Shift+Tab
all apply to the very next gate. Resuming with the user's decision is handled in agent.run_turn.
"""

from collections import Counter
from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

import policy
from config import get_config
from registry import risk_of
from state import AgentState, TERMINAL_STATUSES


def _skip_rejected_steps(
    plan: list[dict],
    rejected_tools: list[str],
    executing_tools: "list[str] | None" = None,
    called: "list[str] | None" = None,
) -> list[dict]:
    """Mark the planned step(s) a rejected tool call was fulfilling as `skipped`.

    Without this, a rejected call leaves its planned step non-terminal: `active_step` keeps the
    lockstep directive pinned to it and `unrun_planned_tools` keeps `route_after_agent` nudging,
    so the agent re-issues the very call the user just declined and the approval prompt fires again
    and again (bounded only by max_iterations — the reject -> infinite re-approve loop). Skipping
    the step retires that planned work so the plan advances past it.

    Matching mirrors `update_plan`/`unrun_planned_tools`: each rejected tool name is consumed,
    positionally as a multiset, against the first non-terminal step that expects it (so two
    same-tool steps don't both get skipped off one rejection) — BUT only after reserving the steps
    that the batch's surviving calls (`executing_tools`, the approved + ungated calls about to run)
    and prior rounds' calls (`called`) will credit. On a mixed approve/reject decision over a
    same-tool batch this is what keeps skip and credit aligned with update_plan's walk: without the
    reserve, the rejection would skip the FIRST matching step, the executed call's credit would
    flow past it to the next one, and the plan would record the opposite of what happened.

    A rejected call whose tool matched no planned step falls back to skipping the first
    non-reserved step with an intended_tool — the planner's `intended_tool` guess didn't match what
    the agent actually called, but that step is still what lockstep was driving, so it must advance
    too. Returns a fresh list; never mutates the input."""
    if not plan:
        return plan
    plan = [dict(s) for s in plan]
    # Calls that have already credited steps (prior rounds) or are about to (this batch's
    # survivors). Done steps consume from this first, exactly like update_plan's walk.
    reserve = Counter(called or []) + Counter(executing_tools or [])
    remaining = Counter(rejected_tools)
    reserved_ids: set = set()
    for step in plan:
        tool = step.get("intended_tool")
        status = step.get("status")
        if status in TERMINAL_STATUSES:
            if status == "done" and tool and reserve.get(tool, 0) > 0:
                reserve[tool] -= 1
            continue
        if not tool:
            continue
        if reserve.get(tool, 0) > 0:
            # An executed (or about-to-execute) call covers this step — leave it for update_plan
            # to credit as done.
            reserve[tool] -= 1
            reserved_ids.add(id(step))
            continue
        if remaining.get(tool, 0) > 0:
            remaining[tool] -= 1
            step["status"] = "skipped"
    # Fallback: a rejection didn't line up with any planned tool. Skip the first un-reserved step
    # so the lockstep directive stops re-pointing the agent at the work it was just told not to do.
    if sum(remaining.values()) > 0:
        for step in plan:
            if step.get("status") in TERMINAL_STATUSES or id(step) in reserved_ids:
                continue
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

    # Dry-run: nothing will actually execute (tool_node stubs every call), so there is no action to
    # approve — pass straight through to the (stubbing) tool node. This is what lets a dry-run show
    # the WHOLE intended arc, including gated calls, without prompting the human for actions that
    # won't happen.
    if bool(cfg.get("runtime.dry_run", False)):
        return Command(goto="tools")

    gated = [
        tc
        for tc in tool_calls
        if not policy.approves(tc["name"], risk_of(tc["name"]), tc.get("args"))
    ]

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
        rejected_ids = {tc["id"] for tc in rejected}
        update["plan"] = _skip_rejected_steps(
            plan,
            [tc["name"] for tc in rejected],
            # The batch's survivors (approved gated + ungated calls) WILL run and be credited by
            # update_plan — reserve their steps so a rejection doesn't skip the wrong one.
            executing_tools=[tc["name"] for tc in tool_calls if tc["id"] not in rejected_ids],
            called=state.get("tools_called", []),
        )

    # Anything left to run (ungated or approved) still runs; a fully-rejected batch goes back to
    # the agent instead.
    if len(rejected) < len(tool_calls):
        return Command(goto="tools", update=update)
    return Command(goto="agent", update=update)
