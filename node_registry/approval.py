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
    approve or reject the whole batch.

    On reject we still emit ToolMessages for every pending call (so the message history stays
    valid — orphaned tool_calls break the next model turn) and route back to the agent to
    respond without having performed the action."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    cfg = get_config()
    gated = [tc for tc in tool_calls if not cfg.auto_approves(risk_of(tc["name"]))]

    if not gated:
        return Command(goto="tools")

    approved = interrupt(
        {
            "type": "approval_request",
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"], "risk": risk_of(tc["name"])}
                for tc in gated
            ],
        }
    )

    if approved:
        return Command(goto="tools")

    # Every pending tool_call still needs a ToolMessage (orphaned calls break the next model
    # turn), but only the calls the user actually rejected should be told not to retry. A
    # read-only call merely bundled into the same batch was never gated — decline it neutrally so
    # the agent can re-issue it on its own next iteration instead of abandoning the result.
    gated_ids = {tc["id"] for tc in gated}
    decline = []
    for tc in tool_calls:
        if tc["id"] in gated_ids:
            content = (
                "Execution declined by the user. Do not retry this action; tell the user you "
                "did not perform it."
            )
        else:
            content = (
                "Not executed: this read-only call was held with a batch the user declined. "
                "Call it again on its own if you still need its result."
            )
        decline.append(
            ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"])
        )

    # Retire the planned step(s) the rejected calls were fulfilling, so the plan stops demanding
    # work the user declined (otherwise lockstep + the nudge re-issue the same call indefinitely).
    update = {"messages": decline}
    plan = state.get("plan", [])
    if plan:
        update["plan"] = _skip_rejected_steps(plan, [tc["name"] for tc in gated])
    return Command(goto="agent", update=update)
