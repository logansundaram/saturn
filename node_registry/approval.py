"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Tool calls at or below the configured `runtime.auto_approve` risk tier pass straight through;
anything riskier pauses via a LangGraph `interrupt` so the user can approve or reject the whole
batch. The policy is read from config each call, so /config can loosen or tighten it live.
Resuming with the user's decision is handled in agent.run_turn.
"""

from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

from config import get_config
from registry import risk_of
from state import AgentState


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

    decline = [
        ToolMessage(
            content="Execution declined by the user. Do not retry this action; tell the user you did not perform it.",
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in tool_calls
    ]
    return Command(goto="agent", update={"messages": decline})
