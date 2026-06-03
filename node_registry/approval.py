"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Read-only tool batches pass straight through. If any pending tool call is side-effecting or
destructive, pause via a LangGraph `interrupt` and let the user approve or reject the whole
batch. Resuming with the user's decision is handled in agent.run_turn.
"""

from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

from registry import risk_of
from state import AgentState


def approval_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """Human-in-the-loop safety gate. Read-only tool batches pass straight through. If any
    pending tool call is side-effecting/destructive, pause via `interrupt` and let the user
    approve or reject the whole batch.

    On reject we still emit ToolMessages for every pending call (so the message history stays
    valid — orphaned tool_calls break the next model turn) and route back to the agent to
    respond without having performed the action."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    gated = [tc for tc in tool_calls if risk_of(tc["name"]) != "read_only"]

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
