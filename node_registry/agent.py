"""
Agent node + its exit routing for the living-plan ReAct loop (Phase 1).

  agent_node        — the reason-and-act step: model with tools bound natively. Reads context
                      + plan + conversation, emits tool calls, or a no-tool-call message to
                      signal it is done gathering. Increments `iteration`.
  route_after_agent — conditional edge after the agent node: tool calls -> approval, otherwise
                      -> synthesize. Also enforces the MAX_ITERATIONS guardrail so a confused
                      model can't spin forever.

render_plan formats the plan as a checklist injected into the agent's context (the text twin
of ui.show_plan).
"""

import time

from langchain.messages import SystemMessage

from llms import llm_with_tools
from state import AgentState
from messages import agent_sys_msg

# Hard cap on loop iterations so a confused model can't spin forever. Becomes config in Phase 3.
MAX_ITERATIONS = 8

_STATUS_GLYPH = {"pending": "○", "active": "▶", "done": "✓", "skipped": "—"}


def render_plan(plan: list[dict]) -> str:
    """Human-readable checklist, injected into the agent's context and streamed to the UI."""
    if not plan:
        return "(no plan)"
    lines = []
    for step in plan:
        glyph = _STATUS_GLYPH.get(step["status"], "○")
        tool = f"  [{step['intended_tool']}]" if step.get("intended_tool") else ""
        lines.append(f"{glyph} {step['step_id']}. {step['label']}{tool}")
    return "\n".join(lines)


def agent_node(state: AgentState):
    """One ReAct decision: look at the plan + conversation, then call tools or finish."""
    start = time.perf_counter()

    messages = [
        agent_sys_msg,
        SystemMessage(content=state.get("context", "")),
        SystemMessage(content="Current plan:\n" + render_plan(state.get("plan", []))),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    print(
        f"agent_node : {time.perf_counter() - start:.4f}s "
        f"(iter {state.get('iteration', 0)}, {len(tool_calls)} tool call(s))"
    )
    return {"messages": [response], "iteration": state.get("iteration", 0) + 1}


def route_after_agent(state: AgentState) -> str:
    """Send tool requests through the approval gate (while under the cap); else finish."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("iteration", 0) < MAX_ITERATIONS:
        return "approval"
    return "synthesize"
