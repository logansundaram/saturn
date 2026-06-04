"""
Agent node + its exit routing for the living-plan ReAct loop (Phase 1).

  agent_node        — the reason-and-act step: model with tools bound natively. Reads context
                      + plan + conversation, emits tool calls, or a no-tool-call message to
                      signal it is done gathering. Increments `iteration`.
  route_after_agent — conditional edge after the agent node: tool calls -> approval, otherwise
                      -> synthesize. Also enforces the iteration guardrail
                      (config runtime.max_iterations) so a confused model can't spin forever.

render_plan formats the plan as a checklist injected into the agent's context (the text twin
of ui.show_plan).
"""

import time

from langchain.messages import SystemMessage

from llms import get_tool_model, extract_tok_per_sec, extract_prompt_tokens
from config import get_config
from state import AgentState
from messages import agent_sys_msg

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

    response = get_tool_model().invoke(messages)

    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
        "tok_per_sec": extract_tok_per_sec(response),
        "context_tokens": extract_prompt_tokens(response),
    }


def route_after_agent(state: AgentState) -> str:
    """Send tool requests through the approval gate (while under the cap); else finish.
    The iteration cap is read from config each call so /config can change it live."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("iteration", 0) < get_config().max_iterations:
        return "approval"
    return "synthesize"
