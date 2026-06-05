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

from langchain.messages import SystemMessage, AIMessage

from llms import get_tool_model, extract_tok_per_sec, extract_prompt_tokens
from config import get_config
from state import AgentState, unrun_planned_tools, active_step
from messages import (
    agent_sys_msg,
    agent_next_step_directive,
    agent_nudge_directive,
    agent_lockstep_directive,
)

_STATUS_GLYPH = {"pending": "○", "active": "▶", "done": "✓", "skipped": "—"}

# How many times in one turn the agent may be sent back to act after finishing with an un-run
# planned tool still pending. A small budget: it rescues the common "answered instead of
# searching" miss without letting a model that truly won't call the tool spin to max_iterations.
NUDGE_BUDGET = 2


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
    """One ReAct decision: look at the plan + conversation, then call tools or finish.

    Two plan-aware injections keep the planner's `intended_tool` annotations driving behaviour
    (not just decorating the trace):
      - a soft NEXT-PLANNED-ACTION pointer at the first un-run gathering step every pass, and
      - when route_after_agent has looped us back (the last message is our own no-tool-call
        answer while planned work remains), a pointed nudge naming the skipped step(s). The
        nudge is what makes the loop-back productive: re-invoking on the same inputs would just
        reproduce the same refusal, so we change the inputs."""
    start = time.perf_counter()

    plan = state.get("plan", [])
    pending = unrun_planned_tools(plan, state.get("tools_called", []))

    # Plan-aware focus. In LOCKSTEP mode (config runtime.lockstep, default on) the model is told to
    # execute exactly the current step — the plan is followed step-by-step. Otherwise fall back to
    # the soft "next planned action" pointer at the first un-run gathering step (advisory).
    extras = []
    lockstep = get_config().lockstep
    current = active_step(plan)
    if lockstep and current:
        extras.append(agent_lockstep_directive(current))
    elif pending:
        extras.append(agent_next_step_directive(pending[0]))

    # Detect a route_after_agent nudge: we only re-enter `agent` directly after our own AIMessage
    # with no tool calls. If planned gathering work is still open at that point, escalate.
    last = state["messages"][-1] if state.get("messages") else None
    nudges = state.get("agent_nudges", 0)
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None) and bool(pending):
        extras.append(agent_nudge_directive(pending))
        nudges += 1

    messages = [
        agent_sys_msg,
        SystemMessage(content=state.get("context", "")),
        SystemMessage(content="Current plan:\n" + render_plan(plan)),
        *state["messages"],
        *extras,
    ]

    response = get_tool_model().invoke(messages)

    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
        "agent_nudges": nudges,
        "tok_per_sec": extract_tok_per_sec(response),
        "context_tokens": extract_prompt_tokens(response),
    }


def route_after_agent(state: AgentState) -> str:
    """Route the agent's output: tool calls -> approval; an early finish with planned work still
    pending -> back to `agent` (the plan-aware nudge); otherwise -> synthesize.

    The iteration cap (read live from config so /config can change it) bounds every loop edge.
    The nudge is additionally bounded by NUDGE_BUDGET so a model that simply won't call the tool
    falls through to an honest synthesize instead of spinning to the iteration cap."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    iteration = state.get("iteration", 0)
    max_iterations = get_config().max_iterations

    if iteration >= max_iterations:
        return "synthesize"
    if has_tool_calls:
        return "approval"

    # No tool calls: the model thinks it's done. If the plan still has an un-run gathering step
    # and we have nudge budget left, send it back to act (agent_node injects the correction).
    if state.get("agent_nudges", 0) < NUDGE_BUDGET and unrun_planned_tools(
        state.get("plan", []), state.get("tools_called", [])
    ):
        return "agent"
    return "synthesize"
