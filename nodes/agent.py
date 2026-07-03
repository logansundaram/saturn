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

from langchain.messages import SystemMessage, AIMessage

from core.llms import (
    get_tool_model,
    extract_tok_per_sec,
    extract_prompt_tokens,
)
from config import get_config
from core.state import AgentState, unrun_planned_tools, active_step
from core.messages import (
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

# How many times in one turn the judge/replan node may insert a gathering step because the agent's
# draft answer was ungrounded. One escalation is enough for the common "answered a 'best X'/current-
# facts question from parametric knowledge instead of searching" miss; more would risk loops (and is
# bounded by the iteration cap regardless). See nodes/replan.py.
REPLAN_BUDGET = 1


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
    plan = state.get("plan", [])
    lockstep = get_config().lockstep

    # We re-enter `agent` directly after our own AIMessage with no tool calls only via the
    # route_after_agent nudge edge (planned work still open). Detect that once here — it drives both
    # the lockstep no-tool advance just below and the nudge escalation further down.
    last = state["messages"][-1] if state.get("messages") else None
    finished_no_tool = isinstance(last, AIMessage) and not getattr(last, "tool_calls", None)

    # Lockstep advance for no-tool steps. Mechanical update_plan only advances on a tool round, so a
    # no-tool reasoning step in the MIDDLE of the plan would otherwise pin `active_step` to itself
    # forever: the lockstep directive keeps telling the model to redo it while the nudge points past
    # it (contradictory), and it never gets marked done (livelock). When we re-enter after producing
    # that step's no-tool message, mark it done so the active pointer moves on — then the lockstep
    # directive and the nudge both target the next (tool) step and agree.
    plan_advanced = False
    if lockstep and finished_no_tool:
        current = active_step(plan)
        if current and not current.get("intended_tool"):
            plan = [dict(s) for s in plan]
            for step in plan:
                if step.get("step_id") == current.get("step_id"):
                    step["status"] = "done"
                    break
            plan_advanced = True

    pending = unrun_planned_tools(plan, state.get("tools_called", []))

    # Plan-aware focus. In LOCKSTEP mode (config runtime.lockstep, default on) the model is told to
    # execute exactly the current step — the plan is followed step-by-step. Otherwise fall back to
    # the soft "next planned action" pointer at the first un-run gathering step (advisory).
    extras = []
    current = active_step(plan)
    if lockstep and current:
        extras.append(agent_lockstep_directive(current))
    elif pending:
        extras.append(agent_next_step_directive(pending[0]))

    # If the model finished with no tool calls while planned gathering work is still open, escalate
    # with the pointed nudge (bounded by NUDGE_BUDGET in route_after_agent). After the lockstep
    # advance above the directive already targets that same work, so the two reinforce rather than
    # fight.
    nudges = state.get("agent_nudges", 0)
    if finished_no_tool and bool(pending):
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

    updates = {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
        "agent_nudges": nudges,
        "tok_per_sec": extract_tok_per_sec(response),
        "context_tokens": extract_prompt_tokens(response),
    }
    # Only emit the plan delta when we actually advanced — keeps the trace's plan diff quiet on the
    # common path (update_plan remains the usual advancer after tool rounds).
    if plan_advanced:
        updates["plan"] = plan
    return updates


def route_after_agent(state: AgentState) -> str:
    """Route the agent's output: tool calls -> approval; an early finish with planned work still
    pending -> back to `agent` (the plan-aware nudge); an apparently-complete finish -> `replan`
    (the judge verifies it's grounded, possibly inserting a web_search); otherwise -> synthesize.

    The iteration cap (read live from config so /config can change it) bounds every loop edge.
    The nudge is additionally bounded by NUDGE_BUDGET and the judge escalation by REPLAN_BUDGET, so
    a model that simply won't act falls through to an honest synthesize instead of spinning to the
    iteration cap."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    iteration = state.get("iteration", 0)
    max_iterations = get_config().max_iterations

    if iteration >= max_iterations:
        return "synthesize"
    if has_tool_calls:
        return "approval"

    # No tool calls: the model thinks it's done. Two separate guards, in priority order:
    unrun = unrun_planned_tools(state.get("plan", []), state.get("tools_called", []))
    # 1) A PLANNED gathering step is still un-run — the agent skipped it. Nudge it back to act
    #    (agent_node injects the pointed correction), bounded by NUDGE_BUDGET.
    if unrun and state.get("agent_nudges", 0) < NUDGE_BUDGET:
        return "agent"
    # 2) Nothing planned is left to run (the agent did everything the plan asked, or the plan never
    #    planned a lookup at all). Before accepting the answer, let the judge verify it's grounded —
    #    it may insert a web_search step if the draft leans on facts that were never looked up.
    if not unrun and state.get("replans", 0) < REPLAN_BUDGET:
        return "replan"
    return "synthesize"
