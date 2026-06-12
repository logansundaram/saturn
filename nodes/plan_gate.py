"""
plan_gate node — the human-in-the-loop *plan-review* checkpoint.

It sits at every step boundary of the living-plan loop (after `plan`, and after each
`update_plan`), immediately before the agent acts. Almost always it's a no-op pass-through: it
checks whether a pause has been *requested* and, if not, returns `{}` and control flows to `agent`.

When a pause IS requested it raises a LangGraph `interrupt()` carrying the current plan, so the
driver (`agent.run_turn`) can hand it to the user, who inspects/edits the plan and resumes. On
resume the (possibly edited) plan is written back into state and execution continues from the
current step with the corrected plan — or, if the user aborted, routing falls through to
`synthesize`. This is what lets a hallucinated or wrong plan be fixed mid-flight instead of running
to a bad conclusion (see the brittleness notes in CLAUDE.md).

Two independent trigger seams feed it, by design, so the *source* of a pause is modular:
  - external / async / between-turns: the shared `interrupts.PauseController` (the mid-turn Esc key,
    handled by `typeahead.InputQueue`, and the `/plan pause` + `/plan review` commands), and
  - in-graph: the `state["pause_requested"]` flag — the seam a future LLM-initiated
    "request a plan review" node/tool would set. The gate handles both identically.

The same controller carries a *third*, non-pausing action: **mid-turn steering** (`source="steer"`,
the typed correction in `reason`). When the user types a correction during execution and hits Esc
(`typeahead.InputQueue`), the gate injects it as a HumanMessage so the agent's next pass heeds it,
then clears the request and passes straight through — the running turn is adjusted, not interrupted.

Determinism across the interrupt: a resumed `interrupt()` re-executes its node from the top, so the
path to the `interrupt()` call must be the same on the re-run. The controller is read
non-destructively (`pending()`/`peek()`) and only `clear()`ed *after* the interrupt returns, and
the state flag doesn't change mid-node — so `should_pause` evaluates the same both times.
"""

from langchain.messages import HumanMessage
from langgraph.types import interrupt

from core.state import AgentState, active_step, STEER_PREFIX
from core.plan_ops import get_pause_controller


def plan_gate_node(state: AgentState):
    controller = get_pause_controller()

    # Mid-turn steering: a correction the user typed during execution (Esc with text). Inject it as
    # a HumanMessage so the agent's next pass adjusts course, then clear the request and pass through
    # — steering edits the running turn WITHOUT interrupting it (unlike the review pause below). No
    # interrupt() here, so the determinism caveat below doesn't apply to this branch.
    req = controller.peek()
    if req is not None and req.source == "steer" and req.reason:
        controller.clear()
        # Built from state.STEER_PREFIX so the standalone form below is recognizable by
        # state.is_steer_message — the consumers that slice the conversation at HumanMessage
        # boundaries (/rewind, /retry full, _compact_history, the grounding recap) skip it.
        note = f"\n{STEER_PREFIX} {req.reason}"
        # Carry the correction on the LAST message rather than appending a fresh HumanMessage: after
        # a tool round (and at the first boundary) the trailing message is already user-role, and a
        # second consecutive user turn is rejected by providers that require role alternation (e.g.
        # Anthropic on the cloud tier). add_messages overwrites by id, so the edited copy replaces it.
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, HumanMessage) and getattr(last, "id", None):
            return {"messages": [HumanMessage(content=str(last.content) + note, id=last.id)]}
        return {"messages": [HumanMessage(content=note.lstrip())]}

    # Decide whether to pause from the two seams. Kept side-effect-free so it's identical on a
    # post-interrupt re-execution (see module docstring).
    paused = bool(state.get("pause_requested"))
    reason = state.get("pause_reason") or ""
    if not paused and controller.pending():
        paused = True
        req = controller.peek()
        reason = (req.reason if req and req.reason else "pause requested")

    if not paused:
        return {}

    plan = state.get("plan", [])
    review = interrupt(
        {
            "type": "plan_review",
            "plan": plan,
            "reason": reason,
            "active_step": active_step(plan),
            "iteration": state.get("iteration", 0),
        }
    )

    # --- resumed here with the user's decision ---
    controller.clear()  # consume the external request now that it's been handled

    updates: dict = {}
    # Reset the in-graph flag only if it was actually set (keep the delta minimal so the trace
    # doesn't render a no-op gate line; see ui.show_node).
    if state.get("pause_requested"):
        updates["pause_requested"] = False
        updates["pause_reason"] = ""

    if isinstance(review, dict):
        edited = review.get("plan")
        if edited is not None and edited != plan:
            updates["plan"] = edited
        if review.get("action") == "abort":
            updates["aborted"] = True
    # A non-dict resume value (e.g. a bare True from an auto-approver that never expected this
    # interrupt) means "continue unchanged" — nothing to update.

    return updates


def route_after_gate(state: AgentState) -> str:
    """After the gate: abort -> wrap up at synthesize; otherwise -> act on the (current) plan."""
    if state.get("aborted"):
        return "synthesize"
    return "agent"
