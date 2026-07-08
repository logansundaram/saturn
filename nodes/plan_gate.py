"""
plan_gate node — the human-in-the-loop *plan-review* checkpoint.

It sits at every step boundary of the plan/execute loop (after `plan`, after `replan`, and after
each rectify cycle that continues), immediately before the execute node acts. Almost always it's
a no-op pass-through: it checks whether a pause has been *requested* and, if not, returns `{}`
and control flows to `execute`.

When a pause IS requested it raises a LangGraph `interrupt()` carrying the current plan, so the
driver (`agent.run_turn`) can hand it to the user, who inspects/edits the plan and resumes. On
resume the (possibly edited) plan is written back into state and execution continues from the
current step with the corrected plan — or, if the user aborted, routing falls through to
`synthesize`. This is what lets a hallucinated or wrong plan be fixed mid-flight instead of
running to a bad conclusion.

One trigger seam feeds it: the shared `plan_ops.PauseController` (the mid-turn Esc key, handled
by `typeahead.InputQueue`, and the `/plan pause` + `/plan review` commands). (A speculative
second in-graph seam — a `state["pause_requested"]` flag for a future LLM-initiated pause —
was deleted 2026-07-04: nothing ever set it, and dead state every writer must keep resetting
is a standing hazard. A future pause source sets the SAME controller.)

The same controller carries a *third*, non-pausing action: **mid-turn steering** (`source="steer"`,
the typed correction in `reason`). Under the plan/execute engine (2026-07-03 transplant) a steer
rides the REPLAN seam: the gate records the correction into the conversation (a HumanMessage, so
history/compaction/recap see it), sets `rectify=True` with the correction as `reasoning`, and
`route_after_gate` sends the turn through `replan` — the remaining steps are redrafted around the
user's correction, then execution continues. The running turn is adjusted, not interrupted.

Determinism across the interrupt: a resumed `interrupt()` re-executes its node from the top, so the
path to the `interrupt()` call must be the same on the re-run. The controller is read
non-destructively (`pending()`/`peek()`) and only `clear()`ed *after* the interrupt returns —
so the pause decision evaluates the same both times.
"""

from langchain.messages import HumanMessage
from langgraph.types import interrupt

from core.state import AgentState, current_step, STEER_PREFIX
from core.plan_ops import get_pause_controller, is_review_retirement


def _review_vetoes(before: list, after: list) -> list:
    """Labels of un-run steps the user's review edit removed (`drop`) or retired (the editor's
    `status` verb → skipped/cancelled/blocked, carrying plan_ops' review stamp). These are the
    human's explicit "do not do this" — recorded onto state["plan_vetoes"] so the engine's
    self-correction (rectify's judge, replan) can never reinstate the work; the human's edit
    outranks the judge, the same principle as the gate's guarded outcome.

    Matching is by label: a step the user merely RELABELED reads as a veto of the old wording,
    which is benign — the reworded step is still in the plan and runs; the note only tells the
    engine not to resurrect the wording the user rewrote."""
    after_labels = {str(s.get("label") or "").strip().lower() for s in after or []}
    before_retired = {
        str(s.get("label") or "").strip().lower()
        for s in before or []
        if is_review_retirement(s)
    }
    out: list = []
    for s in before or []:
        label = str(s.get("label") or "").strip()
        if s.get("result") is None and label and label.lower() not in after_labels:
            out.append(label)  # dropped while still un-run (a dropped DONE step is bookkeeping)
    for s in after or []:
        label = str(s.get("label") or "").strip()
        if (
            is_review_retirement(s)
            and label
            and label.lower() not in before_retired  # only NEWLY retired (re-reviews don't dup)
            and label not in out
        ):
            out.append(label)
    return out


def plan_gate_node(state: AgentState):
    controller = get_pause_controller()

    # Mid-turn steering: a correction the user typed during execution (Esc with text). Record it
    # in the conversation AND arm a replan with the correction as the revision instruction —
    # steering edits the running turn WITHOUT interrupting it (unlike the review pause below). No
    # interrupt() here, so the determinism caveat below doesn't apply to this branch.
    req = controller.peek()
    if req is not None and req.source == "steer" and req.reason:
        controller.clear()
        # Built from state.STEER_PREFIX so the standalone form below is recognizable by
        # state.is_steer_message — the consumers that slice the conversation at HumanMessage
        # boundaries (_compact_history, the grounding recap) skip it.
        note = f"\n{STEER_PREFIX} {req.reason}"
        steer_updates = {
            "rectify": True,
            "reasoning": (
                "Mid-task steering correction from the user: "
                f"{req.reason}\nRedraft the remaining steps to honor this correction."
            ),
        }
        # Carry the correction on the LAST message rather than appending a fresh HumanMessage:
        # at the first boundary the trailing message is already user-role, and a second
        # consecutive user turn is rejected by providers that require role alternation (e.g.
        # Anthropic on the cloud tier). add_messages overwrites by id, so the edited copy
        # replaces it.
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, HumanMessage) and getattr(last, "id", None):
            steer_updates["messages"] = [
                HumanMessage(content=str(last.content) + note, id=last.id)
            ]
        else:
            steer_updates["messages"] = [HumanMessage(content=note.lstrip())]
        return steer_updates

    # Decide whether to pause. Kept side-effect-free so it's identical on a post-interrupt
    # re-execution (see module docstring).
    if not controller.pending():
        return {}
    req = controller.peek()
    reason = req.reason if req and req.reason else "pause requested"

    plan = state.get("plan", [])
    review = interrupt(
        {
            "type": "plan_review",
            "plan": plan,
            "reason": reason,
            "active_step": current_step(plan),
            "iteration": state.get("iteration", 0),
        }
    )

    # --- resumed here with the user's decision ---
    controller.clear()  # consume the external request now that it's been handled

    updates: dict = {}
    if isinstance(review, dict):
        edited = review.get("plan")
        if edited is not None and edited != plan:
            updates["plan"] = edited
            # Record what the user REMOVED as vetoes (read-merge-write; only this node writes
            # the field) so rectify/replan/synthesize treat it as deliberately out of scope.
            vetoes = _review_vetoes(plan, edited)
            if vetoes:
                existing = list(state.get("plan_vetoes") or [])
                updates["plan_vetoes"] = existing + [v for v in vetoes if v not in existing]
        if review.get("action") == "abort":
            updates["aborted"] = True
    # A non-dict resume value (e.g. a bare True from an auto-approver that never expected this
    # interrupt) means "continue unchanged" — nothing to update.

    return updates


def route_after_gate(state: AgentState) -> str:
    """After the gate: abort -> wrap up at synthesize; a steer armed a replan -> revise the
    remaining steps first; otherwise -> execute the current step."""
    if state.get("aborted"):
        return "synthesize"
    if state.get("rectify"):
        return "replan"
    return "execute"
