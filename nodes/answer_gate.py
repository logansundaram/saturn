"""
answer_gate node — the human-in-the-loop *freeze-then-edit* checkpoint of interrupt-and-correct.

It is reached exactly one way: the user pressed Esc while the final answer was streaming, the
synthesize node stopped the stream cleanly and returned the provenance-tagged buffer with
`state: "frozen"`, and `route_after_synthesize` sent the turn here. The node `interrupt()`s with
the frozen text so the driver (app/turn.run_turn → the loop's approver) can open the freeze
editor: the user deletes the bad span, types a correction, and resumes — or accepts the text
as-is as the final answer. The (possibly edited) buffer is written back with the changed region
recorded as a human-authored span (core/provenance.apply_edit), and routing returns to
`synthesize`, which CONTINUES the edited prefix (or finalizes, on "done").

This is token-level surgery in the same family as the plan gate's step-level surgery: freeze →
edit the assistant-message prefix → continue that exact prefix. The correction is a first-class
auditable event — the buffer (spans + edit records) rides this node's state delta into the
trace DB, so the live rail, `/trace` replays, and the answer's own rendering all show which
characters the human wrote.

Determinism across the interrupt: a resumed `interrupt()` re-executes the node from the top, so
everything before the call is a pure read of state — same both times (the plan_gate contract).

Resume-value shapes: `{"action": "resume"|"done", "text": <edited full text>}` from the editor;
anything else (a bare True from an approver that never expected this interrupt — headless)
means "continue unchanged", which simply resumes generation from the frozen text.
"""

from langgraph.types import interrupt

from core import provenance
from core.state import AgentState


def answer_gate_node(state: AgentState):
    buf = state.get("answer_buffer") or provenance.new_buffer()

    decision = interrupt(
        {
            "type": "answer_edit",
            "text": buf.get("text", ""),
            "spans": buf.get("spans", []),
            # The token-confidence overlay: the freeze editor marks low-confidence runs red so
            # the user sees WHERE the model itself was unsure — the natural edit targets.
            "confidence": buf.get("confidence", []),
            "query": state.get("current_query", ""),
        }
    )

    # --- resumed here with the user's decision ---
    action, text = "resume", buf.get("text", "")
    if isinstance(decision, dict):
        if decision.get("action") in ("resume", "done"):
            action = decision["action"]
        if isinstance(decision.get("text"), str):
            text = decision["text"]

    edited = provenance.apply_edit(buf, text)
    return {
        "answer_buffer": {
            **edited,
            "state": "done" if action == "done" else "resume",
            # Whether THIS gate pass changed the text — the rail/replay echo keys on it (the
            # `edits` list is cumulative, so its mere presence can't distinguish this pass).
            "edited": text != buf.get("text", ""),
        }
    }
