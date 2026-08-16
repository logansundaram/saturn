"""
Plan-review vetoes (2026-07-06): the human's plan edit outranks the engine's self-correction.

Before this, a step the user removed at the plan-review editor could be resurrected by rectify's
judge (the request still asked for the work, so the judge ruled the plan incomplete and replan
re-added it), and a user marking ONE middle step `skipped` tripped rectify's guarded branch —
built for gate rejections — cancelling the whole remaining plan. Covers: the review stamp's
producer/parser pair (plan_ops), plan_gate's veto detection on review resume, rectify's
review-retirement exemption (vs. the still-cancelling gate rejection), replan's mechanical
resurrection filter + veto-led revision instruction, and the shared vetoes_block. All offline.
"""

import pytest
from langchain.messages import HumanMessage

from core import plan_ops
from core.plan_context import vetoes_block


def _step(sid, label, tool="calculate", status="pending", result=None):
    return {"step_id": sid, "label": label, "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


# --- the review stamp (one producer, one parser) ---------------------------------------------------

def test_set_status_stamps_the_review_retirement():
    plan = [_step(1, "Compute 17 * 23")]
    edited = plan_ops.set_status(plan, 1, "skipped")
    assert edited[0]["result"] == plan_ops.review_stamp("skipped")
    assert plan_ops.is_review_retirement(edited[0])
    # re-pending clears the stamp so the step runs again
    repended = plan_ops.set_status(edited, 1, "pending")
    assert repended[0]["result"] is None and not plan_ops.is_review_retirement(repended[0])


def test_gate_skips_are_not_review_retirements():
    from nodes.approval import DECLINE_TEXT

    assert not plan_ops.is_review_retirement(
        _step(1, "x", status="skipped", result=DECLINE_TEXT + " run_shell"))
    assert not plan_ops.is_review_retirement(None)


# --- plan_gate: veto detection on review resume -----------------------------------------------------

def test_review_vetoes_detects_drops_and_new_retirements():
    from nodes.plan_gate import _review_vetoes

    before = [
        _step(1, "Read the file", status="done", result="contents"),
        _step(2, "Compute 41 * 7"),
        _step(3, "Compute 17 * 23"),
        _step(4, "Old veto", status="skipped", result=plan_ops.review_stamp("skipped")),
    ]
    # user drops step 2 and retires step 3; the already-retired step 4 must not re-count
    after = plan_ops.set_status(plan_ops.drop_step(before, 2), 2, "skipped")
    vetoes = _review_vetoes(before, after)
    assert "Compute 41 * 7" in vetoes
    assert "Compute 17 * 23" in vetoes
    assert "Old veto" not in vetoes


def test_review_vetoes_ignores_dropped_done_steps():
    from nodes.plan_gate import _review_vetoes

    before = [_step(1, "Read the file", status="done", result="contents"), _step(2, "Sum it")]
    assert _review_vetoes(before, plan_ops.drop_step(before, 1)) == []


def test_plan_gate_records_vetoes_on_review_resume(monkeypatch):
    import nodes.plan_gate as gate
    from core.plan_ops import get_pause_controller

    before = [_step(1, "Compute 17 * 23"), _step(2, "Compute 41 * 7")]
    edited = plan_ops.drop_step(before, 2)
    monkeypatch.setattr(gate, "interrupt",
                        lambda payload: {"action": "continue", "plan": edited})
    pc = get_pause_controller()
    pc.request("user", "test review")
    try:
        out = gate.plan_gate_node({
            "plan": before, "messages": [HumanMessage(content="q")],
            "plan_vetoes": ["earlier veto"],
        })
    finally:
        pc.clear()
    assert out["plan"] == edited
    assert out["plan_vetoes"] == ["earlier veto", "Compute 41 * 7"]


# --- rectify: a user's single-step veto continues; a gate rejection still cancels -------------------

def test_rectify_continues_past_a_review_skipped_step():
    from nodes.rectify import rectify_node

    plan = [
        _step(1, "Compute 17 * 23", status="done", result="391"),
        _step(2, "Compute 41 * 7", status="skipped", result=plan_ops.review_stamp("skipped")),
        _step(3, "Compute 100 - 58"),
    ]
    out = rectify_node({"plan": plan, "replans": 0, "current_query": "q"})
    assert out["rectify"] is False
    assert "plan" not in out  # nothing cancelled — execution continues at step 3


def test_rectify_still_cancels_after_a_gate_rejection():
    from nodes.approval import DECLINE_TEXT
    from nodes.rectify import rectify_node

    plan = [
        _step(1, "Delete the file", tool="run_shell", status="skipped",
              result=DECLINE_TEXT + " run_shell(...)"),
        _step(2, "Confirm deletion"),
    ]
    out = rectify_node({"plan": plan, "replans": 0, "current_query": "q"})
    assert out["rectify"] is False
    assert all(s["status"] == "cancelled" for s in out["plan"] if s["step_id"] == 2)


# --- replan: never resurrect a vetoed step ----------------------------------------------------------

def test_replan_filters_vetoed_labels_and_leads_with_the_veto_block(monkeypatch):
    import nodes.replan as replan

    monkeypatch.setattr(replan, "planner_sys_msg", lambda: HumanMessage(content="sys"))
    monkeypatch.setattr(replan, "registered_tools", lambda: [])
    monkeypatch.setattr(replan, "plan_format", lambda tools: {})
    monkeypatch.setattr(replan, "structured", lambda *a, **k: object())
    monkeypatch.setattr(replan, "to_steps", lambda draft: [
        _step(1, "Compute 41 * 7"),          # the exact resurrection — must drop
        _step(2, "Write the summary"),
    ])
    state = {"plan": [], "replans": 0, "reasoning": "fix it",
             "plan_vetoes": ["Compute 41 * 7"], "current_query": "q"}
    out = replan.replan_node(state)
    labels = [s["label"] for s in out["plan"]]
    assert labels == ["Write the summary"]
    # and the revision instruction the planner sees LEADS with the veto framing
    assert "REMOVED these steps" in replan._revision_instruction(state)
    assert "Compute 41 * 7" in replan._revision_instruction(state)


# --- the shared prompt block ------------------------------------------------------------------------

def test_vetoes_block_formats_and_empties():
    assert vetoes_block({"plan_vetoes": []}) == ""
    assert vetoes_block({}) == ""
    block = vetoes_block({"plan_vetoes": ["Compute 41 * 7", "  ", "Send the email"]})
    assert "deliberately out of scope" in block
    assert "- Compute 41 * 7" in block and "- Send the email" in block
    assert "-  " not in block  # blank entries drop


# ── the pause controller's two slots (transplanted from the engine isolate) ─────────────────
#
# A pause is a request to INTERRUPT; a steer is a request to adjust WITHOUT interrupting.
# With both sharing one slot, a steer typed after Esc-pause (before the next step boundary)
# overwrote the pause: the user saw the ⏸ acknowledgement and never got the editor. Steers now
# QUEUE and plan_gate drains them only when no pause is outstanding — the pause outranks the
# steer, and the path to interrupt() evaluates identically on both LangGraph passes.


def test_a_steer_does_not_overwrite_a_pending_pause():
    from core.plan_ops import PauseController

    c = PauseController()
    c.request("user", "review requested")
    c.request("steer", "also save it")
    assert c.pending()
    assert c.peek().source == "user"
    assert c.steers_pending()


def test_a_queued_steer_survives_the_pause_being_consumed():
    from core.plan_ops import PauseController

    c = PauseController()
    c.request("user", "review")
    c.request("steer", "also save it")
    c.clear()  # the interrupt resolved
    assert not c.pending()
    assert [r.reason for r in c.take_steers()] == ["also save it"]
    assert not c.steers_pending()


def test_peek_shows_a_queued_steer_only_when_no_pause_is_pending():
    from core.plan_ops import PauseController

    c = PauseController()
    c.request("steer", "go deeper")
    assert c.peek().source == "steer" and not c.pending()
    c.reset()
    assert c.peek() is None and not c.steers_pending()


def test_plan_gate_takes_the_pause_path_while_a_steer_is_queued(monkeypatch):
    """The determinism contract: with a pause outstanding the node reaches interrupt() — the
    queued steer is left for the next boundary, never applied instead of the review."""
    from core.plan_ops import get_pause_controller
    from nodes import plan_gate as pg

    c = get_pause_controller()
    c.reset()
    c.request("user", "review")
    c.request("steer", "make it shorter")
    seen = {}

    def fake_interrupt(payload):
        seen["payload"] = payload
        return {"action": "go", "plan": payload["plan"]}

    monkeypatch.setattr(pg, "interrupt", fake_interrupt)
    plan = [{"step_id": 1, "label": "Write the draft to out.txt", "status": "pending",
             "intended_tool": "write_file", "result": None, "needs_resolution": False}]
    out = pg.plan_gate_node({"messages": [HumanMessage("q")], "plan": plan, "iteration": 0})
    assert seen["payload"]["type"] == "plan_review"
    assert "rectify" not in out                 # the steer was NOT applied on this pass
    assert c.steers_pending()                    # …it waits for the next boundary
    out2 = pg.plan_gate_node({"messages": [HumanMessage("q")], "plan": plan, "iteration": 0})
    assert out2["rectify"] is True and "make it shorter" in out2["reasoning"]
    c.reset()


def test_two_queued_steers_are_applied_together_oldest_first(monkeypatch):
    from core.plan_ops import get_pause_controller
    from nodes import plan_gate as pg

    c = get_pause_controller()
    c.reset()
    c.request("steer", "first")
    c.request("steer", "second")
    out = pg.plan_gate_node({"messages": [HumanMessage("q")], "plan": [], "iteration": 0})
    assert out["reasoning"].index("first") < out["reasoning"].index("second")
    assert "first" in out["messages"][0].content and "second" in out["messages"][0].content
    c.reset()
