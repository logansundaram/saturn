"""
Unit tests for the brittle pure-logic core of the loop — the parts whose bugs don't crash but
silently corrupt behaviour (plan accounting, history compaction, observation clamping).

Deliberately dependency-light: every function under test is pure (or near-pure), so these need no
Ollama, no network, no checkpointer. Runnable two ways:
    python tests/test_core.py     # standalone, no pytest required
    pytest tests/                 # if pytest is installed

Coverage maps to the fixes in the brittleness pass:
  - unrun_planned_tools / update_plan_node : multi-step same-tool plans no longer collapse (#4),
    the progress fallback can't mis-credit (#5), update_plan doesn't mutate in place (#6).
  - _compact_history : older turns collapse but the most recent scratchpad is retained.
  - _clamp_observation : large tool output can't overflow the context window.
  - planner tool catalog : built from the live registry (no drift when tools are added).
"""

import os
import sys

# Repo root on the path so `import state` etc. resolve when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.messages import HumanMessage, AIMessage, ToolMessage

from state import unrun_planned_tools
from node_registry.update_plan import update_plan_node
from node_registry.tools import _clamp_observation, _MAX_OBSERVATION


def _step(step_id, status, tool):
    return {"step_id": step_id, "label": f"step {step_id}", "status": status, "intended_tool": tool}


# --- #4: multi-step same-tool plans (positional, not set-membership) -----------------------
def test_unrun_two_same_tool_one_call_leaves_second_pending():
    plan = [_step(1, "done", "web_search"), _step(2, "pending", "web_search")]
    pending = unrun_planned_tools(plan, ["web_search"])
    assert [s["step_id"] for s in pending] == [2], "second search must still be pending after one call"


def test_unrun_two_same_tool_two_calls_clears_both():
    plan = [_step(1, "done", "web_search"), _step(2, "pending", "web_search")]
    assert unrun_planned_tools(plan, ["web_search", "web_search"]) == []


def test_unrun_ignores_no_tool_and_terminal_steps():
    plan = [_step(1, "pending", None), _step(2, "skipped", "web_search"), _step(3, "pending", "calculate")]
    pending = unrun_planned_tools(plan, [])
    assert [s["step_id"] for s in pending] == [3]  # only the un-run tool step


# --- #4/#5/#6: update_plan positional advance, fallback, no in-place mutation ---------------
def test_update_plan_advances_one_step_per_call():
    plan = [_step(1, "active", "web_search"), _step(2, "pending", "web_search"), _step(3, "pending", "write_file")]
    out1 = update_plan_node({"plan": plan, "tools_called": ["web_search"]})["plan"]
    assert [(s["step_id"], s["status"]) for s in out1] == [(1, "done"), (2, "active"), (3, "pending")]
    out2 = update_plan_node({"plan": out1, "tools_called": ["web_search", "web_search"]})["plan"]
    assert [(s["step_id"], s["status"]) for s in out2] == [(1, "done"), (2, "done"), (3, "active")]


def test_update_plan_does_not_mutate_input():
    plan = [_step(1, "active", "web_search"), _step(2, "pending", "web_search")]
    before = [dict(s) for s in plan]
    update_plan_node({"plan": plan, "tools_called": ["web_search"]})
    assert plan == before, "update_plan must work on a copy, not mutate state in place"


def test_update_plan_fallback_advances_active_when_no_match():
    # A tool round happened but matched no planned intended_tool -> advance the active step only.
    plan = [_step(1, "active", "write_file"), _step(2, "pending", None)]
    out = update_plan_node({"plan": plan, "tools_called": ["calculate"]})["plan"]
    assert out[0]["status"] == "done" and out[1]["status"] == "active"


# --- reject -> skip the rejected step so the plan can't re-demand it (no re-approve loop) --------
def test_rejected_step_is_skipped_so_plan_advances():
    from node_registry.approval import _skip_rejected_steps

    plan = [_step(1, "active", "write_file"), _step(2, "pending", "web_search")]
    out = _skip_rejected_steps(plan, ["write_file"])
    assert out[0]["status"] == "skipped", "the rejected tool's step retires"
    assert out[1]["status"] == "pending", "later, un-rejected work is untouched"
    # The skipped step is no longer un-run work, so route_after_agent won't nudge for it.
    assert [s["step_id"] for s in unrun_planned_tools(out, [])] == [2]


def test_rejected_step_skip_is_positional_for_same_tool():
    from node_registry.approval import _skip_rejected_steps

    # Two same-tool steps, one rejection -> only the first non-terminal one retires.
    plan = [_step(1, "active", "write_file"), _step(2, "pending", "write_file")]
    out = _skip_rejected_steps(plan, ["write_file"])
    assert [(s["step_id"], s["status"]) for s in out] == [(1, "skipped"), (2, "pending")]


def test_rejected_step_skip_falls_back_to_active_when_no_tool_match():
    from node_registry.approval import _skip_rejected_steps

    # The agent called a tool the planner didn't anticipate; skip the active step it was driving.
    plan = [_step(1, "active", "write_file"), _step(2, "pending", None)]
    out = _skip_rejected_steps(plan, ["some_other_tool"])
    assert out[0]["status"] == "skipped" and out[1]["status"] == "pending"


def test_rejected_step_skip_does_not_mutate_input():
    from node_registry.approval import _skip_rejected_steps

    plan = [_step(1, "active", "write_file")]
    before = [dict(s) for s in plan]
    _skip_rejected_steps(plan, ["write_file"])
    assert plan == before, "must work on a copy, not mutate state in place"


# --- _compact_history: keep the most recent scratchpad, collapse older turns ----------------
def test_compact_history_keeps_recent_scratchpad_drops_old():
    from agent import _compact_history

    history = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "a"}]),
        ToolMessage(content="old-result", tool_call_id="a", name="web_search"),
        AIMessage(content="answer1"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "b"}]),
        ToolMessage(content="recent-result", tool_call_id="b", name="web_search"),
        AIMessage(content="answer2"),
    ]
    kept = _compact_history(history)  # keep_recent_turns=1
    blob = "|".join(str(m.content) for m in kept)
    assert "old-result" not in blob, "old turn's tool output should be dropped"
    assert "recent-result" in blob, "most recent turn's scratchpad must be retained"
    assert "q1" in blob and "answer1" in blob, "older turn collapses to its Q&A, not vanishes"
    # No orphaned tool-call AIMessage from the old turn survived.
    assert not any(getattr(m, "tool_calls", None) and "old" in str(m.content) for m in kept)


# --- _clamp_observation: large tool output can't blow the context window --------------------
def test_clamp_short_observation_untouched():
    s = "small result"
    assert _clamp_observation(s) == s


def test_clamp_long_observation_truncated_with_marker():
    s = "x" * (_MAX_OBSERVATION + 5000)
    out = _clamp_observation(s)
    assert len(out) < len(s)
    assert "truncated" in out
    assert out.startswith("x") and out.endswith("x")  # head + tail preserved


# --- planner catalog stays in sync with the live registry -----------------------------------
def test_planner_catalog_lists_every_registered_tool():
    import messages
    import registry

    catalog = messages._tool_catalog()
    for t in registry.tool:
        assert t.name in catalog, f"{t.name} missing from planner catalog (drift!)"


# --- #5: registration decorator keeps the registry views consistent -------------------------
def test_registry_views_consistent():
    import registry

    # Every registered tool has a risk tier, and tools_by_name covers the whole list.
    assert {t.name for t in registry.tool} == set(registry.tools_by_name)
    assert all(t.name in registry.TOOL_RISK for t in registry.tool)
    # Risk tiers are valid, and unknown tools fail safe to the strictest tier.
    from toolspec import RISK_TIERS

    assert all(v in RISK_TIERS for v in registry.TOOL_RISK.values())
    assert registry.risk_of("a-tool-that-does-not-exist") == "destructive"
    # The retrieval flag rode along with the tool that declared it.
    assert "search_knowledge_base" in registry.RETRIEVAL_TOOLS


# --- #8: model-presence normalization for the startup health check --------------------------
def test_model_present_normalizes_latest_tag():
    from llms import _model_present

    assert _model_present("qwen3.5:9b", {"qwen3.5:9b"})
    assert _model_present("foo", {"foo:latest"})        # implicit :latest
    assert _model_present("foo:latest", {"foo"})        # and the reverse
    assert not _model_present("missing:9b", {"other:9b"})


# --- #3: calculate tames float epsilon without capping real precision -----------------------
def test_calculate_tames_epsilon_keeps_precision():
    from tool_registry.calculator import calculate

    call = lambda e: calculate.invoke({"expression": e})
    assert call("672.34999999999999 + 0") == "672.35"   # epsilon artifact removed
    assert call("37.0") == "37"                          # whole-number float -> int
    assert call("2+3*4") == "14"
    assert call("1/3") == "0.333333333333"               # real precision preserved (not 0.3333)


# --- Tier 2 #7: LLM compaction folds old turns, keeps the recent one, and fails safe -----------
def test_compaction_folds_old_keeps_recent_turn():
    import compaction

    orig = compaction._llm_summary
    compaction._llm_summary = lambda older: f"SUMMARY({len(older)})"  # stub: offline + deterministic
    try:
        msgs = [
            HumanMessage("q1"), AIMessage("a1"),
            HumanMessage("q2"), AIMessage("a2"),
            HumanMessage("q3"), AIMessage("a3"),
        ]
        new, stats = compaction.summarize_messages(msgs)  # keep_recent_turns=1
        assert stats["summarized_turns"] == 2, "the two older turns fold"
        assert compaction.is_summary(new[0]), "folded turns become one summary message at the head"
        assert new[-2].content == "q3" and new[-1].content == "a3", "recent turn kept verbatim"
        # A prior summary chains forward (folded into the next) rather than accreting.
        chained = new + [HumanMessage("q4"), AIMessage("a4")]
        n2, _ = compaction.summarize_messages(chained)
        assert sum(compaction.is_summary(m) for m in n2) == 1, "summaries don't pile up"
        # Only the recent turn present -> nothing to fold.
        _, s2 = compaction.summarize_messages([HumanMessage("only"), AIMessage("a")])
        assert s2["summarized_turns"] == 0
    finally:
        compaction._llm_summary = orig


def test_compaction_llm_failure_leaves_history_intact():
    import compaction

    orig = compaction._llm_summary

    def boom(older):
        raise RuntimeError("model down")

    compaction._llm_summary = boom
    try:
        msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2")]
        new, stats = compaction.summarize_messages(msgs)
        assert new is msgs and stats["summarized_turns"] == 0, "a failed summary must not lose history"
    finally:
        compaction._llm_summary = orig


# --- Tier 2 #6: type-ahead queue + Esc steering (InputQueue char handling, no console) ----------
def test_typeahead_queues_enter_terminated_lines_fifo():
    import typeahead

    q = typeahead.InputQueue()
    for ch in "first":
        q._on_char(ch)
    q._on_char("\r")
    for ch in "second":
        q._on_char(ch)
    q._on_char("\n")
    assert q.pending()
    assert q.pop() == "first" and q.pop() == "second", "queue drains FIFO"
    assert q.pop() is None


def test_typeahead_blank_not_queued_and_backspace_edits():
    import typeahead

    q = typeahead.InputQueue()
    for ch in "   ":
        q._on_char(ch)
    q._on_char("\r")
    assert not q.pending(), "a blank line never queues"
    for ch in "abx":
        q._on_char(ch)
    q._on_char("\x08")  # backspace removes the x
    q._on_char("c")
    q._on_char("\r")
    assert q.pop() == "abc"


def test_escape_with_text_steers_empty_reviews():
    import interrupts
    import typeahead

    c = interrupts.get_pause_controller()
    c.clear()
    q = typeahead.InputQueue()
    for ch in "use the 2023 figures":
        q._on_char(ch)
    q._on_escape()
    req = c.peek()
    assert req.source == "steer" and req.reason == "use the 2023 figures"
    assert q._buffer == "", "the typed line is consumed as a steer, not left to queue"
    c.clear()
    q._on_escape()  # empty buffer
    assert c.peek().source == "user", "empty Esc asks for a plan-review pause"
    c.clear()


def test_plan_gate_injects_steer_and_consumes_request():
    import interrupts
    from node_registry.plan_gate import plan_gate_node

    c = interrupts.get_pause_controller()
    c.clear()
    c.request("steer", "focus on cost, not schedule")
    upd = plan_gate_node({"messages": [HumanMessage("q")], "plan": [], "iteration": 1})
    assert "messages" in upd, "a steer is injected as a message update"
    assert "focus on cost" in upd["messages"][0].content
    assert not c.pending(), "the steer request is consumed (won't re-inject next boundary)"


# --- Tier 2 #5: write_file diff preview (pure diff classification) ------------------------------
def test_write_diff_new_file_is_all_additions():
    from tui import ui

    rows, is_new, _hidden = ui._diff_lines("___does_not_exist___.txt", "alpha\nbeta\n", True)
    assert is_new
    assert [k for k, _ in rows] == ["hunk", "add", "add"]


def test_write_diff_overwrite_shows_delete_and_add():
    from config import get_config
    from tui import ui

    ws = get_config().path("workspace")
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / "___difftest___.txt"
    p.write_text("one\ntwo\n", encoding="utf-8")
    try:
        rows, is_new, _hidden = ui._diff_lines("___difftest___.txt", "one\nTWO\n", True)
        kinds = [k for k, _ in rows]
        assert not is_new
        assert "del" in kinds and "add" in kinds, "a changed line shows as a delete + an add"
    finally:
        p.unlink()


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
