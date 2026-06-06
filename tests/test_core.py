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
