"""The positional/multiset plan-accounting invariant (CLAUDE.md gotcha #6): `update_plan_node`
and `state.unrun_planned_tools` must keep identical accounting — these tests pin both walkers
and cross-check them on shared scenarios."""

from state import unrun_planned_tools, active_step, steps_to_dicts, PlanStep
from node_registry.update_plan import update_plan_node


def _step(step_id, tool=None, status="pending", label=None):
    return {
        "step_id": step_id,
        "label": label or f"step {step_id}",
        "status": status,
        "intended_tool": tool,
    }


# ── state.unrun_planned_tools ─────────────────────────────────────────────────────────────────


def test_two_same_tool_steps_each_need_their_own_call():
    plan = [_step(1, "web_search"), _step(2, "web_search")]
    pending = unrun_planned_tools(plan, ["web_search"])
    assert [s["step_id"] for s in pending] == [2]
    assert unrun_planned_tools(plan, ["web_search", "web_search"]) == []


def test_done_step_consumes_its_call_first():
    """An already-done step's call must not mask a later same-tool step."""
    plan = [_step(1, "web_search", status="done"), _step(2, "web_search")]
    pending = unrun_planned_tools(plan, ["web_search"])
    assert [s["step_id"] for s in pending] == [2]


def test_steps_without_tools_and_terminal_steps_ignored():
    plan = [_step(1, None), _step(2, "calculate", status="skipped"), _step(3, "calculate")]
    pending = unrun_planned_tools(plan, [])
    assert [s["step_id"] for s in pending] == [3]


def test_empty_inputs():
    assert unrun_planned_tools([], []) == []
    assert unrun_planned_tools(None, None) == []


# ── state.active_step / steps_to_dicts ────────────────────────────────────────────────────────


def test_active_step_is_first_non_terminal():
    plan = [_step(1, status="done"), _step(2, status="skipped"), _step(3), _step(4)]
    assert active_step(plan)["step_id"] == 3
    assert active_step([_step(1, status="done")]) is None
    assert active_step([]) is None


def test_steps_to_dicts_renumbers_and_resets_status():
    steps = [
        PlanStep(step_id=7, label="a", status="active", intended_tool="web_search"),
        PlanStep(step_id=9, label="b"),
    ]
    out = steps_to_dicts(steps)
    assert [s["step_id"] for s in out] == [1, 2]
    assert all(s["status"] == "pending" for s in out)
    assert out[0]["intended_tool"] == "web_search"


# ── node_registry.update_plan ─────────────────────────────────────────────────────────────────


def test_update_plan_positional_credit():
    plan = [_step(1, "web_search"), _step(2, "web_search")]
    out = update_plan_node({"plan": plan, "tools_called": ["web_search"]})
    new = out["plan"]
    assert new[0]["status"] == "done"
    assert new[1]["status"] == "active"  # surfaced as the next step, not done
    # The two walkers agree: exactly the uncredited step is still unrun.
    assert [s["step_id"] for s in unrun_planned_tools(new, ["web_search"])] == [2]


def test_update_plan_does_not_mutate_input():
    plan = [_step(1, "web_search")]
    update_plan_node({"plan": plan, "tools_called": ["web_search"]})
    assert plan[0]["status"] == "pending"


def test_update_plan_fallback_advances_only_tool_steps():
    """A tool round that credited nothing advances the current step — but never a
    no-intended_tool step (the generic fallback plan must not complete off the first round)."""
    # Mismatched tool: the planner guessed web_search, the agent used read_file.
    plan = [_step(1, "web_search")]
    out = update_plan_node({"plan": plan, "tools_called": ["read_file"]})
    assert out["plan"][0]["status"] == "done"
    # A no-tool step is NOT advanced by the fallback.
    plan = [_step(1, None)]
    out = update_plan_node({"plan": plan, "tools_called": ["read_file"]})
    assert out["plan"][0]["status"] == "active"  # surfaced, not completed


def test_update_plan_empty_plan_noop():
    assert update_plan_node({"plan": [], "tools_called": ["x"]}) == {}


def test_walkers_agree_on_fully_credited_plan():
    plan = [_step(1, "web_search"), _step(2, "calculate"), _step(3, None)]
    called = ["web_search", "calculate"]
    out = update_plan_node({"plan": plan, "tools_called": called})
    assert unrun_planned_tools(out["plan"], called) == []
