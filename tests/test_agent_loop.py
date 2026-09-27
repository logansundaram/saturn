"""
The v2 loop (2026-09-27; spec docs/superpowers/specs/2026-09-27-v2-react-loop-design.md):
the leaf modules it needs, the `plan` tool, the agent node's deterministic guards and answer
trailers, the graph wiring, the turn driver's stream filter, the prime lineage, the rail and the
pause prompt. Offline: `nodes.agent._generate` is the one model seam and every test replaces it.
"""

from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ── Task 1: leaf modules ─────────────────────────────────────────────────────────────────────


def test_pause_controller_lives_in_core_pause():
    from core.pause import PauseController, get_pause_controller

    c = get_pause_controller()
    assert isinstance(c, PauseController)
    c.reset()
    c.request("steer", "use metric units")
    assert not c.pending() and c.steers_pending()
    assert [r.reason for r in c.take_steers()] == ["use metric units"]
    c.request("user", "esc")
    assert c.pending() and c.peek().source == "user"
    c.clear()
    assert not c.pending()


def test_grounding_parts_treats_old_context_as_stable():
    from core.context import grounding_parts

    assert grounding_parts({"context": "  old  "}) == ("old", "")
    assert grounding_parts({"context_stable": "s", "context_dynamic": "d"}) == ("s", "d")


def test_agent_task_is_think_off_with_payload_bound():
    from core import serving

    t = serving.task_of("agent")
    assert t.think is False and t.num_predict == 4096 and t.strict is False
    assert serving.task_for_role("tool_caller") == "agent"


def test_agent_sys_msg_is_stable_and_names_plan_tool():
    from core.messages import agent_sys_msg

    a, b = agent_sys_msg(), agent_sys_msg()
    assert isinstance(a, SystemMessage) and a.content == b.content
    assert "plan" in a.content and "data" in a.content.lower()


# ── Task 2: the plan tool ────────────────────────────────────────────────────────────────────


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def test_plan_tool_maps_onto_step_dicts():
    from tools.planning import plan, to_plan

    steps = to_plan([{"label": "read both files", "status": "done"},
                     {"label": "total", "status": "pending"}, {"label": "", "status": "x"}])
    assert steps == [
        {"step_id": 1, "label": "read both files", "status": "done", "intended_tool": None,
         "result": "done", "needs_resolution": False},
        {"step_id": 2, "label": "total", "status": "pending", "intended_tool": None,
         "result": None, "needs_resolution": False},
    ]
    assert "2 step" in plan.invoke({"steps": [{"label": "a"}, {"label": "b", "status": "done"}]})


def test_tool_node_writes_plan_state_from_plan_call():
    from nodes.tools import tool_node

    call = _call("plan", {"steps": [{"label": "read", "status": "done"}, {"label": "sum"}]})
    out = tool_node({"messages": [HumanMessage(content="q"), AIMessage(content="", tool_calls=[call])]})
    assert [s["label"] for s in out["plan"]] == ["read", "sum"]
    assert out["plan"][0]["status"] == "done" and out["plan"][1]["result"] is None
    assert isinstance(out["messages"][0], ToolMessage)
