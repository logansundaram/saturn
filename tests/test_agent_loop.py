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


# ── Task 3: the agent node ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_pause():
    from core.pause import get_pause_controller

    get_pause_controller().reset()
    yield
    get_pause_controller().reset()


def _state(msgs, **kw):
    s = {"messages": msgs, "current_query": str(msgs[0].content) if msgs else "", "context": "",
         "plan": [], "iteration": 0, "tools_called": [], "tool_results": [],
         "documents_retrieved": [], "tool_events": [], "gate_events": []}
    s.update(kw)
    return s


def _round(name, args, cid, content="hello", status="done"):
    return [AIMessage(content="", tool_calls=[_call(name, args, cid)]),
            ToolMessage(content=content, tool_call_id=cid, name=name,
                        additional_kwargs={"saturn_status": status})]


def test_agent_answers_directly_with_trailers(monkeypatch):
    from nodes import agent

    seen = {}

    def fake(llm_input, *, tools):
        seen["tools"] = tools
        seen["input"] = llm_input
        return AIMessage(content="42", response_metadata={
            "eval_count": 10, "eval_duration": 1e9, "prompt_eval_count": 500})

    monkeypatch.setattr(agent, "_generate", fake)
    st = _state([HumanMessage(content="q")], context_stable="STABLE", context_dynamic="DYN",
                tool_results=["read_file(file_path='a.txt') -> hello"])
    out = agent.agent_node(st)
    assert seen["tools"] is True
    assert seen["input"][0].content == agent.agent_sys_msg().content
    assert seen["input"][1].content == "STABLE"
    assert "DYN" in seen["input"][2].content and "q" in seen["input"][2].content
    final = out["messages"][-1]
    assert final.content.startswith("42") and "Sources:" in final.content and "read_file" in final.content
    assert out["iteration"] == 1 and out["tok_per_sec"] == 10.0 and out["context_tokens"] == 500
    assert agent.route_after_agent({"messages": out["messages"]}) == "end"


def test_agent_prompt_keeps_prior_history_before_the_request(monkeypatch):
    from nodes import agent

    seen = {}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: seen.setdefault("input", i) and AIMessage(content="ok"))
    msgs = [HumanMessage(content="first"), AIMessage(content="one"), HumanMessage(content="second")]
    agent.agent_node(_state(msgs, context_stable="S"))
    contents = [m.content for m in seen["input"]]
    assert contents[1:] == ["S", "first", "one", "second"]


def test_agent_emits_tool_calls_to_approval(monkeypatch):
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(
        content="", tool_calls=[_call("read_file", {"path": "a.txt"})]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    ai = out["messages"][-1]
    assert ai.tool_calls[0]["args"] == {"file_path": "a.txt"}  # coerced onto the real schema
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_hygiene_unknown_tool_and_missing_args(monkeypatch):
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(
        content="", tool_calls=[_call("nope", {}, "a"), _call("read_file", {}, "b")]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    tms = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tms] == ["a", "b"]
    assert "unknown tool" in tms[0].content and "read_file(file_path=" in tms[1].content
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_hygiene_malformed_call_gets_a_schema_hint(monkeypatch):
    """A small model's tool call whose arguments were not valid JSON arrives as an
    invalid_tool_call (LangChain refuses non-dict args on tool_calls): refused with the
    schema hint, routed back to the model, never crashed on."""
    from nodes import agent

    bad = {"name": "read_file", "args": "a.txt", "id": "c1", "error": "not json"}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(
        content="", invalid_tool_calls=[bad]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    tm = out["messages"][-1]
    assert isinstance(tm, ToolMessage) and tm.tool_call_id == "c1"
    assert "read_file(file_path=" in tm.content
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_declined_repeat_is_auto_declined(monkeypatch):
    from nodes import agent
    from nodes.approval import DECLINE_TEXT

    args = {"file_path": "x", "content": "y"}
    prior = [HumanMessage(content="q")] + _round("write_file", args, "c1", DECLINE_TEXT, "skipped")
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(
        content="", tool_calls=[_call("write_file", args, "c2")]))
    out = agent.agent_node(_state(prior))
    assert out["messages"][-1].content == agent.ALREADY_DECLINED_TEXT
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_stall_refuses_third_identical_call(monkeypatch):
    from nodes import agent

    args = {"file_path": "a"}
    prior = [HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(
        content="", tool_calls=[_call("read_file", args, "c3")]))
    out = agent.agent_node(_state(prior))
    assert out["messages"][-1].content == agent.STALL_TEXT
    # a second identical call is still allowed
    out2 = agent.agent_node(_state([HumanMessage(content="q")] + _round("read_file", args, "c1")))
    assert agent.route_after_agent({"messages": out2["messages"]}) == "approval"


def test_iteration_cap_answers_without_tools(monkeypatch):
    from config import get_config
    from nodes import agent

    seen = {}

    def fake(llm_input, *, tools):
        seen["tools"] = tools
        seen["last"] = llm_input[-1].content
        return AIMessage(content="partial")

    monkeypatch.setattr(agent, "_generate", fake)
    prior = [HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1", "err", "error")
    out = agent.agent_node(_state(prior, iteration=get_config().max_iterations))
    assert seen["tools"] is False and seen["last"] == agent.BUDGET_NOTE
    final = out["messages"][-1]
    assert final.content.startswith("partial")
    assert agent.INCIDENTS_NOTE_HEADER in final.content and "read_file" in final.content


def test_steer_is_injected_before_the_call(monkeypatch):
    from core.pause import get_pause_controller
    from core.state import STEER_PREFIX
    from nodes import agent

    get_pause_controller().request("steer", "use km")
    seen = {}

    def fake(llm_input, *, tools):
        seen["input"] = llm_input
        return AIMessage(content="ok")

    monkeypatch.setattr(agent, "_generate", fake)
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    steer = [m for m in out["messages"] if isinstance(m, HumanMessage)]
    assert steer and steer[0].content == f"{STEER_PREFIX} use km"
    assert any(getattr(m, "content", "") == f"{STEER_PREFIX} use km" for m in seen["input"])


def test_pause_interrupt_continue_steer_abort(monkeypatch):
    from core.pause import get_pause_controller
    from core.state import STEER_PREFIX
    from nodes import agent

    c = get_pause_controller()
    payloads = []

    def fake_interrupt(v):
        payloads.append(v)
        return fake_interrupt.reply

    monkeypatch.setattr(agent, "interrupt", fake_interrupt)
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools: AIMessage(content="ans"))
    step = {"step_id": 1, "label": "x", "status": "pending", "intended_tool": None,
            "result": None, "needs_resolution": False}

    c.request("user", "esc")
    fake_interrupt.reply = {"action": "continue"}
    out = agent.agent_node(_state([HumanMessage(content="q")], plan=[step]))
    assert payloads[0]["type"] == "pause" and payloads[0]["plan"][0]["label"] == "x"
    assert out["messages"][-1].content.startswith("ans") and not c.pending()

    c.request("user", "esc")
    fake_interrupt.reply = {"action": "steer", "text": "shorter"}
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    assert any(isinstance(m, HumanMessage) and m.content == f"{STEER_PREFIX} shorter"
               for m in out["messages"])

    c.request("user", "esc")
    fake_interrupt.reply = {"action": "abort"}
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    assert out["messages"][-1].content.startswith(agent.ABORT_TEXT)
    assert agent.route_after_agent({"messages": out["messages"]}) == "end"


def test_route_after_agent_mixed_hygiene_goes_to_approval():
    from nodes import agent

    msgs = [HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[_call("nope", {}, "a"),
                                              _call("read_file", {"file_path": "x"}, "b")]),
            ToolMessage(content="err", tool_call_id="a", name="nope")]
    assert agent.route_after_agent({"messages": msgs}) == "approval"


# ── Task 4: graph, turn driver, approval route ───────────────────────────────────────────────


def test_graph_has_four_nodes(isolated_paths, monkeypatch):
    from app import graph

    # DB_PATH is resolved at import; point it at the isolated tmp so the checkpointer opens there.
    (isolated_paths / "database").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(graph, "DB_PATH", str(isolated_paths / "database" / "db.sqlite"))
    nodes = set(graph.build_agent().get_graph().nodes)
    assert {"ground", "agent", "approval", "tools"} <= nodes
    assert not ({"plan", "quick", "execute", "rectify", "replan", "synthesize", "plan_gate",
                 "update_plan", "answer_gate"} & nodes)


def test_approval_rejected_batch_routes_to_agent(monkeypatch):
    from nodes import approval

    monkeypatch.setattr(approval, "interrupt", lambda v: False)
    call = _call("write_file", {"file_path": "a", "content": "b"})
    cmd = approval.approval_node({"messages": [HumanMessage(content="q"),
                                               AIMessage(content="", tool_calls=[call])], "plan": []})
    assert cmd.goto == "agent"
    assert cmd.update["messages"][0].additional_kwargs["saturn_status"] == "skipped"


def test_approval_ignores_calls_the_agent_already_answered(monkeypatch):
    from nodes import approval

    prompted = []
    monkeypatch.setattr(approval, "interrupt", lambda v: prompted.append(v) or True)
    msgs = [HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[_call("write_file", {"file_path": "a", "content": "b"}, "w"),
                                              _call("nope", {}, "n")]),
            ToolMessage(content="err", tool_call_id="n", name="nope")]
    cmd = approval.approval_node({"messages": msgs, "plan": []})
    assert cmd.goto == "tools" and [c["id"] for c in prompted[0]["tool_calls"]] == ["w"]


def test_run_turn_streams_agent_tokens_only():
    from langchain.messages import AIMessageChunk

    from app import turn

    class G:
        def stream(self, *a, **k):
            yield ("messages", (AIMessageChunk(content="hi"), {"langgraph_node": "agent"}))
            yield ("messages", (AIMessageChunk(content="no"), {"langgraph_node": "tools"}))
            yield ("updates", {"agent": {"iteration": 1}})

        def get_state(self, config):
            return SimpleNamespace(next=(), values={"messages": []}, tasks=[])

    toks, ups = [], []
    turn.run_turn(G(), {}, {}, approver=lambda v: True, on_update=lambda n, d: ups.append(n),
                  on_token=lambda t, lp=None: toks.append(t))
    assert toks == ["hi"] and ups == ["agent"]


def test_on_update_discards_a_streamed_preamble_before_a_tool_call():
    from app.turn import _make_on_update

    class Answer:
        started = True

        def __init__(self):
            self.discarded = 0

        def discard(self):
            self.discarded += 1

    class Tracer:
        def log_event(self, *a):
            pass

    a = Answer()
    on_update = _make_on_update(Tracer(), 1, show_ui=False, answer=a)
    on_update("agent", {"messages": [AIMessage(content="let me look",
                                                tool_calls=[_call("read_file", {"file_path": "x"})])]})
    assert a.discarded == 1
    on_update("agent", {"messages": [AIMessage(content="the answer")]})
    assert a.discarded == 1


def test_response_stream_discard_forgets_the_preamble(monkeypatch):
    import importlib

    r = importlib.import_module("tui.ui.response")  # the package re-exports a same-named function
    monkeypatch.setattr(r, "_RICH", False, raising=False)
    s = r.ResponseStream()
    s.feed("let me")
    assert s.started
    s.discard()
    assert not s.started and "".join(s._chars) == ""
