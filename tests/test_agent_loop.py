"""
The v2 loop (2026-09-27; spec docs/superpowers/specs/2026-09-27-v2-react-loop-design.md):
the leaf modules it needs, the `plan` tool, the agent node's deterministic guards and answer
trailers, the graph wiring, the turn driver's stream filter, the prime lineage, the rail and the
pause prompt. Offline: `nodes.agent._generate` is the one model seam and every test replaces it.
"""

from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage


# ── Task 1: leaf modules ─────────────────────────────────────────────────────────────────────


def test_pause_controller_lives_in_core_pause():
    from core.pause import PauseController, get_pause_controller

    c = get_pause_controller()
    assert isinstance(c, PauseController)
    c.reset()
    c.request("steer", "use metric units")
    assert not c.pending()
    assert [r.reason for r in c.take_steers()] == ["use metric units"]
    c.request("user", "esc")
    assert c.pending() and c.peek().source == "user"
    c.clear()
    assert not c.pending()


def test_grounding_parts_treats_old_context_as_stable():
    from core.state import grounding_parts

    assert grounding_parts({"context": "  old  "}) == ("old", "")
    assert grounding_parts({"context_stable": "s", "context_dynamic": "d"}) == ("s", "d")


def test_agent_task_is_think_off_with_payload_bound(monkeypatch):
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "m")
    kw = llms.invoke_kwargs(None, 0.0, task="agent")
    assert kw["reasoning"] is False and kw["options"]["num_predict"] == 4096


def test_think_flag_rides_invoke_kwargs_with_its_budget(monkeypatch):
    """A thinking pass sends `reasoning=True` and widens num_predict by runtime.think_budget
    (thinking tokens count against the bound; the answer must still fit after them)."""
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "m")
    _think_cfg(monkeypatch, think_budget=1000)
    kw = llms.invoke_kwargs(None, 0.0, task="agent", think=True)
    assert kw["reasoning"] is True and kw["options"]["num_predict"] == 4096 + 1000


def _think_cfg(monkeypatch, **runtime):
    from config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "runtime": {**cfg._data.get("runtime", {}), **runtime}})


def test_first_pass_never_thinks(monkeypatch):
    from nodes import agent
    _think_cfg(monkeypatch, think="adaptive")
    seen = []
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(content="a"))
    agent.agent_node(_state([HumanMessage(content="q")], plan=[{"step_id": 1}]))
    assert seen == [False]


def test_think_follows_an_error_in_the_latest_round(monkeypatch):
    """Thinking is spent on evidence, not on pass count (2026-09-29): a pass thinks only when
    the tool round just before it had an error (a tool failure or a hygiene refusal), because
    that is where the model needs a new approach. A clean round, a plan, a declined or blocked
    call (whose next move is already known: say it was not done), a high pass count, or an
    error further back in the turn all leave the pass think-off."""
    from nodes import agent
    _think_cfg(monkeypatch, think="adaptive")
    seen = []
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(content="a"))
    q = [HumanMessage(content="q")]
    clean = q + _round("read_file", {"file_path": "a"}, "c1")
    failed = q + _round("read_file", {"file_path": "a"}, "c1", "err", "error")
    cases = [
        (clean, {}, False),
        (clean, {"plan": [{"step_id": 1, "label": "x", "status": "pending"}]}, False),
        (failed, {}, True),
        (q + _round("write_file", {"file_path": "a"}, "c1", "no", "skipped"), {}, False),
        (q + _round("web_search", {"query": "a"}, "c1", "blocked", "blocked"), {}, False),
        (clean, {"iteration": 8}, False),
        (failed + _round("search_files", {"pattern": "a"}, "c2"), {}, False),
    ]
    for msgs, extra, _want in cases:
        agent.agent_node(_state(msgs, **{"iteration": 1, **extra}))
    assert seen == [want for _m, _e, want in cases]


def test_an_error_anywhere_in_a_batch_or_behind_a_steer_counts(monkeypatch):
    """The latest round is every ToolMessage after the last tool-calling message — one failed
    call in a two-call batch is evidence — and a steer note the user typed after it does not
    hide it."""
    from core.state import STEER_PREFIX
    from nodes import agent
    _think_cfg(monkeypatch, think="adaptive")
    seen = []
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(content="a"))
    batch = [HumanMessage(content="q"),
             AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "a"}, "c1"),
                                               _call("read_file", {"file_path": "b"}, "c2")]),
             ToolMessage(content="ok", tool_call_id="c1", name="read_file",
                         additional_kwargs={"saturn_status": "done"}),
             ToolMessage(content="err", tool_call_id="c2", name="read_file",
                         additional_kwargs={"saturn_status": "error"})]
    agent.agent_node(_state(batch, iteration=1))
    agent.agent_node(_state(batch + [HumanMessage(content=STEER_PREFIX + " use b2")], iteration=1))
    assert seen == [True, True]


def test_an_empty_thinking_pass_is_rerun_without_thinking(monkeypatch):
    """On a pass whose right move is a short answer, qwen3.5 in thinking mode writes the answer
    inside its reasoning and emits no content (reproduced on the 4b and 9b, 2026-09-29). A
    thinking pass that returns neither text nor a call is run ONCE more think-off; a pass
    that produced a call, or a think-off pass, is never rerun."""
    from nodes import agent
    _think_cfg(monkeypatch, think="adaptive")
    failed = [HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1", "err", "error")

    seen = []
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(
        content="" if think else "It is not there."))
    out = agent.agent_node(_state(failed, iteration=1))
    assert seen == [True, False]
    assert out["messages"][-1].content.startswith("It is not there.")

    seen.clear()
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(
        content="", tool_calls=[_call("search_files", {"pattern": "a"}, "c2")]))
    agent.agent_node(_state(failed, iteration=1))
    assert seen == [True]

    seen.clear()
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append(think) or AIMessage(content=""))
    agent.agent_node(_state([HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1"), iteration=1))
    assert seen == [False]


def test_capped_pass_and_the_off_knob_never_think(monkeypatch):
    from config import get_config
    from nodes import agent
    seen = []
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.append((tools, think)) or AIMessage(content="a"))
    hard = [HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1", "err", "error")
    _think_cfg(monkeypatch, think="adaptive")
    agent.agent_node(_state(hard, iteration=get_config().max_iterations - 1))
    _think_cfg(monkeypatch, think="off")
    agent.agent_node(_state(hard, iteration=5))
    _think_cfg(monkeypatch, think="on")
    agent.agent_node(_state([HumanMessage(content="q")]))
    assert seen == [(True, False), (True, False), (True, True)]


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
        {"step_id": 1, "label": "read both files", "status": "done", "result": "done"},
        {"step_id": 2, "label": "total", "status": "pending", "result": None},
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

    def fake(llm_input, *, tools, think=False):
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
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.setdefault("input", i) and AIMessage(content="ok"))
    msgs = [HumanMessage(content="first"), AIMessage(content="one"), HumanMessage(content="second")]
    agent.agent_node(_state(msgs, context_stable="S"))
    contents = [m.content for m in seen["input"]]
    assert contents[1:] == ["S", "first", "one", "second"]


def test_agent_emits_tool_calls_to_approval(monkeypatch):
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", {"path": "a.txt"})]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    ai = out["messages"][-1]
    assert ai.tool_calls[0]["args"] == {"file_path": "a.txt"}  # coerced onto the real schema
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_hygiene_unknown_tool_and_missing_args(monkeypatch):
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("nope", {}, "a"), _call("read_file", {}, "b")]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    tms = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tms] == ["a", "b"]
    assert "unknown tool" in tms[0].content and "read_file(file_path=" in tms[1].content
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_hygiene_refuses_arguments_that_belong_to_another_tool(monkeypatch):
    """`recall(fact=…, replaces=…)` is `remember`'s call under the wrong name (the 4b's
    supersession miss, 2026-09-29): recall has no required arguments, so coercion used to
    drop the foreign ones and RUN it. Refused with remember's shape instead; a call whose
    arguments fit its own tool is never second-guessed."""
    from core.tool_args import tool_for_args
    from nodes import agent

    assert tool_for_args("recall", {"fact": "I live in Berlin", "replaces": "#2"}) == "remember"
    assert tool_for_args("recall", {"query": "Berlin"}) is None
    assert tool_for_args("recall", {}) is None
    assert tool_for_args("read_file", {"path": "a.txt"}) is None  # an alias of its own arg
    assert tool_for_args("mcp_x_y", {"fact": "z"}) is None  # not our schema to police
    # A no-required-args tool's OPTIONAL argument under a common alias is its own call, never
    # redirected to a tool that happens to share the alias (list_directory → read_file).
    assert tool_for_args("list_directory", {"path": "src"}) is None
    assert tool_for_args("recall", {"text": "coffee"}) is None
    assert tool_for_args("recall", {"q": "coffee"}) is None
    # A stray argument is redirected only when it NAMES another tool's own argument and names
    # exactly one tool: a loose alias ("name", "query") fits half the registry and proves nothing.
    assert tool_for_args("list_directory", {"name": "src"}) is None
    assert tool_for_args("current_time", {"query": "now"}) is None
    assert tool_for_args("current_time", {"timezone": "UTC"}) is None

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("recall", {"fact": "I live in Berlin", "replaces": "#2"}, "a")]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    tm = out["messages"][-1]
    assert isinstance(tm, ToolMessage) and tm.tool_call_id == "a"
    assert "belong to remember" in tm.content and "remember(fact=" in tm.content
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_hygiene_malformed_call_gets_a_schema_hint(monkeypatch):
    """A small model's tool call whose arguments were not valid JSON arrives as an
    invalid_tool_call (LangChain refuses non-dict args on tool_calls): refused with the
    schema hint, routed back to the model, never crashed on."""
    from nodes import agent

    bad = {"name": "read_file", "args": "a.txt", "id": "c1", "error": "not json"}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
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
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("write_file", args, "c2")]))
    out = agent.agent_node(_state(prior))
    assert out["messages"][-1].content == agent.ALREADY_DECLINED_TEXT
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_stall_refuses_third_identical_call(monkeypatch):
    from nodes import agent

    args = {"file_path": "a"}
    prior = [HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", args, "c3")]))
    out = agent.agent_node(_state(prior))
    assert out["messages"][-1].content == agent.STALL_TEXT
    # a second identical call is still allowed
    out2 = agent.agent_node(_state([HumanMessage(content="q")] + _round("read_file", args, "c1")))
    assert agent.route_after_agent({"messages": out2["messages"]}) == "approval"


def test_a_stall_refusal_is_not_reported_as_a_failed_call():
    """The stall guard refuses a third identical call because the call ALREADY ran twice. The
    refusal is not that call's outcome: the answer must not tell the user a read that succeeded
    'could not be completed'."""
    from nodes import agent

    args = {"file_path": "a.txt"}
    turn = ([HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
            + _round("read_file", args, "c3", agent.STALL_TEXT, "error"))
    assert agent.incidents(turn) == []
    # a call that never succeeded is still an incident, stall refusal or not
    turn = ([HumanMessage(content="q")] + _round("read_file", args, "c1", "Error: gone", "error")
            + _round("read_file", args, "c2", "Error: gone", "error")
            + _round("read_file", args, "c3", agent.STALL_TEXT, "error"))
    assert agent.incidents(turn) == ["read_file(file_path='a.txt') — failed: Error: gone"]


def test_stall_guard_counts_repeats_since_something_changed(monkeypatch):
    """Edit, test, edit, test: the third `pytest -q` is not a loop — a write landed since the
    last one, so its result is new information. Only repeats with nothing changed in between
    are a stall."""
    from nodes import agent

    test = {"command": "pytest -q"}
    prior = ([HumanMessage(content="fix the test")]
             + _round("run_shell", test, "c1", "1 failed", "error")
             + _round("edit_file", {"file_path": "a.py", "old": "x", "new": "y"}, "c2", "edited")
             + _round("run_shell", test, "c3", "1 failed", "error")
             + _round("edit_file", {"file_path": "a.py", "old": "y", "new": "z"}, "c4", "edited"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("run_shell", test, "c5")]))
    out = agent.agent_node(_state(prior))
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"

    # a read in between changes nothing, and a FAILED write changed nothing either
    prior = ([HumanMessage(content="fix the test")]
             + _round("run_shell", test, "c1", "1 failed", "error")
             + _round("read_file", {"file_path": "a.py"}, "c2")
             + _round("run_shell", test, "c3", "1 failed", "error")
             + _round("edit_file", {"file_path": "a.py", "old": "q", "new": "z"}, "c4", "Error: not found", "error"))
    out = agent.agent_node(_state(prior))
    assert out["messages"][-1].content == agent.STALL_TEXT


def _cap_state(extra_passes=0, extra_msgs=()):
    """A turn standing at the cap: the next pass is number max_iterations (+ extra_passes)."""
    from config import get_config

    prior = [HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1") + list(extra_msgs)
    return _state(prior, iteration=get_config().max_iterations - 1 + extra_passes)


def test_at_the_cap_calls_are_refused_and_the_prompt_is_unchanged(monkeypatch):
    """The cap does not change the prompt. The pass at max_iterations is an ordinary bound pass
    — same system section, same catalog, so the daemon's prompt cache holds on the turn's
    largest prompt — and any call it emits is answered here with the budget refusal, which
    routes straight back for the answer."""
    from nodes import agent

    seen = {}

    def fake(llm_input, *, tools, think=False):
        seen.update(tools=tools, last=llm_input[-1], think=think)
        return AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "b"}, "c2"),
                                                 _call("web_search", {"query": "z"}, "c3")])

    monkeypatch.setattr(agent, "_generate", fake)
    out = agent.agent_node(_cap_state())
    assert seen["tools"] is True and seen["think"] is False
    assert isinstance(seen["last"], ToolMessage)             # no budget note appended
    refusals = out["messages"][-2:]
    assert [m.content for m in refusals] == [agent.BUDGET_TEXT] * 2
    assert [m.tool_call_id for m in refusals] == ["c2", "c3"]
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_a_text_answer_at_the_cap_is_just_the_answer(monkeypatch):
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(content="all done"))
    out = agent.agent_node(_cap_state())
    assert out["messages"][-1].content.startswith("all done")
    assert agent.INCIDENTS_NOTE_HEADER not in out["messages"][-1].content


def test_after_the_refusal_the_answer_names_what_was_not_run(monkeypatch):
    from nodes import agent

    refused = _round("read_file", {"file_path": "b"}, "c2", agent.BUDGET_TEXT, "error")
    seen = {}

    def fake(llm_input, *, tools, think=False):
        seen.update(tools=tools, think=think)
        return AIMessage(content="I read a; b was not read.")

    monkeypatch.setattr(agent, "_generate", fake)
    out = agent.agent_node(_cap_state(extra_passes=1, extra_msgs=refused))
    assert seen == {"tools": True, "think": False}           # still the cached prefix; never thinks
    final = out["messages"][-1].content
    assert final.startswith("I read a; b was not read.")
    assert agent.INCIDENTS_NOTE_HEADER in final
    assert "read_file(file_path='b') — not run: the turn's action budget was spent" in final


def test_a_model_that_keeps_calling_gets_the_tools_taken_away(monkeypatch):
    """The hard stop: a model that answers its refused calls with more calls is rerun once
    with tools unbound and the budget note — a real answer, never a stub, never another pass."""
    from nodes import agent

    refused = _round("read_file", {"file_path": "b"}, "c2", agent.BUDGET_TEXT, "error")
    seen = []

    def fake(llm_input, *, tools, think=False):
        seen.append((tools, llm_input[-1].content))
        if tools:
            return AIMessage(content="let me try", tool_calls=[_call("read_file", {"file_path": "c"}, "c3")])
        return AIMessage(content="partial")

    monkeypatch.setattr(agent, "_generate", fake)
    out = agent.agent_node(_cap_state(extra_passes=1, extra_msgs=refused))
    assert [t for t, _ in seen] == [True, False] and seen[1][1] == agent.BUDGET_NOTE
    final = out["messages"][-1]
    assert final.content.startswith("partial") and not getattr(final, "tool_calls", None)
    assert agent.route_after_agent({"messages": out["messages"]}) == "end"


def test_steer_is_injected_before_the_call(monkeypatch):
    from core.pause import get_pause_controller
    from core.state import STEER_PREFIX
    from nodes import agent

    get_pause_controller().request("steer", "use km")
    seen = {}

    def fake(llm_input, *, tools, think=False):
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
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(content="ans"))
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


def test_a_steer_survives_a_pause_that_lands_mid_pass(monkeypatch):
    """Esc arriving right after the steers were read: the interrupt re-runs the node from the
    top on resume, so steers drained BEFORE the interrupt were lost. They are drained only past
    it, so the correction reaches the model either way."""
    from core.pause import get_pause_controller
    from core.state import STEER_PREFIX
    from nodes import agent

    c = get_pause_controller()
    c.request("steer", "use km")
    real_take = c.take_steers

    def take_then_esc():  # the race: the pause lands between the drain and the pause check
        out = real_take()
        c.request("user", "esc")
        return out

    monkeypatch.setattr(c, "take_steers", take_then_esc)

    class _Interrupted(Exception):
        pass

    def fake_interrupt(v):
        if not fake_interrupt.resumed:
            raise _Interrupted  # LangGraph's interrupt: the node stops and re-runs on resume
        return {"action": "continue"}

    fake_interrupt.resumed = False
    monkeypatch.setattr(agent, "interrupt", fake_interrupt)
    seen = {}
    monkeypatch.setattr(agent, "_generate",
                        lambda i, *, tools, think=False: seen.setdefault("input", i) and AIMessage(content="ans"))
    state = _state([HumanMessage(content="q")])
    try:
        out = agent.agent_node(state)
    except _Interrupted:  # resumed: the node runs again from the top
        monkeypatch.setattr(c, "take_steers", real_take)
        fake_interrupt.resumed = True
        out = agent.agent_node(state)
    assert any(isinstance(m, HumanMessage) and m.content == f"{STEER_PREFIX} use km"
               for m in out["messages"])


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
    s = r.ResponseStream()
    s.feed("let me")
    assert s.started
    s.discard()
    assert not s.started and "".join(s._chars) == ""


# ── Task 5: the prime lineage ────────────────────────────────────────────────────────────────


def test_prime_lineage_is_the_bound_agent_prefix(monkeypatch):
    from core import prime
    from core.messages import agent_sys_msg

    sent = []

    class M:
        def bind_tools(self, tools):
            sent.append(("bound", len(tools)))
            return self

        def invoke(self, msgs, **kw):
            sent.append(("invoke", [m.content for m in msgs], kw.get("reasoning")))
            return AIMessage(content="")

    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr("core.llms.get_model", lambda: M())
    assert prime.prime("STABLE") == 1
    assert sent[0][0] == "bound" and sent[0][1] > 0
    assert sent[1][1] == [agent_sys_msg().content, "STABLE"] and sent[1][2] is True


# ── Task 7: the rail and the pause prompt ────────────────────────────────────────────────────


def _plain_rail(monkeypatch):
    import importlib

    base = importlib.import_module("tui.ui._base")
    trace = importlib.import_module("tui.ui.trace")
    base._trace_started = False
    base._t_last = None
    base._status = dict(base._status, node="", iteration=0, tools=0, tok_per_sec=0.0)
    return trace


def test_rail_agent_row_hidden_for_the_answer_shown_for_a_call(capsys, monkeypatch):
    trace = _plain_rail(monkeypatch)
    trace.show_node("agent", {"messages": [AIMessage(content="reading it",
                                                      tool_calls=[_call("read_file", {"file_path": "x"})])],
                              "iteration": 1})
    out = capsys.readouterr().out
    assert "agent" in out and "reading it" in out
    trace.show_node("agent", {"messages": [AIMessage(content="the answer")], "iteration": 2})
    assert "agent" not in capsys.readouterr().out


def test_rail_approval_row_only_for_a_human_decision(capsys, monkeypatch):
    trace = _plain_rail(monkeypatch)
    trace.show_node("approval", {})
    assert "approval" not in capsys.readouterr().out
    trace.show_node("approval", {"gate_events": [{"calls": [{"id": "1", "name": "write_file", "approved": False}],
                                                  "decision": "rejected", "quarantine": False, "step": None}]})
    out = capsys.readouterr().out
    assert "approval" in out and "rejected write_file" in out


def test_rail_tool_result_preview_shown_by_default(capsys, monkeypatch):
    trace = _plain_rail(monkeypatch)
    trace.show_node("tools", {"tool_events": [{"name": "read_file", "args": {"file_path": "x"},
                                               "result": "hello world", "dur": 0.01, "ok": True}]})
    assert "hello world" in capsys.readouterr().out


def test_pause_prompt_decisions(monkeypatch):
    import importlib

    from tui import ui

    p = importlib.import_module("tui.ui.prompt")
    monkeypatch.setattr(p, "ask", lambda *a, **k: "")
    assert ui.pause_prompt({"reason": "esc", "plan": []}) == {"action": "continue"}
    monkeypatch.setattr(p, "ask", lambda *a, **k: "use km")
    assert ui.pause_prompt({"reason": "esc", "plan": []}) == {"action": "steer", "text": "use km"}
    monkeypatch.setattr(p, "ask", lambda *a, **k: "q")
    assert ui.pause_prompt({"reason": "esc", "plan": []}) == {"action": "abort"}


# ── Task 8: /trace why renders agent passes ──────────────────────────────────────────────────


def test_trace_why_renders_agent_passes(isolated_paths, capsys):
    import json
    import sqlite3

    from commands import trace as tr
    from stores.trace import Tracer

    (isolated_paths / "database").mkdir(parents=True, exist_ok=True)
    db = str(isolated_paths / "database" / "db.sqlite")
    t = Tracer(db)
    run_id = t.start_run("th", "q")
    t.log_event(run_id, "agent", {"messages": [AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "x"})])]})
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO llm_calls (run_id, seq, ts, node, model, dur, prompt_tokens, output_tokens, "
                  "input, output, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (run_id, 1, "2026-09-27T00:00:00", "agent", "m", 0.1, 10, 5, "[]",
                   json.dumps({"content": "let me read it", "tool_calls": [_call("read_file", {"file_path": "x"})],
                               "reasoning": "the answer\n is in x"}),
                   "ok"))
        c.execute("INSERT INTO llm_calls (run_id, seq, ts, node, model, dur, prompt_tokens, output_tokens, "
                  "input, output, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (run_id, 2, "2026-09-27T00:00:01", "agent", "m", 0.1, 10, 5, "[]",
                   json.dumps({"content": "x holds 3 lines", "tool_calls": []}), "ok"))
    t.end_run(run_id, "ok", "done")
    tr._why(SimpleNamespace(db_path=db, state={}), [str(run_id)])
    out = capsys.readouterr().out
    assert "pass 1" in out and "read_file" in out and "let me read it" in out
    assert "thought: the answer is in x" in out  # the recorded reasoning, whitespace folded
    assert out.index("thought:") < out.index("→ chose to call")
    assert "pass 2: answered" in out and out.count("thought:") == 1
    assert "rectify" not in out


# ── final review fixes ───────────────────────────────────────────────────────────────────────


def test_config_template_keeps_the_runtime_block():
    """The template seeds every fresh install: its runtime knobs must live under `runtime:`
    (a lost header silently re-parents them under `paths:` and every runtime.* lookup falls
    back to code defaults — airgap included)."""
    import pathlib

    import yaml

    data = yaml.safe_load(pathlib.Path("config.default.yaml").read_text())
    rt = data["runtime"]
    for key in ("max_iterations", "auto_approve", "num_ctx", "llm_timeout", "keep_alive", "prime",
                "airgap", "quarantine", "citations", "grant_scope", "think", "think_budget"):
        assert key in rt, key
    assert "quick_path" not in rt and "think_after" not in rt
    assert all(isinstance(v, str) for v in data["paths"].values())


def test_ask_user_runs_alone(monkeypatch):
    """LangGraph re-executes tool_node from the top when ask_user's interrupt resumes, so any
    sibling call would run twice: hygiene keeps the question and refuses its siblings (and a
    second question) with a note to ask first."""
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(content="", tool_calls=[
        _call("read_file", {"file_path": "a"}, "r"), _call("ask_user", {"question": "which?"}, "q"),
        _call("ask_user", {"question": "and?"}, "q2")]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    tms = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
    assert set(tms) == {"r", "q2"} and all(agent.ASK_ALONE_TEXT == m.content for m in tms.values())
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_on_update_discards_when_hygiene_answered_a_call():
    from app.turn import _make_on_update

    class Answer:
        started = True
        discarded = 0

        def discard(self):
            self.discarded += 1

    class Tracer:
        def log_event(self, *a):
            pass

    a = Answer()
    on_update = _make_on_update(Tracer(), 1, show_ui=False, answer=a)
    on_update("agent", {"messages": [AIMessage(content="let me", tool_calls=[_call("nope", {}, "n")]),
                                     ToolMessage(content="err", tool_call_id="n", name="nope")]})
    assert a.discarded == 1


def test_a_parse_failure_retracts_what_the_failed_attempt_streamed(monkeypatch):
    """The failed attempt's tokens already reached the response region; the retry would stream
    a second attempt on top of them. The node emits a RETRACT on the custom stream first."""
    from langchain_core.exceptions import OutputParserException

    from nodes import agent

    written = []
    monkeypatch.setattr(agent, "get_stream_writer", lambda: written.append)
    attempts = []

    def flaky(llm_input, *, tools, think=False):
        attempts.append(1)
        if len(attempts) == 1:
            raise OutputParserException("bad tool json")
        return AIMessage(content="ok")

    monkeypatch.setattr(agent, "_generate", flaky)
    agent.agent_node(_state([HumanMessage(content="q")]))
    assert written == [agent.RETRACT]


def test_run_turn_hands_a_retract_to_on_retract():
    from app import turn
    from nodes.agent import RETRACT

    class G:
        def stream(self, *a, **k):
            assert "custom" in k.get("stream_mode", [])
            yield ("messages", (AIMessageChunk(content="garbled"), {"langgraph_node": "agent"}))
            yield ("custom", RETRACT)
            yield ("messages", (AIMessageChunk(content="ok"), {"langgraph_node": "agent"}))

        def get_state(self, config):
            return SimpleNamespace(next=(), values={"messages": []}, tasks=[])

    seen = []
    turn.run_turn(G(), {}, {}, approver=lambda v: True, on_token=seen.append,
                  on_retract=lambda: seen.append("<retract>"))
    assert seen == ["garbled", "<retract>", "ok"]


def test_generate_parse_failure_retries_once_then_answers_honestly(monkeypatch):
    from langchain_core.exceptions import OutputParserException

    from nodes import agent

    calls = []

    def flaky(llm_input, *, tools, think=False):
        calls.append(llm_input[-1].content)
        if len(calls) == 1:
            raise OutputParserException("bad tool json")
        return AIMessage(content="ok")

    monkeypatch.setattr(agent, "_generate", flaky)
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    assert out["messages"][-1].content.startswith("ok")
    assert len(calls) == 2 and calls[1] == agent.MALFORMED_NOTE

    def broken(llm_input, *, tools, think=False):
        raise OutputParserException("bad tool json")

    monkeypatch.setattr(agent, "_generate", broken)
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    assert out["messages"][-1].content.startswith(agent.MALFORMED_TEXT)
    assert agent.route_after_agent({"messages": out["messages"]}) == "end"

    def down(llm_input, *, tools, think=False):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(agent, "_generate", down)
    with pytest.raises(RuntimeError):
        agent.agent_node(_state([HumanMessage(content="q")]))


def test_incidents_note_uses_user_wording_and_dedupes(monkeypatch):
    from nodes import agent
    from nodes.approval import DECLINE_TEXT

    args = {"file_path": "x", "content": "y"}
    prior = ([HumanMessage(content="q")] + _round("write_file", args, "c1", DECLINE_TEXT, "skipped")
             + _round("write_file", args, "c2", agent.ALREADY_DECLINED_TEXT, "skipped")
             + _round("web_search", {"query": "z"}, "c3", "air-gap refused", "blocked"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(content="sorry"))
    out = agent.agent_node(_state(prior))
    text = out["messages"][-1].content
    assert text.count("write_file(") == 1 and "declined at the approval gate" in text
    assert "Do not retry" not in text and "blocked by the air-gap" in text


def test_incidents_note_omits_a_call_that_later_succeeded():
    """A failed call the model re-issued and that then RAN is done, not an incident — the note
    must not tell the user a write that happened did not (the ask-alone refusal, a transient
    web error retried). The call's LAST outcome decides."""
    from nodes import agent

    args = {"file_path": "x", "content": "y"}
    turn = ([HumanMessage(content="q")]
            + _round("write_file", args, "c1", agent.ASK_ALONE_TEXT, "error")
            + _round("write_file", args, "c2", "Created x", "done")
            + _round("web_search", {"query": "z"}, "c3", "Created", "done")
            + _round("web_search", {"query": "z"}, "c4", "Error: timeout", "error"))
    assert agent.incidents(turn) == ["web_search(query='z') — failed: Error: timeout"]


def test_cap_lands_on_the_max_iterations_pass(monkeypatch):
    from config import get_config
    from nodes import agent

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", {"file_path": "a"}, "c1")]))
    cap = get_config().max_iterations
    before = agent.agent_node(_state([HumanMessage(content="q")], iteration=cap - 2))
    at = agent.agent_node(_state([HumanMessage(content="q")], iteration=cap - 1))
    assert agent.route_after_agent({"messages": before["messages"]}) == "approval"
    assert agent.route_after_agent({"messages": at["messages"]}) == "agent"
    assert at["messages"][-1].content == agent.BUDGET_TEXT


def test_plan_call_is_not_a_source():
    from nodes.tools import tool_node

    out = tool_node({"messages": [HumanMessage(content="q"), AIMessage(content="", tool_calls=[
        _call("plan", {"steps": [{"label": "a"}]})])]})
    assert out["tool_results"] == [] and out["plan"]


def test_prior_answer_trailers_are_stripped_from_history(monkeypatch):
    from nodes import agent

    seen = {}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.setdefault("input", i) and AIMessage(content="ok"))
    earlier = ("42\n\n" + agent.INCIDENTS_NOTE_HEADER + "\n- read_file(x) — error: boom\n\n"
               "Sources:\n  [1] calculate(expression='6*7')")
    msgs = [HumanMessage(content="first"), AIMessage(content=earlier), HumanMessage(content="second")]
    agent.agent_node(_state(msgs))
    prior_ai = [m for m in seen["input"] if isinstance(m, AIMessage)]
    assert prior_ai and prior_ai[0].content == "42"


def test_issuing_message_walks_back_over_answered_calls():
    from core.state import issuing_message

    ai = AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"},
                                           {"name": "b", "args": {}, "id": "2"}])
    last, answered = issuing_message([HumanMessage("q"), ai, ToolMessage("x", tool_call_id="1")])
    assert last is ai and answered == {"1"}
    assert issuing_message([ToolMessage("x", tool_call_id="9")]) == (None, {"9"})


def test_coercion_maps_optional_argument_aliases():
    """The alias a small model emits for an optional argument lands on the real name, so
    list_directory(path='src') lists src rather than the workspace root."""
    from core.tool_args import coerce_args

    assert coerce_args("list_directory", {"path": "src"}) == {"directory": "src"}
    assert coerce_args("recall", {"text": "coffee"}) == {"query": "coffee"}
    assert coerce_args("find_files", {"pattern": "*.md", "dir": "notes"}) == {"pattern": "*.md", "directory": "notes"}
    assert coerce_args("search_files", {"pattern": "x", "path": "src"}) == {"pattern": "x", "directory": "src"}
