"""
core/think.py — which passes think (spec docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md).
The decision is a pure function of the turn's messages, so it is tested as tables: the level
normaliser, the kind of step, and the level × kind × policy decision.
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from core import think
from core.state import STEER_PREFIX


def _cfg(monkeypatch, think_policy=None, **runtime):
    """Set runtime.* for one test. `think_policy` is not a setting: it points `auto` at a
    policy for the test (core.think.set_policy is what the benchmark's baseline run uses)."""
    from config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "runtime": {**cfg._data.get("runtime", {}), **runtime}})
    if think_policy is not None:
        assert think_policy in think.POLICIES
        monkeypatch.setattr(think, "_POLICY", think_policy)


def _call(name, cid):
    return {"name": name, "args": {}, "id": cid, "type": "tool_call"}


def _round(*calls):
    """One tool round: (name, status[, content]) per call."""
    ai = AIMessage(content="", tool_calls=[_call(c[0], f"c{i}") for i, c in enumerate(calls)])
    out = [ai]
    for i, c in enumerate(calls):
        out.append(ToolMessage(content=c[2] if len(c) > 2 else "ok", tool_call_id=f"c{i}",
                               name=c[0], additional_kwargs={"saturn_status": c[1]}))
    return out


Q = [HumanMessage(content="q")]
STEER = [HumanMessage(content=STEER_PREFIX + " use the other file")]


# ── the level ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, want", [
    ("fast", ("fast", True)), ("auto", ("auto", True)), ("deep", ("deep", True)),
    ("off", ("fast", True)), ("adaptive", ("auto", True)), ("on", ("deep", True)),
    # a bare `on` / `off` in config.yaml is a YAML boolean — the bug this normaliser closes
    (True, ("deep", True)), (False, ("fast", True)),
    ("true", ("deep", True)), ("False", ("fast", True)), (" Fast ", ("fast", True)),
    (None, ("auto", True)),
    ("yes", ("auto", False)), (2, ("auto", False)), ("adaptve", ("auto", False)), ("", ("auto", False)),
])
def test_normalise_reads_every_spelling_and_flags_the_rest(raw, want):
    assert think.normalise(raw) == want


def test_level_prefers_the_turns_override(monkeypatch):
    _cfg(monkeypatch, think="fast")
    assert think.level() == "fast" and think.level({}) == "fast"
    assert think.level({"think_level": "deep"}) == "deep"
    assert think.level({"think_level": "nonsense"}) == "fast"


def test_an_unrecognised_setting_is_reported_and_runs_as_its_default(monkeypatch):
    _cfg(monkeypatch, think="always")
    assert think.level() == "auto"
    (line,) = think.problems()
    assert "'always'" in line
    _cfg(monkeypatch, think=True)
    assert think.problems() == [] and think.level() == "deep"


def test_auto_is_one_rule_and_not_a_setting(monkeypatch):
    """`auto` thinks before acting (`act`). There is no runtime.think_policy: a line a user
    left in config.yaml changes nothing. `recover` exists only as the benchmark's baseline."""
    from pathlib import Path

    import yaml

    from config import get_config
    assert think.policy() == "act" and think.POLICIES == ("act", "recover")
    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "runtime": {**cfg._data.get("runtime", {}),
                                                                "think_policy": "recover"}})
    assert think.policy() == "act" and think.problems() == []
    template = yaml.safe_load((Path(__file__).parent.parent / "config.default.yaml").read_text())
    assert "think_policy" not in template["runtime"]
    assert template["runtime"]["think"] == "auto" and template["runtime"]["think_budget"] == 1024
    with pytest.raises(ValueError):
        think.set_policy("decide-draft")
    assert think.policy() == "act"


def test_budget_reads_the_config_and_survives_garbage(monkeypatch):
    _cfg(monkeypatch, think_budget=300)
    assert think.budget() == 300
    _cfg(monkeypatch, think_budget="lots")
    assert think.budget() == think.BUDGET == 1024


# ── the kind of step ─────────────────────────────────────────────────────────────────────────

ASK_FIRST = "Not executed: ask first"


@pytest.mark.parametrize("label, msgs, capped, want", [
    ("pass one", Q, False, "first"),
    ("a read came back", Q + _round(("read_file", "done")), False, "information"),
    ("a read failed", Q + _round(("read_file", "error")), False, "recovery"),
    ("one failure in a batch", Q + _round(("read_file", "done"), ("read_file", "error")), False, "recovery"),
    ("a write completed", Q + _round(("write_file", "done")), False, "wrap-up"),
    ("a send completed", Q + _round(("send_message", "done")), False, "wrap-up"),
    ("a shell command returned output", Q + _round(("run_shell", "done")), False, "information"),
    ("a write and a read", Q + _round(("write_file", "done"), ("read_file", "done")), False, "information"),
    ("declined at the gate", Q + _round(("write_file", "skipped")), False, "wrap-up"),
    ("blocked by the air-gap", Q + _round(("web_search", "blocked")), False, "wrap-up"),
    ("a plan was written", Q + _round(("plan", "done")), False, "information"),
    ("an error further back", Q + _round(("read_file", "error")) + _round(("search_files", "done")),
     False, "information"),
    ("an error behind a steer", Q + _round(("read_file", "error")) + STEER, False, "recovery"),
    ("a steer after a clean round", Q + _round(("read_file", "done")) + STEER, False, "steered"),
    ("a steer after a write", Q + _round(("write_file", "done")) + STEER, False, "steered"),
    ("the question answered, a sibling told to wait",
     Q + _round(("ask_user", "done", "the blue one"), ("write_file", "error", ASK_FIRST)), False, "information"),
    ("the cap wins over an error", Q + _round(("read_file", "error")), True, "capped"),
])
def test_step_kind(label, msgs, capped, want):
    assert think.step_kind(msgs, capped, mechanical=(ASK_FIRST,)) == want, label


def test_a_mechanical_refusal_is_evidence_unless_named():
    msgs = Q + _round(("ask_user", "done"), ("write_file", "error", ASK_FIRST))
    assert think.step_kind(msgs, False) == "recovery"
    assert think.step_kind(msgs, False, mechanical=(ASK_FIRST,)) == "information"


def test_step_kind_reads_only_the_latest_turn_slice():
    """The caller passes this turn's messages; an unknown tool name is not an action (its round
    reads as information), and a missing stamp is a completed call."""
    msgs = Q + [AIMessage(content="", tool_calls=[_call("mcp_x_lookup", "c0")]),
                ToolMessage(content="rows", tool_call_id="c0", name="mcp_x_lookup")]
    assert think.step_kind(msgs, False) == "information"


# ── the decision ─────────────────────────────────────────────────────────────────────────────

T, D, N = "think", "draft", "no"
TABLE = {
    #               fast  auto  auto under `recover` (the baseline)  deep
    "first":       (N,    D,    N,                                   T),
    "information": (N,    D,    N,                                   T),
    "recovery":    (N,    T,    T,                                   T),
    "steered":     (N,    T,    N,                                   T),
    "wrap-up":     (N,    D,    N,                                   T),
    "capped":      (N,    N,    N,                                   N),
}
COLUMNS = (("fast", "act"), ("auto", "act"), ("auto", "recover"), ("deep", "act"))


def test_decide_is_the_specs_table():
    assert set(TABLE) == set(think.KINDS)
    for kind, row in TABLE.items():
        for (level, policy), want in zip(COLUMNS, row):
            d = think.decide(level, kind, policy)
            got = T if d.think else D if d.draft else N
            assert got == want, (kind, level, policy)
            assert not (d.think and d.draft)
            assert d.why == think.KIND_WORDS[kind]


def test_fast_and_deep_ignore_the_policy():
    for policy in think.POLICIES:
        for kind in think.KINDS:
            assert think.decide("fast", kind, policy) == think.decide("fast", kind)
            assert think.decide("deep", kind, policy) == think.decide("deep", kind)


def test_a_model_without_a_thinking_mode_never_thinks():
    for level in think.LEVELS:
        for policy in think.POLICIES:
            for kind in think.KINDS:
                d = think.decide(level, kind, policy, supported=False)
                assert not d.think and not d.draft


def test_supported_follows_the_daemons_rejection(monkeypatch):
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "plain")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    assert think.supported()
    llms._NO_THINK_SUPPORT.add("plain")
    assert not think.supported()


# ── the record ───────────────────────────────────────────────────────────────────────────────


def test_entry_and_describe():
    d = think.decide("deep", "first")
    e = think.entry(n=1, kind="first", decision=d, outcome="thought",
                    thought={"seconds": 1.84, "tokens": 91, "text": "the  file\nhas it " * 60})
    assert e["asked"] and e["pass"] == 1 and e["seconds"] == 1.84 and e["tokens"] == 91
    assert len(e["text"]) <= 400 and "\n" not in e["text"]
    assert think.describe(e) == "first move · thought 1.8s"
    assert think.describe({**e, "draft": True}).endswith("(rethought a drafted call)")

    quiet = think.entry(n=2, kind="wrap-up", decision=think.decide("auto", "wrap-up"))
    assert not quiet["asked"] and think.describe(quiet) == "wrap-up · no thought"
    cut = think.entry(n=3, kind="recovery", decision=d, outcome="cut-budget", thought={"tokens": 1024})
    assert cut["asked"] and "cut at 1024 tokens" in think.describe(cut)
    assert "Esc" in think.describe({**cut, "outcome": "cut-esc"})
    assert "empty" in think.describe({**cut, "outcome": "empty"})
    bad = think.entry(n=1, kind="recovery", decision=d, outcome="malformed")
    assert bad["asked"] and "malformed" in think.describe(bad)
    plain = think.entry(n=1, kind="recovery", decision=d, outcome="unsupported")
    assert not plain["asked"] and "no thinking mode" in think.describe(plain)


# ── the pass (nodes/agent.py) ────────────────────────────────────────────────────────────────
# `_generate` is the one model seam; these replace it and read what the node asked of it.


def _state(msgs, **kw):
    s = {"messages": msgs, "current_query": str(msgs[0].content) if msgs else "", "context": "",
         "plan": [], "iteration": 0, "tools_called": [], "tool_results": [],
         "documents_retrieved": [], "tool_events": [], "gate_events": []}
    s.update(kw)
    return s


def _seam(monkeypatch, reply):
    """Replace `_generate` with `reply(think) -> AIMessage`; returns the list of think flags
    it was called with."""
    from nodes import agent
    seen = []

    def fake(llm_input, *, tools, think=False):
        seen.append(think)
        return reply(think)

    monkeypatch.setattr(agent, "_generate", fake)
    return seen


CALL = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "a"},
                                          "id": "c9", "type": "tool_call"}])


def _thought(**kw):
    from nodes.agent import THOUGHT_KEY
    return {THOUGHT_KEY: {"seconds": 1.5, "tokens": 40, "text": "the file has it", "cut": None, **kw}}


def test_every_pass_records_one_think_entry(monkeypatch):
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="recover")
    _seam(monkeypatch, lambda think: AIMessage(content="hi"))
    out = agent.agent_node(_state(Q))
    (e,) = out["think"]
    assert (e["pass"], e["kind"], e["asked"], e["outcome"]) == (1, "first", False, "none")
    failed = Q + _round(("read_file", "error"))
    seen = _seam(monkeypatch, lambda think: AIMessage(content="not there", response_metadata=_thought()))
    out = agent.agent_node(_state(failed, iteration=1))
    (e,) = out["think"]
    assert seen == [True]
    assert (e["pass"], e["kind"], e["outcome"], e["seconds"], e["tokens"]) == (2, "recovery", "thought", 1.5, 40)
    assert e["text"] == "the file has it"
    # the report never rides the conversation's message
    assert agent.THOUGHT_KEY not in out["messages"][-1].response_metadata


def test_draft_first_a_text_answer_is_one_call(monkeypatch):
    """decide-draft: the first move runs think-off. A chat answer stands — a chat turn pays
    nothing for the policy."""
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="act")
    seen = _seam(monkeypatch, lambda think: AIMessage(content="Hello!"))
    out = agent.agent_node(_state(Q))
    assert seen == [False]
    assert out["messages"][-1].content.startswith("Hello!")
    assert out["think"][0]["outcome"] == "none" and not out["think"][0]["draft"]


def test_draft_first_a_drafted_call_is_rethought(monkeypatch):
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="act")
    rethought = AIMessage(content="", tool_calls=[{"name": "list_directory", "args": {"path": "."},
                                                   "id": "c8", "type": "tool_call"}],
                          response_metadata=_thought())
    seen = _seam(monkeypatch, lambda think: rethought if think else CALL)
    out = agent.agent_node(_state(Q))
    assert seen == [False, True]
    assert [c["name"] for c in out["messages"][-1].tool_calls] == ["list_directory"]
    e = out["think"][0]
    assert e["outcome"] == "thought" and e["draft"] and e["asked"]


@pytest.mark.parametrize("rethink, outcome", [
    (lambda: AIMessage(content=""), "empty"),
    (lambda: AIMessage(content="", response_metadata=_thought(cut="budget", tokens=1025)), "cut-budget"),
    (lambda: AIMessage(content="", response_metadata=_thought(cut="esc")), "cut-esc"),
])
def test_draft_first_an_unusable_rethink_keeps_the_draft(monkeypatch, rethink, outcome):
    """Never a third call: the drafted call is the fallback for a thought that came to nothing."""
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="act")
    seen = _seam(monkeypatch, lambda think: rethink() if think else CALL)
    out = agent.agent_node(_state(Q))
    assert seen == [False, True]
    assert [c["name"] for c in out["messages"][-1].tool_calls] == ["read_file"]
    e = out["think"][0]
    assert e["outcome"] == outcome and not e["draft"] and e["asked"]


@pytest.mark.parametrize("cut, outcome", [("budget", "cut-budget"), ("esc", "cut-esc")])
def test_a_cut_thought_is_dropped_and_the_pass_answers_without_it(monkeypatch, cut, outcome):
    from nodes import agent
    _cfg(monkeypatch, think="deep")
    seen = _seam(monkeypatch, lambda think: AIMessage(content="", response_metadata=_thought(cut=cut, tokens=77))
                 if think else AIMessage(content="Here it is."))
    out = agent.agent_node(_state(Q))
    assert seen == [True, False]
    assert out["messages"][-1].content.startswith("Here it is.")
    e = out["think"][0]
    assert e["outcome"] == outcome and e["tokens"] == 77


def test_a_model_that_turns_out_not_to_think_is_not_rerun(monkeypatch):
    """The daemon rejects the think flag during the call (core/llms retries it without the
    flag and remembers the tag): that call WAS the think-off pass, so an empty result is not
    rerun — the rerun would be the identical call."""
    from core import llms
    from nodes import agent
    monkeypatch.setattr(llms, "model_tag", lambda: "plain")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    _cfg(monkeypatch, think="deep")

    def reply(think):
        llms._NO_THINK_SUPPORT.add("plain")
        return AIMessage(content="")

    seen = _seam(monkeypatch, reply)
    out = agent.agent_node(_state(Q))
    assert seen == [True]
    assert out["think"][0]["outcome"] == "unsupported" and not out["think"][0]["asked"]
    # from then on the decision itself says no
    seen = _seam(monkeypatch, lambda think: AIMessage(content="a"))
    agent.agent_node(_state(Q))
    assert seen == [False]


def test_an_unsupported_tag_gets_no_extra_output_room(monkeypatch):
    from core import llms
    monkeypatch.setattr(llms, "model_tag", lambda: "plain")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", {"plain"})
    kw = llms.invoke_kwargs(None, 0.0, task="agent", think=True)
    assert "reasoning" not in kw and kw["options"]["num_predict"] == llms.NUM_PREDICT["agent"]
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    _cfg(monkeypatch, think_budget=None)
    kw = llms.invoke_kwargs(None, 0.0, task="agent", think=True)
    assert kw["reasoning"] is True and kw["options"]["num_predict"] == llms.NUM_PREDICT["agent"]


def test_the_budget_default_is_1024(monkeypatch):
    from config import get_config
    from core import llms
    cfg = get_config()
    runtime = {k: v for k, v in cfg._data.get("runtime", {}).items() if k != "think_budget"}
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "runtime": runtime})
    monkeypatch.setattr(llms, "model_tag", lambda: "m")
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    kw = llms.invoke_kwargs(None, 0.0, task="agent", think=True)
    assert kw["options"]["num_predict"] == llms.NUM_PREDICT["agent"] + 1024


def test_ask_first_is_not_evidence(monkeypatch):
    """ask_user runs alone; its sibling is told to wait (stamped `error` for the incidents
    note). Nothing failed, so the pass after the user answers does not think under `recover`."""
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="recover")
    seen = _seam(monkeypatch, lambda think: AIMessage(content="a"))
    msgs = Q + _round(("ask_user", "done", "the blue one"), ("write_file", "error", agent.ASK_ALONE_TEXT))
    out = agent.agent_node(_state(msgs, iteration=1))
    assert seen == [False] and out["think"][0]["kind"] == "information"


def test_the_turns_own_level_wins(monkeypatch):
    from nodes import agent
    _cfg(monkeypatch, think="fast")
    seen = _seam(monkeypatch, lambda think: AIMessage(content="a", response_metadata=_thought() if think else {}))
    agent.agent_node(_state(Q))
    agent.agent_node(_state(Q, think_level="deep"))
    assert seen == [False, True]


# ── the bounded thought (nodes/agent._generate over a fake stream) ───────────────────────────


class _Model:
    def bind_tools(self, _tools):
        return self


def _stream_of(monkeypatch, chunks):
    """Point `_generate` at a stream of `chunks`; returns {"closed": bool, "kwargs": {...}}."""
    from nodes import agent
    seen = {"closed": False, "read": 0}

    def fake_stream(runnable, llm_input, *, tag="", **kwargs):
        seen["kwargs"] = kwargs
        try:
            for c in chunks:
                seen["read"] += 1
                yield c
        finally:
            seen["closed"] = True

    monkeypatch.setattr(agent, "llm_stream", fake_stream)
    monkeypatch.setattr(agent, "get_model", lambda: _Model())
    monkeypatch.setattr(agent, "model_tag", lambda: "m")
    return seen


def _r(text):
    from langchain.messages import AIMessageChunk
    return AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})


def _c(text):
    from langchain.messages import AIMessageChunk
    return AIMessageChunk(content=text)


def test_generate_reports_the_thought(monkeypatch):
    from nodes import agent
    seen = _stream_of(monkeypatch, [_r("the file "), _r("has it"), _c("4"), _c("2")])
    ai = agent._generate([HumanMessage(content="q")], tools=True, think=True)
    t = ai.response_metadata[agent.THOUGHT_KEY]
    assert ai.content == "42"
    assert (t["tokens"], t["text"], t["cut"]) == (2, "the file has it", None) and t["seconds"] >= 0
    assert seen["closed"] and seen["kwargs"]["reasoning"] is True
    # a pass that did not reason carries no report
    _stream_of(monkeypatch, [_c("hi")])
    assert agent.THOUGHT_KEY not in agent._generate([HumanMessage(content="q")], tools=True).response_metadata


def test_generate_cuts_a_thought_at_the_budget(monkeypatch):
    from nodes import agent
    _cfg(monkeypatch, think_budget=3)
    seen = _stream_of(monkeypatch, [_r("a ")] * 10 + [_c("never read")])
    ai = agent._generate([HumanMessage(content="q")], tools=True, think=True)
    t = ai.response_metadata[agent.THOUGHT_KEY]
    assert t["cut"] == "budget" and t["tokens"] == 3 and ai.content == "" and not ai.tool_calls
    assert seen["closed"] and seen["read"] == 3  # the stream stopped generating, at the budget
    assert agent._is_empty(ai)


def test_esc_stops_a_thought_and_the_pause_stays_pending(monkeypatch):
    from core.pause import get_pause_controller
    from nodes import agent
    controller = get_pause_controller()
    controller.reset()
    try:
        controller.request("user", "you pressed Esc to pause")
        seen = _stream_of(monkeypatch, [_r("hmm ")] * 5 + [_c("late")])
        ai = agent._generate([HumanMessage(content="q")], tools=True, think=True)
        assert ai.response_metadata[agent.THOUGHT_KEY]["cut"] == "esc" and seen["read"] == 1
        assert controller.pending()  # handled at the next pass boundary, as ever
        # a pass that was not asked to think is never cut, whatever it streams (the adapter
        # surfaces no reasoning on a think-off call; this guards the loop against one that did)
        _stream_of(monkeypatch, [_r("hmm "), _c("ok")])
        ai = agent._generate([HumanMessage(content="q")], tools=True, think=False)
        assert ai.content == "ok" and ai.response_metadata[agent.THOUGHT_KEY]["cut"] is None
    finally:
        controller.reset()


def test_a_zero_budget_does_not_cut(monkeypatch):
    from nodes import agent
    _cfg(monkeypatch, think_budget=0)
    _stream_of(monkeypatch, [_r("a ")] * 6 + [_c("ok")])
    ai = agent._generate([HumanMessage(content="q")], tools=True, think=True)
    assert ai.content == "ok" and ai.response_metadata[agent.THOUGHT_KEY]["cut"] is None


# ── the record (stores/trace.py, /trace why) ─────────────────────────────────────────────────


def _handler(tmp_path):
    from stores.trace import LLMTraceHandler, Tracer
    db = str(tmp_path / "trace.sqlite")
    tracer = Tracer(db)
    rid = tracer.start_run("thread", "q")
    return db, rid, LLMTraceHandler(tracer, rid)


def _outputs(db, rid):
    import json
    import sqlite3
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT output, status FROM llm_calls WHERE run_id = ? ORDER BY seq", (rid,)).fetchall()
    return [(json.loads(o), s) for o, s in rows]


def test_llm_calls_record_what_the_call_was_sent_with(tmp_path):
    """`think` is what the call was SENT with (invocation_params' `reasoning`), so a thinking
    call that produced no reasoning is still told from a think-off one."""
    from langchain_core.outputs import ChatGeneration, LLMResult
    db, rid, h = _handler(tmp_path)
    for run, reasoning in (("a", True), ("b", False), ("c", None)):
        params = {} if reasoning is None else {"reasoning": reasoning}
        h.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id=run,
                              metadata={"langgraph_node": "agent"}, invocation_params=params)
        h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hi"))]]), run_id=run)
    assert [(o["think"], s) for o, s in _outputs(db, rid)] == [(True, "ok"), (False, "ok"), (False, "ok")]


def test_a_cut_thought_keeps_its_reasoning_in_the_record(tmp_path):
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk, LLMResult
    db, rid, h = _handler(tmp_path)
    h.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id="a",
                          metadata={"langgraph_node": "agent"}, invocation_params={"reasoning": True})
    partial = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "let me think about"})
    h.on_llm_error(GeneratorExit(), run_id="a",
                   response=LLMResult(generations=[[ChatGenerationChunk(message=partial)]]))
    # a think-off stream closed by a cancelled turn keeps its old wording
    h.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id="b",
                          metadata={"langgraph_node": "agent"}, invocation_params={"reasoning": False})
    h.on_llm_error(GeneratorExit(), run_id="b")
    (cut, s1), (cancelled, s2) = _outputs(db, rid)
    assert s1 == s2 == "cancelled"
    assert cut["think"] and cut["reasoning"] == "let me think about" and "thought was cut" in cut["error"]
    assert not cancelled["think"] and "reasoning" not in cancelled and "freeze/cancel" in cancelled["error"]


def test_trace_why_says_when_it_thought(tmp_path, capsys):
    from types import SimpleNamespace

    from commands import trace as tr
    from stores.trace import Tracer

    db = str(tmp_path / "db.sqlite")
    t = Tracer(db)
    d = think.decide("deep", "first")
    rid = t.start_run("th", "q")
    t.log_event(rid, "agent", {"iteration": 1, "think": [think.entry(
        n=1, kind="first", decision=d, outcome="thought", thought={"seconds": 1.84, "tokens": 90})]})
    t.log_event(rid, "tools", {"tools_called": ["write_file"]})
    t.log_event(rid, "agent", {"iteration": 2, "think": [think.entry(
        n=2, kind="wrap-up", decision=think.decide("auto", "wrap-up"))]})
    t.end_run(rid, "ok", "done")
    quiet = t.start_run("th2", "hi")
    t.log_event(quiet, "agent", {"iteration": 1, "think": [think.entry(
        n=1, kind="first", decision=think.decide("auto", "first"))]})
    t.end_run(quiet, "ok", "hello")

    tr._why(SimpleNamespace(db_path=db, state={}), [str(rid)])
    out = capsys.readouterr().out
    assert "when it thought" in out
    assert "pass 1: first move · thought 1.8s" in out and "pass 2: wrap-up · no thought" in out
    tr._why(SimpleNamespace(db_path=db, state={}), [str(quiet)])
    assert "when it thought" not in capsys.readouterr().out  # a turn of plain passes: nothing to explain


# ── what the user sees (app/turn.py, tui/ui) ─────────────────────────────────────────────────


def test_run_turn_tells_on_thinking_when_a_thought_begins_and_ends():
    from types import SimpleNamespace

    from app import turn

    class G:
        def stream(self, *a, **k):
            yield ("custom", {"type": "thinking", "phase": "start"})
            yield ("custom", {"type": "thinking", "phase": "end"})
            yield ("custom", {"type": "retract"})

        def get_state(self, config):
            return SimpleNamespace(next=(), values={"messages": []}, tasks=[])

    seen = []
    turn.run_turn(G(), {}, {}, approver=lambda v: True, on_thinking=seen.append)
    assert seen == [True, False]
    turn.run_turn(G(), {}, {}, approver=lambda v: True)  # no callback: the events are ignored


def test_the_status_bar_says_it_is_thinking_and_that_esc_stops_it(monkeypatch):
    import importlib
    import time

    from tui.ui import _base
    sb = importlib.import_module("tui.ui.statusbar")
    monkeypatch.setattr(sb, "_usage", None)
    monkeypatch.setattr(sb, "_live_refresh", lambda: None)
    monkeypatch.setattr(_base, "_status", {**_base._status, "thinking": None})
    calm = sb._StatusBar().__rich__().plain
    assert "thinking" not in calm and calm.rstrip().endswith("esc pause · ctrl-c cancel")
    sb.set_thinking(True)
    _base._status["thinking"] = time.perf_counter() - 3.2
    plain = sb._StatusBar().__rich__().plain
    assert "thinking 3." in plain and plain.rstrip().endswith("esc stops thinking · ctrl-c cancel")
    sb.set_thinking(False)
    assert "thinking" not in sb._StatusBar().__rich__().plain


def _entry(outcome="thought", kind="first", **thought):
    return think.entry(n=1, kind=kind, decision=think.decide("deep", kind), outcome=outcome,
                       thought={"seconds": 1.84, "tokens": 90, "text": "the file has it", **thought})


def test_the_rail_row_and_leaf_show_the_thought(monkeypatch):
    import importlib

    from tui.ui import _base
    tr = importlib.import_module("tui.ui.trace")
    assert "thought 1.8s" in tr._metric_parts({"iteration": 1, "think": [_entry()]})
    assert "thought dropped" in tr._metric_parts({"think": [_entry("cut-budget", tokens=1025)]})
    quiet = think.entry(n=1, kind="wrap-up", decision=think.decide("auto", "wrap-up"))
    assert not any("thought" in p for p in tr._metric_parts({"think": [quiet]}))
    assert not any("thought" in p for p in tr._metric_parts({"iteration": 1}))

    leaves = []
    monkeypatch.setattr(tr, "_node_leaf", lambda text, style: leaves.append(text))
    tr._render_think_leaf(_entry())
    tr._render_think_leaf(_entry("cut-budget", kind="recovery", tokens=1025))
    tr._render_think_leaf(_entry("empty"))
    tr._render_think_leaf(quiet)
    assert leaves == ["thought (first move): the file has it",
                      "after an error · thought cut at 1025 tokens — answered without it",
                      "first move · thought came back empty — answered without it"]

    # the turn's thinking time accumulates for the receipt, and a finished pass clears the flag
    monkeypatch.setattr(_base, "_status", {**_base._status, "thinking": 1.0, "thought_s": 0.0})
    monkeypatch.setattr(tr, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_live_refresh", lambda: None)
    answer = {"iteration": 2, "messages": [AIMessage(content="done")], "think": [_entry()]}
    tr.show_node("agent", answer)
    tr.show_node("agent", answer)
    assert _base._status["thinking"] is None and round(_base._status["thought_s"], 2) == 3.68
    resp = importlib.import_module("tui.ui.response")
    assert "thought 3.7s" in resp._stats_parts()
    _base._status["thought_s"] = 0.0
    assert not any("thought" in p for p in resp._stats_parts())


# ── the loop benchmark's think flags (benchmark.py) ──────────────────────────────────────────


def test_benchmark_think_flag_sets_a_level_or_a_policy_in_memory(monkeypatch):
    import benchmark
    _cfg(monkeypatch, think="fast", think_policy="act")  # (the helper restores the policy after)
    assert benchmark.apply_think("recover") == "recover"   # the baseline: auto under the old rule
    assert (think.level(), think.policy(), benchmark.think_label()) == ("auto", "recover", "recover")
    assert benchmark.apply_think("deep") == "deep"
    assert (think.level(), benchmark.think_label()) == ("deep", "deep")
    assert benchmark.apply_think("off") == "fast" and benchmark.think_label() == "fast"
    assert benchmark.apply_think("act") == "act"
    assert benchmark.apply_think("auto") == "act"          # auto is named by its policy
    for gone in ("smart", "decide-draft", "decide", "first"):
        with pytest.raises(SystemExit):
            benchmark.apply_think(gone)


def test_benchmark_tier_flag_is_in_memory_and_refuses_an_unknown_tier(monkeypatch):
    import benchmark
    from config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data, "tiers": {"4b": {"model": "a"}, "9b": {"model": "b"}},
                                       "active_tier": "4b"})
    assert benchmark.apply_tier("9b") == "9b" and cfg.get("active_tier") == "9b"
    with pytest.raises(SystemExit):
        benchmark.apply_tier("900b")


def test_loop_summary_counts_what_thinking_cost():
    import benchmark
    deep = think.decide("deep", "first")
    quiet = think.entry(n=2, kind="wrap-up", decision=think.decide("auto", "wrap-up"), prompt_s=0.25)
    results = [
        {"id": "a", "shape": "chat", "tags": [], "iterations": 1, "latency_s": 1.0, "think": [
            think.entry(n=1, kind="first", decision=deep, outcome="thought", draft=True,
                        thought={"seconds": 1.5}, prompt_s=0.5)]},
        {"id": "b", "shape": "multi", "tags": [], "iterations": 3, "latency_s": 4.0, "think": [
            think.entry(n=1, kind="first", decision=deep, outcome="empty", thought={"seconds": 0.5}),
            think.entry(n=2, kind="recovery", decision=deep, outcome="cut-budget", thought={"seconds": 9.0}),
            quiet]},
        {"id": "c", "shape": "chat", "tags": [], "iterations": 1, "latency_s": 1.0},  # an errored task
    ]
    assert benchmark.summarize_loop(results)["thinking"] == {
        "passes": 4, "thought": 3, "drafts": 1, "empty": 1, "cut": 1, "seconds": 11.0,
        "prompt_seconds": 0.75}


def test_loop_report_name_carries_the_tier_and_the_think_mode(monkeypatch, tmp_path):
    import benchmark
    monkeypatch.setattr(benchmark, "_log_dir", lambda: tmp_path)
    one = benchmark.loop_report_path("9b", "decide-draft")
    assert one.parent == tmp_path and one.name.startswith("loop_9b_decide-draft_") and one.suffix == ".json"
    assert benchmark.loop_report_path("4b", "deep", run=2, runs=3).name.endswith("_2.json")


# ── review fixes (2026-10-04) ────────────────────────────────────────────────────────────────


def test_a_thinking_call_malformed_twice_is_not_called_empty(monkeypatch):
    """`_generate_or_retry` returns None when the output could not be parsed twice. That is not
    an empty thought and nothing was rerun: the record says so, and the turn answers honestly."""
    from nodes import agent
    _cfg(monkeypatch, think="deep")
    calls = []
    monkeypatch.setattr(agent, "_generate_or_retry", lambda i, *, tools, think=False: calls.append(think))
    out = agent.agent_node(_state(Q))
    assert calls == [True]
    assert out["think"][0]["outcome"] == "malformed"
    assert out["messages"][-1].content.startswith(agent.MALFORMED_TEXT)


def _why(tmp_path, capsys, entries, rows):
    """Render /trace why for one run: `entries` are the passes' think records, `rows` the
    agent's llm_calls outputs in order."""
    import json
    import sqlite3
    from types import SimpleNamespace

    from commands import trace as tr
    from stores.trace import Tracer

    db = str(tmp_path / "why.sqlite")
    t = Tracer(db)
    rid = t.start_run("th", "q")
    for e in entries:
        t.log_event(rid, "agent", {"iteration": e["pass"], "think": [e]})
    with sqlite3.connect(db) as c:
        for seq, row in enumerate(rows, 1):
            c.execute("INSERT INTO llm_calls (run_id, seq, ts, node, model, dur, prompt_tokens, output_tokens, "
                      "input, output, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (rid, seq, "2026-10-04T00:00:00", "agent", "m", 0.1, 10, 5, "[]", json.dumps(row), "ok"))
    t.end_run(rid, "ok", "done")
    tr._why(SimpleNamespace(db_path=db, state={}), [str(rid)])
    return capsys.readouterr().out


def _row(content="", calls=(), think=False, reasoning=""):
    out = {"content": content, "think": think,
           "tool_calls": [{"name": n, "args": {"q": "x"}} for n in calls]}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def test_trace_why_never_shows_a_discarded_draft_as_a_choice(tmp_path, capsys):
    """Under decide-draft the first move's think-off draft is retracted and rethought. The audit
    view says what was ISSUED, names the draft as a draft, and numbers passes the way the
    "when it thought" section does — one per agent pass, not one per model call."""
    drafted = think.decide("auto", "first", "decide-draft")
    entries = [
        think.entry(n=1, kind="first", decision=drafted, outcome="thought", draft=True, thought={"seconds": 1.1}),
        think.entry(n=2, kind="information", decision=think.decide("auto", "information", "decide"),
                    outcome="cut-budget", thought={"tokens": 1024}),
    ]
    rows = [_row(calls=["web_search"]),                                           # the draft
            _row(calls=["calculate"], think=True, reasoning="arithmetic needs the calculator"),
            _row(think=True, reasoning="the result is"),                          # the cut thought
            _row(content="It is 391.")]                                           # the rerun
    out = _why(tmp_path, capsys, entries, rows)
    assert "→ chose to call: calculate(" in out and "→ chose to call: web_search" not in out
    assert "drafted without thinking, then rethought — not issued: web_search(" in out
    assert "pass 2: answered — It is 391." in out and "pass 3" not in out
    assert "a thought was dropped (empty, cut or stopped): the result is" in out
    assert "pass 2: new information · thought cut at 1024 tokens" in out


def test_trace_why_keeps_a_draft_that_stood(tmp_path, capsys):
    drafted = think.decide("auto", "first", "decide-draft")
    entries = [think.entry(n=1, kind="first", decision=drafted, outcome="empty")]
    rows = [_row(calls=["read_file"]), _row(think=True)]
    out = _why(tmp_path, capsys, entries, rows)
    assert "→ chose to call: read_file(" in out and "a thought was dropped" in out
    assert "not issued" not in out


def test_trace_why_falls_back_to_one_line_per_call_when_the_records_do_not_line_up(tmp_path, capsys):
    """An extra call the think records cannot place (a malformed-output retry): no guessing."""
    entries = [think.entry(n=1, kind="first", decision=think.decide("auto", "first"))]
    out = _why(tmp_path, capsys, entries, [_row(calls=["read_file"]), _row(content="done")])
    assert "pass 1:" in out and "pass 2: answered — done" in out


def test_the_legend_promises_esc_only_for_an_empty_line(monkeypatch):
    import importlib
    import time

    from tui.ui import _base
    sb = importlib.import_module("tui.ui.statusbar")
    monkeypatch.setattr(sb, "_usage", None)
    monkeypatch.setattr(_base, "_status", {**_base._status, "thinking": time.perf_counter()})
    monkeypatch.setattr(sb, "_input_state", {"buffer": "use the other file", "queued": 0})
    plain = sb._StatusBar().__rich__().plain
    assert "thinking" in plain and plain.rstrip().endswith("esc pause · ctrl-c cancel")


def test_the_pause_note_says_the_thought_is_being_stopped(monkeypatch):
    import importlib

    from tui.ui import _base
    ro = importlib.import_module("tui.ui.readouts")
    said = []
    monkeypatch.setattr(ro, "_glyph_line", lambda *a, **k: said.append(a[2]))
    monkeypatch.setattr(_base, "_status", {**_base._status, "thinking": None})
    ro.pause_note()
    monkeypatch.setattr(_base, "_status", {**_base._status, "thinking": 12.0})
    ro.pause_note()
    assert said == ["pausing at the next pass…", "stopping the thought — pausing at the next pass…"]


def test_a_batch_of_runs_survives_a_file_the_tasks_did_not_expect(monkeypatch, tmp_path):
    """`--runs N`: a model that leaves `bench_summary.txt` behind must not abort run 2 — every
    bench_* file present at exit was made by the run (enter refuses to start over any)."""
    import benchmark
    from core import workspace
    monkeypatch.setattr(workspace, "reset", lambda: None)
    monkeypatch.setattr(workspace, "root", lambda: tmp_path)
    (tmp_path / "keep.txt").write_text("the user's")
    for _run in range(2):
        with benchmark._loop_fixtures():
            (tmp_path / "bench_summary.txt").write_text("unexpected")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_act_drafts_every_deciding_pass_and_thinks_only_before_a_call(monkeypatch):
    """`act`: think before acting, never before a text answer. A pass after an information
    round (and after a completed action) is drafted like the first move — a text answer is one
    call, a drafted tool call is rethought."""
    from nodes import agent
    _cfg(monkeypatch, think="auto", think_policy="act")
    after_read = Q + _round(("read_file", "done"))
    seen = _seam(monkeypatch, lambda think: AIMessage(content="It says 4471."))
    out = agent.agent_node(_state(after_read, iteration=1))
    assert seen == [False] and out["think"][0]["outcome"] == "none"
    seen = _seam(monkeypatch, lambda think: AIMessage(
        content="", tool_calls=CALL.tool_calls, response_metadata=_thought() if think else {}))
    for msgs in (after_read, Q + _round(("write_file", "done"))):
        seen.clear()
        out = agent.agent_node(_state(msgs, iteration=1))
        assert seen == [False, True] and out["think"][0]["draft"]
    # after an error it thinks outright, as `recover` does
    seen = _seam(monkeypatch, lambda think: AIMessage(content="not there", response_metadata=_thought()))
    agent.agent_node(_state(Q + _round(("read_file", "error")), iteration=1))
    assert seen == [True]
