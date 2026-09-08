"""KV-cache stability of the engine's prompts (2026-09-04).

Measured against the live daemon (Ollama 0.33 / qwen3.5:9b): llama-server restores a prompt
only up to a saved context checkpoint (N-1024 and N-4 of an earlier prompt, the restore point of
a lineage), so a prompt reuses its predecessor's prefill exactly when everything that changed
sits AFTER such a checkpoint. Three things broke that for every node: the per-result caps were
recomputed from the count of results (an earlier result's rendering changed when a later one
landed), the request rode INSIDE the grounding message (only N-1024 was ever reachable), and the
native tool bind rendered the tool schema into the system message (a full re-prefill per step).
This file pins the prefix-stable layouts, the stable/dynamic grounding split, the idle primes,
and the grammar-constrained argument path. Fully offline: every model seam is a stub.
"""

import json

from langchain.messages import AIMessage, HumanMessage

from core import plan_context, structured as st
from nodes import execute as ex


def _step(step_id, tool=None, result=None, status="pending", label=None):
    return {"step_id": step_id, "label": label or f"step {step_id}", "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


def _state(plan, **kw):
    base = {"messages": [HumanMessage("the request")], "plan": plan,
            "current_query": "the request", "context": "", "iteration": 0, "replans": 0}
    base.update(kw)
    return base


# ── results block: prefix-stable per-result caps ────────────────────────────────────────────


def _long_plan(n, size=5000):
    return [_step(i, "read_file", result=f"R{i}:" + "B" * size, status="done")
            for i in range(1, n + 1)]


def test_results_block_is_prefix_stable_as_results_land():
    """Rendering k results must be a byte-prefix of rendering k+1: an earlier result's cap is
    fixed when it lands, never recomputed from how many results followed it."""
    plan = _long_plan(8)
    for k in range(1, len(plan)):
        shorter = plan_context.results_block(plan[:k])
        longer = plan_context.results_block(plan[:k + 1])
        assert longer.startswith(shorter), f"result {k + 1} rewrote an earlier result"


def test_plan_txt_done_lines_are_prefix_stable():
    """The rectify/replan plan text: the DONE lines already rendered stay byte-identical when the
    next step lands (only the pending tail changes)."""
    plan = _long_plan(6) + [_step(7, "read_file"), _step(8, "calculate")]
    before = plan_context.plan_txt(plan[:5] + plan[5:])  # step 6 done, 7-8 pending
    plan[5]["result"] = None  # roll step 6 back to pending
    earlier = plan_context.plan_txt(plan)
    done_prefix = earlier.split("\n6. [PENDING]")[0]
    assert before.startswith(done_prefix)


def test_results_block_respects_the_budget_and_the_floor():
    n = 20
    plan = _long_plan(n)
    block = plan_context.results_block(plan)
    assert len(block) < plan_context._BLOCK_BUDGET + n * (plan_context._RESULT_FLOOR + 120)
    for i in range(1, n + 1):
        # every result keeps at least the floor of its text
        assert f"R{i}:" + "B" * (plan_context._RESULT_FLOOR - 10) in block
    # early results ride whole up to the cap; late ones are squeezed to the floor
    assert "R1:" + "B" * (plan_context._RESULT_CAP - 10) in block
    assert "R20:" + "B" * (plan_context._RESULT_FLOOR + 50) not in block


def test_the_landing_caps_are_a_running_budget():
    caps = plan_context.landing_caps(["x" * 5000, "x" * 5000, "x" * 5000, "x" * 5000])
    assert caps[0] == plan_context._RESULT_CAP
    assert caps[1] == plan_context._RESULT_CAP
    assert caps[2] == plan_context._BLOCK_BUDGET - 2 * plan_context._RESULT_CAP
    assert caps[3] == plan_context._RESULT_FLOOR
    # a short result consumes only what it renders, leaving the rest to the next
    caps = plan_context.landing_caps(["short", "x" * 5000, "x" * 5000, "x" * 5000])
    assert caps[3] > plan_context._RESULT_FLOOR


def test_callout_tail_stays_under_one_prefill_batch():
    """The previous-step callout + the step line are the execute prompt's per-step TAIL; at
    ~4 chars/token it must stay well inside llama-server's 1024-token batch or the N-1024
    checkpoint of the previous step's prompt lands past the divergence and nothing restores."""
    assert plan_context._CALLOUT_CAP <= 2000


# ── execute context: message-boundary layout ───────────────────────────────────────────────


def test_exec_parts_put_the_stable_grounding_first_and_the_step_last():
    plan = [_step(1, "read_file", result="alpha", status="done"), _step(2, "calculate")]
    state = _state(plan, context="STABLE\n\nDYN", context_stable="STABLE", context_dynamic="DYN")
    parts = plan_context.exec_parts(state, plan[1])
    assert parts[0] == "STABLE"
    assert parts[1].startswith("User's overall request: the request")
    assert "DYN" in parts[1]
    assert parts[2].startswith("Results from earlier steps")
    assert parts[-1].startswith("The immediately preceding step")
    assert parts[-1].rstrip().endswith("Your current step: step 2")
    # the joined string is the same context the trace/tests have always read
    assert plan_context.exec_context(state, plan[1]) == "\n\n".join(parts)


def test_exec_parts_without_a_split_fall_back_to_the_whole_context():
    plan = [_step(1, "read_file")]
    state = _state(plan, context="WHOLE")
    parts = plan_context.exec_parts(state, plan[0])
    assert parts[0] == "WHOLE" and parts[1].startswith("User's overall request")
    assert plan_context.grounding_parts({"context": "WHOLE"}) == ("WHOLE", "")


# ── grounding: stable / dynamic split ───────────────────────────────────────────────────────


def test_grounding_node_splits_stable_and_per_turn_sections(isolated_paths, monkeypatch):
    from nodes import ground

    (isolated_paths / "database" / "workspace").mkdir(parents=True)
    (isolated_paths / "database" / "workspace" / "SATURDAY.md").write_text("be terse")
    monkeypatch.setattr(ground, "memory_context_split",
                        lambda q: ("- #1 likes tea", "- #2 [entities] tea shop", [2]))
    monkeypatch.setattr(ground, "mark_used", lambda ids: 0)
    msgs = [HumanMessage("earlier q"), AIMessage("earlier a"), HumanMessage("now")]
    out = ground.grounding_node({"messages": msgs, "current_query": "now", "attachments": "ATT"})
    stable, dynamic = out["context_stable"], out["context_dynamic"]
    assert out["context"] == stable + "\n\n" + dynamic
    assert stable.startswith("## Grounding context")
    assert "be terse" in stable and "likes tea" in stable
    for per_turn in ("tea shop", "earlier q", "ATT"):
        assert per_turn in dynamic and per_turn not in stable
    # the stable half is exactly what the idle prime rebuilds between turns
    assert ground.stable_grounding() == stable


def test_memory_context_split_keeps_the_always_layers_query_independent(isolated_paths):
    from stores import memory_registry as mr

    mr.add_memory("prefers tabs", layer="user")
    mr.add_memory("the tea shop closes at five", layer="entities")
    always_a, matched_a, ids_a = mr.memory_context_split("when does the tea shop close")
    always_b, matched_b, ids_b = mr.memory_context_split("unrelated question")
    assert always_a == always_b and "prefers tabs" in always_a
    assert "tea shop" in matched_a and matched_b == ""
    assert ids_a and not ids_b
    # the joined block is what memory_context always returned
    assert mr.memory_context("when does the tea shop close")[0].startswith(always_a)


# ── plan / synthesize prompt layouts ────────────────────────────────────────────────────────


def test_plan_prompt_puts_the_request_in_its_own_message_after_the_stable_grounding(monkeypatch):
    from nodes import plan as pl

    seen = {}

    def fake_structured(role, messages, schema, fmt, shape, default=None, **kw):
        seen["messages"] = messages
        return default

    monkeypatch.setattr(pl, "structured", fake_structured)
    pl.plan_node({"plan": [], "current_query": "read notes.md", "context": "STABLE\n\nDYN",
                  "context_stable": "STABLE", "context_dynamic": "DYN"})
    msgs = seen["messages"]
    assert msgs[1].content == "Grounding context:\nSTABLE"
    assert msgs[2].content == "DYN\n\nUser request:\nread notes.md"


def test_synthesize_prompt_starts_with_the_stable_grounding_message(monkeypatch):
    from nodes import synthesize as syn
    from tests.test_grounding import _Model, _buf, base_state, step

    seen = {}
    monkeypatch.setattr(syn, "get_model", lambda role: _Model([]))
    monkeypatch.setattr(syn, "model_id", lambda role: "test-model")
    monkeypatch.setattr(syn.continuation, "supports", lambda m: False)

    def first_pass(llm_input, freeze):
        seen["input"] = llm_input
        return _buf("fine"), False, {}, None

    monkeypatch.setattr(syn, "_stream_first_pass", first_pass)
    state = base_state("q", plan=[step(1, "Think", None, "x", "done")], context="STABLE\n\nDYN",
                       context_stable="STABLE", context_dynamic="DYN")
    syn.synthesize_node(state)
    msgs = seen["input"]
    assert msgs[1].content == "Relevant context:\nSTABLE"
    assert msgs[2].content.endswith("DYN")
    assert msgs[3].content.startswith("Completed steps")


# ── execute: tool arguments under a grammar, no native bind ─────────────────────────────────


class _ArgModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def bind_tools(self, tools):  # must never be used any more
        raise AssertionError("bind_tools renders the schema into the system prompt")

    def invoke(self, msgs, **kw):
        self.calls.append((msgs, kw))
        return AIMessage(content=self.replies.pop(0))


def test_tool_call_is_generated_under_the_tool_schema_grammar(monkeypatch):
    from tools.registry import tools_by_name

    model = _ArgModel([json.dumps({"arguments": {"file_path": "notes.md"}})])
    monkeypatch.setattr(ex, "get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["read_file"], ["G", "step"])
    assert args == {"file_path": "notes.md"} and failure is None
    msgs, kw = model.calls[0]
    fmt = kw["format"]
    assert fmt["properties"]["arguments"]["properties"]["file_path"]
    assert "refusal" in fmt["properties"]
    # the context parts ride as separate user messages; the tool brief is the tail
    assert [m.content for m in msgs[1:3]] == ["G", "step"]
    assert "read_file" in msgs[-1].content


def test_tool_refusal_lands_as_the_text_fallback(monkeypatch):
    from tools.registry import tools_by_name

    model = _ArgModel([json.dumps({"refusal": "the value was not found"})] * 3)
    monkeypatch.setattr(ex, "get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["write_file"], "ctx")
    assert args is None
    assert "answered in text" in failure and "not found" in failure


def test_tool_call_retries_with_a_schema_hint_after_an_empty_object(monkeypatch):
    from tools.registry import tools_by_name

    model = _ArgModel(["{}", json.dumps({"arguments": {"expression": "1+2"}})])
    monkeypatch.setattr(ex, "get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["calculate"], "ctx")
    assert args == {"expression": "1+2"}
    assert "rejected" in model.calls[1][0][-1].content


def test_tool_call_format_for_an_mcp_shaped_tool_uses_its_own_schema():
    from langchain_core.tools import StructuredTool

    def remote(city: str, days: int = 1) -> str:
        """Weather forecast."""
        return ""

    t = StructuredTool.from_function(remote, name="mcp_wx_forecast")
    fmt = ex.tool_call_format(t)
    props = fmt["properties"]["arguments"]["properties"]
    assert set(props) == {"city", "days"}


# ── the idle primes ─────────────────────────────────────────────────────────────────────────


def test_prime_sends_one_boundary_request_per_lineage(monkeypatch):
    from core import prime

    class M:
        def __init__(self):
            self.calls = []

        def invoke(self, msgs, **kw):
            self.calls.append((msgs, kw))
            return AIMessage(content="")

    model = M()
    monkeypatch.setattr("core.llms.get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    monkeypatch.setattr(prime, "ENABLED", True)
    n = prime.prime("STABLE")
    assert n == 5 == len(model.calls)
    firsts = [m[0][0].content for m in model.calls]
    from core.messages import (EXECUTE_REASONING_SYS, EXECUTE_TOOL_SYS, planner_sys_msg,
                               quick_sys_msg, synthesize_sys_msg)

    # The quick router first: it is the first call of most turns (nodes/quick.py).
    assert firsts == [quick_sys_msg().content, planner_sys_msg().content,
                      EXECUTE_TOOL_SYS.content, synthesize_sys_msg.content,
                      EXECUTE_REASONING_SYS.content]
    seconds = [m[0][1].content for m in model.calls]
    assert seconds == ["STABLE", "Grounding context:\nSTABLE", "STABLE",
                       "Relevant context:\nSTABLE", "STABLE"]
    for _msgs, kw in model.calls:
        assert kw["options"]["num_predict"] == 1
        assert kw["reasoning"] is True  # think ON: think-off adds tokens past the boundary
        assert kw["options"]["num_ctx"] == st._invoke_kwargs("planner", None, 0.0)["options"]["num_ctx"]


def test_prime_never_raises_and_reports_zero_when_the_daemon_is_down(monkeypatch):
    from core import prime

    class Down:
        def invoke(self, msgs, **kw):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("core.llms.get_model", lambda role: Down())
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    monkeypatch.setattr(prime, "ENABLED", True)
    assert prime.prime("STABLE") == 0


def test_priming_is_off_under_tests_and_the_config_knob(monkeypatch):
    from core import prime

    assert prime.ENABLED is False  # conftest: no test may reach a model
    assert prime.start_priming() is None
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: False)
    assert prime.start_priming() is None


def test_start_priming_runs_the_rebuild_on_a_daemon_thread(monkeypatch):
    from core import prime

    seen = []
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: True)
    monkeypatch.setattr(prime, "prime_now", lambda only=None: seen.append("primed") or 4)
    t = prime.start_priming()
    t.join(timeout=5)
    assert t.daemon and not t.is_alive() and seen == ["primed"]


def test_warm_up_thread_primes_after_the_weights_load(monkeypatch):
    from app import startup
    from core import prime

    seen = []
    monkeypatch.setattr(startup, "warm_model", lambda role="tool_caller": seen.append("warm") or True)
    monkeypatch.setattr(prime, "prime_now", lambda only=None: seen.append(("prime", only)) or 1)
    monkeypatch.setattr(prime, "ENABLED", True)
    monkeypatch.setattr(prime, "_config_enabled", lambda: True)
    t = startup.start_warm_up()
    t.join(timeout=5)
    assert seen == ["warm", ("prime", ("quick", "planner"))]


def test_prime_stops_between_lineages_when_a_turn_starts(monkeypatch):
    from core import prime

    class M:
        def __init__(self):
            self.calls = 0

        def invoke(self, msgs, **kw):
            self.calls += 1
            prime.set_busy(True)  # the user typed mid-sequence
            return AIMessage(content="")

    model = M()
    monkeypatch.setattr("core.llms.get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    monkeypatch.setattr(prime, "ENABLED", True)
    try:
        assert prime.prime("STABLE") == 1 == model.calls
    finally:
        prime.set_busy(False)


def test_an_empty_arguments_object_is_a_call_for_a_no_argument_tool():
    """`{"arguments": {}}` is how the grammar spells a call to a tool that takes nothing (an MCP
    tool without parameters); it must parse as a call, not as 'no tool call emitted'."""
    args, refusal = ex._parse_arguments('{"arguments": {}}')
    assert args == {} and refusal == ""


def test_tool_brief_carries_the_whole_description():
    """The brief replaces the schema block the native bind put in the system prompt, so the
    model must still read every argument's meaning — `remember`'s `replaces` semantics live in
    the last third of its description (measured 2026-09-04: with it cut off, a correction was
    stored as a duplicate instead of superseding the old fact)."""
    from tools.registry import tools_by_name

    brief = ex.tool_brief(tools_by_name["remember"])
    assert "supersedes" in brief and "…" not in brief
