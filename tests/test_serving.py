"""The serving layer's measured slices (transplanted from the engine isolate, M5 — the SPLIT the
transplant proposal named: think-per-task, num_predict caps, the think-rejection fallback, and the
repetition retry trigger; the per-family temperature ladders, the token budget and the
single-HumanMessage prompt stayed behind).

  - `think` is set EXPLICITLY per task, never left to the model's default (every local model
    Saturn targets defaults to thinking ON) — ON for the planner only (its rationale decides
    "answer or ask"; without it the 9b stubs open requests to a lone ask_user), OFF for every
    other task (measured: think-off fixed `absence` 66 → 100 %, `no_capability` 2/5 → 5/5).
  - every generation carries a `num_predict` circuit breaker: a whitespace loop under a JSON
    grammar or a small model that starts repeating lands as a truncated generation, not a full
    context window.
  - a model with no thinking template 400s on `think` in EITHER direction, so the rejection is
    caught ONCE per model tag and the call retried without the flag — never papered over globally.
  - a degenerate draw (`reviewedreviewedreviewed…`) triggers a repeat penalty on the NEXT rung
    only — never a global setting, which would corrupt exact tool arguments.
"""

import types

import pytest
from langchain.messages import AIMessage, HumanMessage

import textutil
from core import llms, serving, structured
from nodes import execute as ex


# ── the task table ──────────────────────────────────────────────────────────────────────────


def test_no_task_thinks_and_the_planner_reasons_in_the_grammar():
    # The planner's rationale is where it decides "answer directly or ask": with think off and
    # no rationale the 9b planned a lone stub for "write me a story" (4/4 replayed draws,
    # 2026-09-03). Since 2026-09-08 the rationale is a bounded FIRST field of the plan grammar
    # (structured chain-of-thought) and the planner runs think off like every other task.
    for task in ("plan", "judge", "tool_args", "reasoning", "answer", "correction"):
        assert not serving.thinks(task), task
    from core import structured as st

    fmt = st.plan_format(["read_file"])
    assert list(fmt["properties"]) == ["rationale", "plan"]  # the rationale decodes FIRST
    assert fmt["properties"]["rationale"]["maxLength"] == st.RATIONALE_MAX_CHARS
    assert fmt["required"] == ["rationale", "plan"]
    assert st.PLAN_SHAPE.startswith('Respond with ONLY this JSON and nothing else: {"rationale"')
    assert st.to_steps(st._PlanOut(rationale="x", plan=[])) == []  # to_steps ignores it
    assert not serving.thinks("some-unknown-task")   # unknown → the strictest safe shape


def test_every_task_has_an_output_bound():
    for task in ("plan", "judge", "tool_args", "reasoning", "answer", "correction", "unknown"):
        assert serving.num_predict(task) > 0
    assert serving.num_predict("judge") <= serving.num_predict("plan")


def test_role_maps_to_a_default_task():
    assert serving.task_for_role("planner") == "plan"
    assert serving.task_for_role("judge") == "judge"
    assert serving.task_for_role("tool_caller") == "tool_args"
    assert serving.task_for_role("synthesizer") == "answer"


# ── the invoke kwargs ───────────────────────────────────────────────────────────────────────


def _ollama(monkeypatch):
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)


def test_invoke_kwargs_carry_think_num_predict_and_num_ctx(monkeypatch):
    _ollama(monkeypatch)
    kw = structured._invoke_kwargs("judge", {"type": "object"}, 0.0)
    assert kw["reasoning"] is False
    assert kw["options"]["num_predict"] == serving.num_predict("judge")
    assert "num_ctx" in kw["options"] and kw["options"]["temperature"] == 0.0
    assert kw["format"] == {"type": "object"}
    kw = structured._invoke_kwargs("planner", None, 0.0)
    assert kw["reasoning"] is False and "format" not in kw   # explicit, never the model default


def test_invoke_kwargs_task_override_and_repetition(monkeypatch):
    _ollama(monkeypatch)
    kw = structured._invoke_kwargs("tool_caller", None, 0.4, task="reasoning")
    assert kw["options"]["num_predict"] == serving.num_predict("reasoning")
    assert "repeat_penalty" not in kw["options"]
    kw = structured._invoke_kwargs("tool_caller", None, 0.4, task="reasoning", repetition=True)
    assert kw["options"]["repeat_penalty"] > 1.0 and kw["options"]["repeat_last_n"] > 0


def test_invoke_kwargs_omit_think_once_the_model_rejected_it(monkeypatch):
    _ollama(monkeypatch)
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", {structured._model_tag("judge")})
    kw = structured._invoke_kwargs("judge", None, 0.0)
    assert "reasoning" not in kw and "num_predict" in kw["options"]


def test_non_ollama_role_gets_no_kwargs(monkeypatch):
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: False)
    assert structured._invoke_kwargs("judge", None, 0.0) == {}


# ── the think-rejection fallback ────────────────────────────────────────────────────────────


class _Rejects:
    """A runnable whose daemon 400s on `think` — the first call with the flag raises."""

    def __init__(self, reply="ok"):
        self.calls = []
        self.reply = reply

    def invoke(self, msgs, **kw):
        self.calls.append(kw)
        if "reasoning" in kw:
            raise RuntimeError('400: "qwen2.5:3b" does not support thinking')
        return AIMessage(content=self.reply)

    def stream(self, msgs, **kw):
        self.calls.append(kw)
        if "reasoning" in kw:
            raise RuntimeError("thinking is not supported by this model")
        yield AIMessage(content=self.reply)


def test_generate_retries_without_think_and_remembers_the_tag(monkeypatch):
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    r = _Rejects()
    out = llms.generate(r, [HumanMessage("q")], tag="qwen2.5:3b", reasoning=False, options={})
    assert out.content == "ok"
    assert [("reasoning" in c) for c in r.calls] == [True, False]
    assert "qwen2.5:3b" in llms._NO_THINK_SUPPORT


def test_generate_reraises_an_unrelated_error(monkeypatch):
    class Boom:
        def invoke(self, msgs, **kw):
            raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError, match="connection refused"):
        llms.generate(Boom(), [], tag="m", reasoning=False)


def test_stream_retries_without_think(monkeypatch):
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    r = _Rejects("streamed")
    chunks = list(llms.stream(r, [HumanMessage("q")], tag="m2", reasoning=False))
    assert [c.content for c in chunks] == ["streamed"]
    assert "m2" in llms._NO_THINK_SUPPORT


# ── the repetition trigger ──────────────────────────────────────────────────────────────────


def test_looks_repetitive_detects_the_two_loop_shapes():
    assert textutil.looks_repetitive("reviewed" * 6)
    assert textutil.looks_repetitive("\n".join(["same line"] * 5))
    assert not textutil.looks_repetitive("The quick brown fox jumps over the lazy dog.")
    assert not textutil.looks_repetitive('{"a": "b", "c": "d", "e": "f"}')
    assert not textutil.looks_repetitive("")


def test_reasoning_call_retries_a_degenerate_draw_with_a_repeat_penalty(monkeypatch):
    seen = []

    class M:
        def __init__(self):
            self.replies = ["reviewed" * 8, "A sensible sentence."]

        def invoke(self, msgs, **kw):
            seen.append(kw)
            return AIMessage(content=self.replies.pop(0))

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)
    content, _resp = ex._reasoning_call("ctx")
    assert content == "A sensible sentence."
    assert "repeat_penalty" not in seen[0]["options"]
    assert seen[1]["options"]["repeat_penalty"] > 1.0
    assert seen[0]["reasoning"] is False          # a reasoning step never thinks


def test_tool_call_ladder_arms_the_repeat_penalty_after_a_degenerate_text_answer(monkeypatch):
    from tools.registry import tools_by_name

    seen = []

    class M:
        def __init__(self):
            self.replies = [
                AIMessage(content="okokokokokokokokok"),   # degenerate, no call
                AIMessage(content='{"arguments": {"expression": "1+1"}}'),
            ]

        def invoke(self, msgs, **kw):
            seen.append(kw)
            return self.replies.pop(0)

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["calculate"], "ctx")
    assert args == {"expression": "1+1"} and failure is None
    assert "repeat_penalty" not in seen[0]["options"] and seen[1]["options"]["repeat_penalty"] > 1.0
    assert seen[0]["options"]["num_predict"] == serving.num_predict("tool_args")


def test_structured_arms_the_repeat_penalty_after_a_degenerate_unparseable_draw(monkeypatch):
    """The isolate wired the repetition trigger into core.structured too: a looping,
    unparseable JSON draw is retried with the repeat penalty on the next rung only."""
    from core import structured as st

    seen = []

    class M:
        def __init__(self):
            self.replies = ['{"rectify": ' + "tru" * 20, '{"rectify": false, "reasoning": "ok"}']

        def invoke(self, msgs, **kw):
            seen.append(kw)
            return AIMessage(content=self.replies.pop(0))

    model = M()
    monkeypatch.setattr(llms, "get_model", lambda role: model)
    monkeypatch.setattr(st, "_role_is_ollama", lambda role: True)
    out = st.structured("judge", [HumanMessage("q")], st.RectifyBool, st.RECTIFY_FORMAT,
                        st.RECTIFY_SHAPE, default=None)
    assert out.rectify is False
    assert "repeat_penalty" not in seen[0]["options"] and seen[1]["options"]["repeat_penalty"] > 1.0


# ── the tool_payload bound (2026-09-02): write_file/edit_file carry the file in their args ────


def test_tool_payload_task_exists_and_is_larger_than_tool_args():
    """write_file/edit_file carry the WHOLE payload in their arguments (measured 2026-09-02:
    a 2,956-char story is ~690 tokens; the 512 tool_args cap cut every attempt at exactly 512
    with done_reason=length and no call parsed). The payload cap is still a circuit breaker."""
    assert serving.num_predict("tool_payload") >= 4096
    assert serving.num_predict("tool_payload") > serving.num_predict("tool_args")
    assert serving.thinks("tool_payload") is False
    assert serving.task_of("tool_payload").strict is True


def test_task_for_tool_routes_write_tools_to_the_payload_cap():
    assert ex._task_for_tool("write_file") == "tool_payload"
    assert ex._task_for_tool("edit_file") == "tool_payload"
    assert ex._task_for_tool("calculate") == "tool_args"
    assert ex._task_for_tool("mcp_remote_thing") == "tool_args"


def test_generate_tool_call_uses_the_payload_cap_for_write_file(monkeypatch):
    from tools.registry import tools_by_name

    seen = []

    class M:
        def invoke(self, msgs, **kw):
            seen.append(kw)
            return AIMessage(content='{"arguments": {"file_path": "a.txt", "content": "hi"}}')

    monkeypatch.setattr(ex, "get_model", lambda role: M())
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)
    args, failure, _ = ex._generate_tool_call(tools_by_name["write_file"], "ctx")
    assert failure is None and args["content"] == "hi"
    assert seen[0]["options"]["num_predict"] == serving.num_predict("tool_payload")


def test_judge_cap_clears_a_long_rationale():
    """Run 16 (2026-09-02): the rectify judge's first attempt hit exactly 512 tokens writing its
    `reasoning` field and the JSON never closed; the retry happened to finish in 241. The cap is
    a breaker, not a budget — it must sit above a verbose-but-healthy verdict."""
    assert serving.num_predict("judge") >= 1024
    assert serving.num_predict("judge") <= serving.num_predict("plan")


def test_structured_logs_a_truncated_draw(monkeypatch):
    import diag

    lines = []
    monkeypatch.setattr(diag, "log", lambda s: lines.append(s))
    monkeypatch.setattr(structured, "_role_is_ollama", lambda role: True)

    class M:
        def invoke(self, msgs, **kw):
            return AIMessage(content='{"reasoning":"cut', response_metadata={"done_reason": "length"})

    monkeypatch.setattr("core.llms.get_model", lambda role: M())
    out = structured.structured("judge", [], structured.RectifyBool, structured.RECTIFY_FORMAT,
                                structured.RECTIFY_SHAPE,
                                default=structured.RectifyBool(rectify=False, reasoning="d"))
    assert out.reasoning == "d"
    assert any("cut off at num_predict" in l for l in lines)
