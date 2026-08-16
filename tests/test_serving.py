"""The serving layer's measured slices (transplanted from the engine isolate, M5 — the SPLIT the
transplant proposal named: think-per-task, num_predict caps, the think-rejection fallback, and the
repetition retry trigger; the per-family temperature ladders, the token budget and the
single-HumanMessage prompt stayed behind).

  - `think` is set EXPLICITLY per task, never left to the model's default (every local model
    Saturn targets defaults to thinking ON): the planner is the ONE task that keeps its rationale
    (measured: the floor tier's plan collapses to a one-step stub without it); judges, tool
    arguments, reasoning prose and the answer run without (measured: think-off fixed `absence`
    66 → 100 %, `no_capability` 2/5 → 5/5; planner latency 5.7 s → 0.8 s on the floor tier).
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


def test_only_the_planner_thinks():
    assert serving.thinks("plan")
    for task in ("judge", "tool_args", "reasoning", "answer", "correction"):
        assert not serving.thinks(task), task
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
    assert kw["reasoning"] is True and "format" not in kw


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
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "1+1"},
                                                   "id": "c1", "type": "tool_call"}]),
            ]

        def bind_tools(self, tools):
            return self

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
