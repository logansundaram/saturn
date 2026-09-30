"""The background model calls — compaction's summary, the memory review's proposals, /init's
SATURN.md draft — go through `core.llms.invoke_kwargs` with a TASK, like the agent's own call:
thinking explicitly OFF (a qwen3.x default would put reasoning into a summary every later turn
carries) and a `num_predict` bound (a repetition loop must not run to the context window).
Offline: get_model is a fake that records the kwargs."""

from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage, HumanMessage

from core import llms


@pytest.fixture
def seen(monkeypatch):
    calls = []

    class _Fake:
        def invoke(self, messages, **kwargs):
            calls.append(kwargs)
            return AIMessage(content='# Project\n{"facts": []}')

    monkeypatch.setattr(llms, "get_model", lambda *a, **k: _Fake())
    monkeypatch.setattr(llms, "_NO_THINK_SUPPORT", set())
    return calls


def _bounded_and_think_off(kwargs):
    assert kwargs.get("reasoning") is False, kwargs
    assert kwargs["options"].get("num_predict"), kwargs
    assert kwargs["options"].get("num_ctx"), kwargs


def test_compaction_summary_is_bounded_and_think_off(seen):
    from core import compaction

    compaction._llm_summary([HumanMessage(content="hi"), AIMessage(content="hello")])
    (kwargs,) = seen
    _bounded_and_think_off(kwargs)


def test_memory_review_is_bounded_and_think_off(seen):
    from core import memory_review

    memory_review.llm_candidates([HumanMessage(content="I live in Berlin"), AIMessage(content="ok")])
    (kwargs,) = seen
    _bounded_and_think_off(kwargs)
    assert "format" in kwargs  # still the constrained decode


def test_init_draft_is_bounded_and_think_off(seen, isolated_paths, tmp_path):
    import commands  # noqa: F401 — registers every command
    from commands._framework import CommandContext, dispatch
    from core import workspace

    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "main.py").write_text("print('x')\n", encoding="utf-8")
    workspace.set_root(folder)
    dispatch("/init", CommandContext(state={}, make_initial_state=dict, db_path=""))
    (kwargs,) = seen
    _bounded_and_think_off(kwargs)
    assert (folder / "SATURN.md").read_text().startswith("# Project")
