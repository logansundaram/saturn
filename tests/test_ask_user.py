"""
ask_user — the mid-run question to the human (2026-07-06).

The tool pauses the graph via LangGraph interrupt() (the gate/plan-review checkpoint machinery);
the typed answer resumes as the tool's observation. Offline pieces under test: the resume-value
contract (string answer / headless bare-True / empty reply), the tool_node GraphInterrupt
re-raise (control flow must never be swallowed as a tool error), the read_only tier (asking never
faces the approval gate), the arg-recovery + synonym tables, and the REPL prompt renderer.
"""

import pytest
from langchain.messages import AIMessage
from langgraph.errors import GraphInterrupt

import nodes.tools as nodes_tools
import tools.interaction as interaction
from core.structured import norm_tool
from core.tool_args import coerce_args, schema_hint
from tools.registry import risk_of, tools_by_name


# --- the resume-value contract ---------------------------------------------------------------

def _invoke(monkeypatch, resume_value, question="Which file?"):
    monkeypatch.setattr(interaction, "interrupt", lambda payload: resume_value)
    return tools_by_name["ask_user"].invoke({"question": question})


def test_typed_answer_becomes_the_observation(monkeypatch):
    out = _invoke(monkeypatch, "  the June report  ")
    assert out == "The user answered: the June report"


def test_headless_bare_true_reports_no_answer(monkeypatch):
    # The headless approver resolves unknown interrupts with a bare True — the tool must report
    # the absence honestly, never stringify True into a fake answer.
    out = _invoke(monkeypatch, True)
    assert out.startswith("[no answer")
    assert "True" not in out


def test_empty_reply_reports_no_answer(monkeypatch):
    # Enter / Ctrl-C at the prompt returns "" — same honest degradation.
    assert _invoke(monkeypatch, "").startswith("[no answer")
    assert _invoke(monkeypatch, "   ").startswith("[no answer")


def test_interrupt_payload_carries_type_and_question(monkeypatch):
    seen = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return "ok"

    monkeypatch.setattr(interaction, "interrupt", fake_interrupt)
    tools_by_name["ask_user"].invoke({"question": " Which one? "})
    assert seen == {"type": "ask_user", "question": "Which one?"}


# --- tool_node: the interrupt is control flow, not a tool error --------------------------------

def _tool_call_state(name, args):
    return {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "tc1"}])
        ]
    }


def test_tool_node_reraises_graph_interrupt(monkeypatch):
    # A swallowed GraphInterrupt would answer the question with the exception repr and never
    # reach the human — the generic except in tool_node must let it bubble to LangGraph.
    def raising_interrupt(payload):
        raise GraphInterrupt()

    monkeypatch.setattr(interaction, "interrupt", raising_interrupt)
    with pytest.raises(GraphInterrupt):
        nodes_tools.tool_node(_tool_call_state("ask_user", {"question": "hm?"}))


def test_tool_node_still_absorbs_ordinary_tool_errors(monkeypatch):
    # The re-raise is scoped to GraphInterrupt: a real failure still lands on the step as an
    # error observation instead of crashing the turn.
    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(interaction, "interrupt", boom)
    out = nodes_tools.tool_node(_tool_call_state("ask_user", {"question": "hm?"}))
    assert out["messages"][0].content.startswith("Error calling ask_user")
    assert out["messages"][0].additional_kwargs["saturn_status"] == "error"


# --- registration + recovery tables ------------------------------------------------------------

def test_ask_user_is_read_only():
    # Asking mutates nothing — it must never face the approval gate (the question IS the
    # interaction), and the planner's clarify-first guidance depends on it being free.
    assert risk_of("ask_user") == "read_only"


def test_coerce_args_maps_aliases_onto_question():
    assert coerce_args("ask_user", {"prompt": "Which file?"}) == {"question": "Which file?"}
    assert coerce_args("ask_user", {"question": "x"}) == {"question": "x"}
    assert coerce_args("ask_user", {}) is None  # missing → retry with the schema hint
    assert "ask_user(question=" in schema_hint("ask_user", "missing arg")


def test_norm_tool_synonyms_resolve_to_ask_user():
    valid = {"ask_user"}
    for raw in ("ask", "ask_human", "ask_the_user", "user_input", "question", "ask_user"):
        assert norm_tool(raw, valid) == "ask_user"


# --- the REPL prompt renderer -------------------------------------------------------------------

def test_answer_question_returns_the_typed_line(monkeypatch, capsys):
    import importlib

    # importlib on purpose: `import tui.ui.prompt as m` binds tui.ui's re-exported prompt()
    # FUNCTION (the package attribute shadows the submodule under `as`-import semantics).
    prompt_mod = importlib.import_module("tui.ui.prompt")

    monkeypatch.setattr(prompt_mod, "ask", lambda text: "the June report")
    got = prompt_mod.answer_question({"type": "ask_user", "question": "Which file?"})
    assert got == "the June report"
    assert "Which file?" in capsys.readouterr().out


def test_answer_question_tolerates_a_missing_question(monkeypatch, capsys):
    import importlib

    # importlib on purpose: `import tui.ui.prompt as m` binds tui.ui's re-exported prompt()
    # FUNCTION (the package attribute shadows the submodule under `as`-import semantics).
    prompt_mod = importlib.import_module("tui.ui.prompt")

    monkeypatch.setattr(prompt_mod, "ask", lambda text: "")
    assert prompt_mod.answer_question({}) == ""
    assert "(no question given)" in capsys.readouterr().out
