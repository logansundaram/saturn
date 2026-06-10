"""commands/rewind.py — drop_last_turn (the engine behind /rewind and /retry full): boundary
detection, scratch clearing, and the autosave refresh."""

import json
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from commands._session import _autosave_file
from commands.rewind import drop_last_turn


def _ctx(messages, **scratch):
    state = {
        "messages": messages,
        "current_query": scratch.get("current_query", "second question"),
        "plan": scratch.get("plan", [{"step_id": 1, "label": "x", "status": "done",
                                      "intended_tool": None}]),
        "iteration": 3,
        "agent_nudges": 1,
        "replans": 1,
        "tools_called": ["web_search"],
        "tool_results": ["web_search(q) -> r"],
        "documents_retrieved": ["d"],
        "tool_events": [{"name": "web_search"}],
    }
    return SimpleNamespace(state=state)


def test_drop_last_turn_removes_from_last_human(isolated_paths):
    msgs = [
        HumanMessage(content="first question"),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "t1"}]),
        ToolMessage(content="result", tool_call_id="t1"),
        AIMessage(content="second answer"),
    ]
    ctx = _ctx(msgs)
    dropped = drop_last_turn(ctx)
    assert dropped == "second question"
    assert [m.content for m in ctx.state["messages"]] == ["first question", "first answer"]


def test_drop_last_turn_clears_per_turn_scratch(isolated_paths):
    ctx = _ctx([HumanMessage(content="q"), AIMessage(content="a")])
    drop_last_turn(ctx)
    s = ctx.state
    assert s["current_query"] == ""
    assert s["plan"] == [] and s["tools_called"] == [] and s["tool_results"] == []
    assert s["documents_retrieved"] == [] and s["tool_events"] == []
    assert s["iteration"] == 0 and s["agent_nudges"] == 0 and s["replans"] == 0


def test_drop_last_turn_refreshes_autosave(isolated_paths):
    ctx = _ctx([
        HumanMessage(content="keep me"),
        AIMessage(content="kept answer"),
        HumanMessage(content="rewind me"),
        AIMessage(content="rewound answer"),
    ])
    drop_last_turn(ctx)
    payload = json.loads(_autosave_file().read_text(encoding="utf-8"))
    contents = [m["data"]["content"] for m in payload["messages"]]
    assert "rewind me" not in contents and "keep me" in contents


def test_drop_last_turn_empty_conversation(isolated_paths):
    ctx = _ctx([])
    assert drop_last_turn(ctx) is None
    # Repeatable until empty, then None again.
    ctx = _ctx([HumanMessage(content="only"), AIMessage(content="a")])
    assert drop_last_turn(ctx) == "only"
    assert drop_last_turn(ctx) is None
