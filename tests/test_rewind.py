"""commands/rewind.py — drop_last_turn (the engine behind /rewind and /retry full): boundary
detection, scratch clearing, and the autosave refresh."""

import json
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from commands._session import _autosave_file
from commands.conversation import drop_last_turn


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
        "gate_events": [{"decision": "approved", "mode": "prompt"}],
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
    # The human-gate record goes too: "gate_events empty" must always mean "never asked" — a
    # rewound turn's decisions can't linger for /glass or an export to misread.
    assert s["gate_events"] == []
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


def test_drop_last_turn_skips_steer_and_summary_boundaries(isolated_paths):
    """A standalone mid-turn steer note and a compaction summary are HumanMessages but NOT turn
    boundaries: rewinding a steered turn must drop the whole turn from its real question, and a
    summary must never be mistaken for a turn to drop."""
    from core.compaction import _SUMMARY_PREFIX
    from core.state import STEER_PREFIX

    msgs = [
        HumanMessage(content=f"{_SUMMARY_PREFIX}:\nolder turns, summarized"),
        HumanMessage(content="real question"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "t1"}]),
        ToolMessage(content="result", tool_call_id="t1"),
        HumanMessage(content=f"{STEER_PREFIX} actually check the other repo"),
        AIMessage(content="steered answer"),
    ]
    ctx = _ctx(msgs)
    assert drop_last_turn(ctx) == "real question"
    # The whole steered turn went (question, scratchpad, steer note, answer); the summary stayed.
    remaining = ctx.state["messages"]
    assert len(remaining) == 1 and str(remaining[0].content).startswith(_SUMMARY_PREFIX)
    # The summary alone is not a rewindable turn.
    assert drop_last_turn(ctx) is None


def test_drop_last_turn_empty_conversation(isolated_paths):
    ctx = _ctx([])
    assert drop_last_turn(ctx) is None
    # Repeatable until empty, then None again.
    ctx = _ctx([HumanMessage(content="only"), AIMessage(content="a")])
    assert drop_last_turn(ctx) == "only"
    assert drop_last_turn(ctx) is None


def test_rewinding_the_only_turn_clears_the_autosave(isolated_paths):
    """write_autosave skips an empty conversation by design, so rewinding the ONLY turn must
    clear the slot explicitly — otherwise a crash/quit + /resume resurrects the rewound turn."""
    ctx = _ctx([HumanMessage(content="only question"), AIMessage(content="only answer")])
    # Simulate the end-of-turn autosave that captured the turn.
    from commands._session import write_autosave

    write_autosave(ctx.state)
    assert _autosave_file().exists()
    drop_last_turn(ctx)
    assert not _autosave_file().exists()
