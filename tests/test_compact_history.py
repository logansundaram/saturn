"""agent._compact_history — the mechanical per-turn compaction: older turns collapse to Q&A,
the most recent turn keeps its full ReAct scratchpad (what makes follow-ups work)."""

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from agent import _compact_history


def _turn(q, with_tools=False, answer="ans"):
    msgs = [HumanMessage(content=q)]
    if with_tools:
        msgs.append(
            AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "c1"}])
        )
        msgs.append(ToolMessage(content="42", tool_call_id="c1"))
    msgs.append(AIMessage(content=answer))
    return msgs


def test_recent_turn_scratchpad_kept_older_compacted():
    msgs = _turn("first", with_tools=True) + _turn("second", with_tools=True)
    out = _compact_history(msgs)
    # Older turn: only Human + final AI survive.
    assert out[0].content == "first"
    assert isinstance(out[1], AIMessage) and out[1].content == "ans"
    # Recent turn: intact, scratchpad and all.
    recent = out[2:]
    assert [type(m).__name__ for m in recent] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]


def test_empty_and_tool_call_ai_messages_dropped_from_old_turns():
    msgs = _turn("old", with_tools=True) + [HumanMessage(content="new")]
    out = _compact_history(msgs)
    assert all(not getattr(m, "tool_calls", None) for m in out[:-1])
    assert not any(isinstance(m, ToolMessage) for m in out[:-1])


def test_keep_zero_strips_everything():
    msgs = _turn("only", with_tools=True)
    out = _compact_history(msgs, keep_recent_turns=0)
    assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage"]


def test_empty_history():
    assert _compact_history([]) == []
