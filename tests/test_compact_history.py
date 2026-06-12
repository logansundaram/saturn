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


def test_steer_note_is_not_a_turn_boundary():
    """A standalone mid-turn steer note (plan_gate's appended HumanMessage) must not be treated
    as the most recent turn's start — that would compact away the real question's scratchpad
    this function promises to keep."""
    from core.state import STEER_PREFIX

    msgs = _turn("old turn") + [
        HumanMessage(content="recent question"),
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "c1"}]),
        ToolMessage(content="42", tool_call_id="c1"),
        HumanMessage(content=f"{STEER_PREFIX} no, the OTHER file"),
        AIMessage(content="steered answer"),
    ]
    out = _compact_history(msgs)
    # The boundary is the real question: its full scratchpad (tool call + observation) survives.
    boundary = next(i for i, m in enumerate(out) if m.content == "recent question")
    assert any(isinstance(m, ToolMessage) for m in out[boundary:])
    # The old turn still compacted to Q&A.
    assert out[0].content == "old turn"


def test_summary_is_not_a_turn_boundary():
    from core.compaction import _SUMMARY_PREFIX

    msgs = [HumanMessage(content=f"{_SUMMARY_PREFIX}:\nolder stuff")] + _turn(
        "only", with_tools=True
    )
    out = _compact_history(msgs)
    # The summary is carried history; the real turn behind it keeps its scratchpad.
    assert str(out[0].content).startswith(_SUMMARY_PREFIX)
    assert any(isinstance(m, ToolMessage) for m in out)
