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


# ── auto-compaction (app/session._maybe_autocompact) ─────────────────────────────────────────


def _research_turn(q, n_reads, size, cid="r"):
    """One turn that gathered `n_reads` observations of `size` characters each."""
    msgs = [HumanMessage(content=q)]
    for i in range(n_reads):
        msgs.append(AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"file_path": f"{i}.md"}, "id": f"{cid}{i}"}]))
        msgs.append(ToolMessage(content=f"HEAD{i} " + "x" * size + f" TAIL{i}",
                                tool_call_id=f"{cid}{i}", name="read_file",
                                additional_kwargs={"saturn_status": "done"}))
    msgs.append(AIMessage(content="the answer"))
    return msgs


def _autocompact(monkeypatch, messages, used, window=32000):
    import core.llms as llms
    from app import session
    from core import compaction

    notes = []
    monkeypatch.setattr(llms, "active_context_window", lambda: window)
    monkeypatch.setattr(compaction, "_llm_summary", lambda older: "SUMMARY")
    monkeypatch.setattr(session.ui, "note", notes.append)
    state = session._maybe_autocompact({"messages": list(messages), "context_tokens": used})
    return state["messages"], notes


def _chars(msgs):
    return sum(len(str(m.content)) for m in msgs)


def test_autocompact_shrinks_the_turn_that_filled_the_window(monkeypatch):
    """A single research turn fills the window. There is no older turn to fold, so the trigger
    used to do nothing and the next request overflowed num_ctx. Its observations are trimmed
    instead: every call and its outcome stay, each result keeps a head and a tail."""
    turn = _research_turn("research llamas", n_reads=10, size=11000)
    out, notes = _autocompact(monkeypatch, turn, used=29000)

    assert _chars(out) < _chars(turn) / 4
    assert [type(m) for m in out] == [type(m) for m in turn]         # nothing dropped
    assert out[0].content == "research llamas" and out[-1].content == "the answer"
    tool_out = [m for m in out if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_out] == [f"r{i}" for i in range(10)]
    assert all(m.additional_kwargs == {"saturn_status": "done"} for m in tool_out)
    assert "HEAD3" in tool_out[3].content and "TAIL3" in tool_out[3].content
    assert "compacted" in tool_out[3].content                        # the model is told
    assert notes and "trimmed 10 tool result" in notes[-1]


def test_autocompact_leaves_the_recent_turn_alone_when_folding_older_turns_is_enough(monkeypatch):
    older = _research_turn("first", n_reads=8, size=11000, cid="a")
    recent = _research_turn("second", n_reads=1, size=2000, cid="b")
    out, notes = _autocompact(monkeypatch, older + recent, used=29000)

    assert "SUMMARY" in out[0].content
    assert out[1:] == recent                                         # verbatim: a follow-up can refer back
    assert not any("trimmed" in n for n in notes)


def test_autocompact_does_nothing_below_the_threshold(monkeypatch):
    turn = _research_turn("research llamas", n_reads=10, size=11000)
    out, notes = _autocompact(monkeypatch, turn, used=9000)
    assert out == turn and notes == []
