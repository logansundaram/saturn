"""nodes/synthesize.cancel_orphaned_calls — the forced-landing orphan guard. When the
iteration cap or token budget routes to synthesize while the trailing AIMessage still carries
tool_calls, every call must get a cancellation ToolMessage, or the carried conversation (and its
autosave) holds an assistant tool_use with no tool_result — a hard provider error next turn."""

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from nodes.synthesize import cancel_orphaned_calls


def test_unanswered_calls_get_cancellation_messages():
    last = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": "x"}, "id": "t1"},
            {"name": "read_file", "args": {"file_path": "a.md"}, "id": "t2"},
        ],
    )
    out = cancel_orphaned_calls(last)
    assert [m.tool_call_id for m in out] == ["t1", "t2"]
    assert all(isinstance(m, ToolMessage) for m in out)
    assert all("Not executed" in str(m.content) for m in out)


def test_no_tool_calls_no_cancellations():
    assert cancel_orphaned_calls(AIMessage(content="a draft answer")) == []
    assert cancel_orphaned_calls(HumanMessage(content="a question")) == []
    assert cancel_orphaned_calls(None) == []
