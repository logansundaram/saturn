"""nodes/ground._recent_exchanges — the Q&A recap the planner/synthesizer (which read
`context`, not raw `messages`) resolve follow-ups against. Pairing the wrong question with the
latest answer corrupts every downstream node, so the boundary rules are pinned here."""

from langchain.messages import AIMessage, HumanMessage

from core.compaction import _SUMMARY_PREFIX
from nodes.ground import _recent_exchanges
from core.state import STEER_PREFIX


def test_pairs_questions_with_final_answers():
    msgs = [
        HumanMessage(content="first question"),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "t1"}]),
        AIMessage(content="second answer"),
    ]
    out = _recent_exchanges(msgs)
    assert "first question" in out and "first answer" in out
    assert "second question" in out and "second answer" in out


def test_in_flight_query_not_paired():
    msgs = [HumanMessage(content="just asked, no answer yet")]
    assert _recent_exchanges(msgs) == ""


def test_summary_message_never_becomes_the_question():
    """After compaction the summary HumanMessage leads `messages`; it must not pair with the
    most recent answer and push the real question out of the recap."""
    msgs = [
        HumanMessage(content=f"{_SUMMARY_PREFIX}:\nolder turns, summarized"),
        HumanMessage(content="real question"),
        AIMessage(content="real answer"),
    ]
    out = _recent_exchanges(msgs)
    assert "real question" in out and "real answer" in out
    assert _SUMMARY_PREFIX not in out


def test_steer_note_never_becomes_the_question():
    msgs = [
        HumanMessage(content="real question"),
        HumanMessage(content=f"{STEER_PREFIX} check the other repo"),
        AIMessage(content="steered answer"),
    ]
    out = _recent_exchanges(msgs)
    assert "real question" in out and "steered answer" in out
    assert STEER_PREFIX not in out


def test_failed_turn_question_superseded():
    """A question left unanswered by a failed turn must not pair with the NEXT turn's answer —
    the latest real question wins."""
    msgs = [
        HumanMessage(content="failed-turn question"),
        HumanMessage(content="retried question"),
        AIMessage(content="the answer"),
    ]
    out = _recent_exchanges(msgs)
    assert "retried question" in out and "the answer" in out
    assert "failed-turn question" not in out
