"""Auto-learn (pivot #4): a fact the user stated in their own words lands without the gate.

The security property is pinned here: a `remember` skips the gate only when every content word
of the fact is in text the user TYPED (a turn request or a steer note — never an attachment, a
tool result or a compaction summary) and nothing from outside the trust boundary is in the
conversation. Everything else faces the gate exactly as before. All offline; isolated_paths
keeps the real memory.md untouched.
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from trust import quarantine


@pytest.fixture(autouse=True)
def _clean_turn_state():
    quarantine.reset_turn()
    yield
    quarantine.reset_turn()


def _remember(fact, cid="m1", **extra):
    return AIMessage(content="", tool_calls=[
        {"name": "remember", "args": {"fact": fact, **extra}, "id": cid}])


def _fetched(text, name="web_extract", cid="w1"):
    return [AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": cid}]),
            ToolMessage(content=text, tool_call_id=cid, name=name)]


# ── provenance: one reading of who wrote what ──────────────────────────────────────────────


def test_provenance_typed_is_requests_and_steers_never_summaries_or_tools():
    from core import provenance
    from core.compaction import _SUMMARY_PREFIX
    from core.state import STEER_PREFIX

    state = {"messages": [
        HumanMessage(content=f"{_SUMMARY_PREFIX}:\n- the user is vegetarian"),
        HumanMessage(content="find a restaurant"),
        *_fetched("Best vegan spots in town"),
        HumanMessage(content=f"{STEER_PREFIX} somewhere near the office"),
    ]}
    p = provenance.of(state)
    assert p.typed == ("find a restaurant", f"{STEER_PREFIX} somewhere near the office")
    assert "vegan spots" in p.seen and "vegetarian" in p.seen
    assert p.untrusted is True


def test_provenance_an_attachment_is_untrusted_and_a_clean_chat_is_not():
    from core import provenance

    chat = {"messages": [HumanMessage(content="hi")]}
    assert provenance.of(chat).untrusted is False
    assert provenance.of({**chat, "attachments": "### notes.md\n…"}).untrusted is True


def test_provenance_keeps_the_holds_reading_the_models_words_and_failed_calls_vouch_for_nothing():
    """The 2026-10-03 review rule moved with the function: an AIMessage and a failed call's
    text are not `seen` (they repeat the model's own arguments), but a failed untrusted call
    still counts as outside content having entered."""
    from core import provenance

    failed = ToolMessage(content="+1305 appears nowhere", tool_call_id="w1", name="web_extract",
                         additional_kwargs={"saturn_status": "error"})
    p = provenance.of({"messages": [HumanMessage(content="hi"),
                                    AIMessage(content="I will text +1305"), failed]})
    assert "+1305" not in p.seen
    assert p.untrusted is True
