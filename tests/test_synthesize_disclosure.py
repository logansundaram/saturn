"""synthesize's terminal delta must disclose UNCONDITIONALLY (transplanted from the
agent_architecture isolate, M4).

Every mechanical trailer used to be guarded on `content.strip()` — on the model having written
prose — so an empty generation produced a turn with no answer, no statement that a rejected
write had not happened, and no Sources footer. That is a fail-OPEN on the one path the engine
promises to be fail-closed, and it was measured live (`held.guard.reject_write` returned '' on
one model while passing on two others). Offline: `_final_updates` is a pure function.
"""

from nodes.synthesize import INCIDENTS_NOTE_HEADER, NO_ANSWER_TEXT, _final_updates


def _delta(text, incidents=(), sources=()):
    buf = {"text": text, "spans": [], "edits": [], "confidence": []}
    return _final_updates(
        buf, list(incidents), list(sources), [], tok_per_sec=0.0, context_tokens=0
    )


def test_an_empty_generation_still_discloses_incidents():
    d = _delta("", incidents=["step 2 (Write the total): Execution declined by the user."])
    content = d["messages"][-1].content
    assert content.startswith(NO_ANSWER_TEXT)
    assert INCIDENTS_NOTE_HEADER in content
    assert "Execution declined by the user." in content


def test_an_empty_generation_still_carries_the_sources_footer():
    d = _delta("", sources=[("read_file(file_path='a.txt') -> hi", "a.txt")])
    content = d["messages"][-1].content
    assert content.startswith(NO_ANSWER_TEXT)
    assert "Sources:" in content


def test_a_normal_answer_is_unaffected_by_the_empty_case_fallback():
    d = _delta("The total is 573.", incidents=["step 3 (Save it): skipped"])
    content = d["messages"][-1].content
    assert content.startswith("The total is 573.")
    assert NO_ANSWER_TEXT not in content
    assert INCIDENTS_NOTE_HEADER in content


def test_no_trailers_when_nothing_to_disclose():
    d = _delta("Plain answer.")
    assert d["messages"][-1].content == "Plain answer."
    # The provenance buffer's text is NOT rewritten by the trailers (spans keep indexing prose).
    assert d["answer_buffer"]["text"] == "Plain answer."
    assert d["answer_buffer"]["state"] == "complete"


def test_empty_generation_keeps_the_buffer_empty():
    # NO_ANSWER_TEXT is a fact stated on the MESSAGE; the provenance buffer records what the
    # model actually produced (nothing), so the human-span math never sees invented prose.
    d = _delta("", incidents=["step 1 (x): error"])
    assert d["answer_buffer"]["text"] == ""
