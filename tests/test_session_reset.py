"""
app/session._fresh_turn — the per-turn reset is DERIVED from _initial_state (one canonical
field list, 2026-07-10): every AgentState field resets across turns unless it is a documented
carry, so a newly added field can never silently leak into the next turn's accumulators.
"""

from langchain.messages import AIMessage, HumanMessage

from app.session import _CARRY_ACROSS_TURNS, _fresh_turn, _initial_state


def test_fresh_turn_resets_every_initial_state_field(isolated_paths):
    state = _initial_state()
    state["messages"] = [HumanMessage("first question"), AIMessage("first answer")]
    state["context_tokens"] = 777  # the documented gauge carry
    # Dirty EVERY per-turn field with a sentinel the reset must clear.
    for key, fresh in _initial_state().items():
        if key in _CARRY_ACROSS_TURNS:
            continue
        state[key] = ["DIRTY"] if isinstance(fresh, list) else "DIRTY"

    out = _fresh_turn(state, "next question")

    for key, fresh in _initial_state().items():
        if key in _CARRY_ACROSS_TURNS:
            continue
        if key == "current_query":
            assert out[key] == "next question"
        else:
            assert out[key] == fresh, f"{key} leaked across the turn boundary"
    assert out["context_tokens"] == 777, "the context gauge is a documented carry"
    assert out["messages"][-1].content == "next question"


def test_carry_list_matches_initial_state():
    """The carries must be real _initial_state keys — a renamed field would silently turn a
    deliberate carry into a full reset (or vice versa)."""
    fields = set(_initial_state())
    assert set(_CARRY_ACROSS_TURNS) <= fields
