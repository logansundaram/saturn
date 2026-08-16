"""The answer groundedness gate + the computed-value check (transplanted from the engine isolate,
M2 + M8): synthesis is an LLM step whose output nothing verified — a figure the model computed in
prose reached the user indistinguishable from one a tool returned. Now: every figure the answer
states (>= 3 integer digits, or any non-integer) is diffed against the OBSERVATION POOL (the
human's words + this turn's tool observations — never a reasoning step's text, which would let an
invented figure launder itself); an untraceable figure gets ONE corrective regeneration carrying
the specific complaint; survivors are DISCLOSED in a trailer, never suppressed. The inverse check:
the last computed figure of the turn must appear in the answer (off when the turn has incidents —
a disclosure outranks a figure). Both correct on a normal first pass only; on a RESUME (the human
edited the prefix) they DETECT and mark without regenerating — the human's text outranks the
engine's self-correction. Measured: 5 firings / 120 runs, 0 false positives; value_dropped 2 → 0.
"""

import types

from langchain.messages import AIMessage, HumanMessage

import textutil
from core import provenance
from nodes import synthesize as syn


def step(step_id=1, label="", tool=None, result=None, status="pending"):
    return {"step_id": step_id, "label": label, "status": status, "intended_tool": tool,
            "result": result, "needs_resolution": False}


def base_state(current_query="q", plan=None, **kw):
    s = {"messages": [HumanMessage(current_query)], "plan": plan or [], "context": "",
         "current_query": current_query, "iteration": 0, "replans": 0,
         "tool_results": [], "documents_retrieved": [], "tool_events": [], "plan_vetoes": []}
    s.update(kw)
    return s


# ── the figure helpers (textutil, leaf) ─────────────────────────────────────────────────────


def test_figures_are_measurements_not_discourse_counts():
    lits = [lit for _v, lit in textutil.figure_literals("I read 2 files in 9 hours: 515 and 0.86 and 1,234")]
    assert lits == ["515", "0.86", "1,234"]


def test_citation_markers_are_engine_syntax_not_claims():
    assert textutil.figure_literals("The total is 515 [123].") == [(515.0, "515")]


def test_untraceable_figures_flags_prose_arithmetic():
    assert textutil.untraceable_figures("alpha 515, beta 551", "amount: 120\namount: 340") == ["515", "551"]


def test_a_gathered_value_is_traceable():
    assert textutil.untraceable_figures("The total is 460.", "calculate(expression='120+340') -> 460") == []


def test_a_rounding_of_an_observation_is_traceable():
    assert textutil.untraceable_figures("about 0.86", "ratio 0.857142") == []
    assert textutil.untraceable_figures("about 515", "515.4") == []
    assert textutil.untraceable_figures("about 0.9", "0.857142") == []   # one decimal: 0.9 ok
    assert textutil.untraceable_figures("exactly 0.858", "0.857142") == ["0.858"]


def test_comma_grouping_does_not_create_a_false_positive():
    assert textutil.untraceable_figures("1,234 units", "count 1234") == []


# ── the pool and the applies rule ───────────────────────────────────────────────────────────


def test_the_gate_is_off_when_the_turn_observed_nothing():
    st = base_state(plan=[step(1, "Think", None, "SHA-256 has 256 bits", "done")])
    assert not syn.gate_applies(st)
    assert syn.ungrounded_figures({"text": "256 bits, 512 rounds"}, st, "q") == ()


def test_the_gate_is_on_as_soon_as_one_tool_step_produced_a_result():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120", "done")])
    assert syn.gate_applies(st)


def test_a_reasoning_steps_result_is_not_an_observation():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120", "done"),
                          step(2, "Think", None, "so the total is 515", "done")])
    assert "515" not in syn.observation_pool(st, "q")
    assert syn.ungrounded_figures({"text": "the total is 515"}, st, "q") == ("515",)


def test_the_request_itself_is_a_legitimate_source():
    st = base_state(plan=[step(1, "Read", "read_file", "x", "done")])
    assert syn.ungrounded_figures({"text": "you asked about 2027"}, st, "what about 2027?") == ()


# ── the ladder ──────────────────────────────────────────────────────────────────────────────


class _Model:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, msgs, **kw):
        self.calls.append(msgs)
        r = self.replies.pop(0) if self.replies else ""
        if isinstance(r, Exception):
            raise r
        return AIMessage(content=r)


def _buf(text):
    return provenance.append_model(provenance.new_buffer(), text)


def test_an_ungrounded_figure_triggers_exactly_one_corrective_regeneration():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120\namount: 340", "done")])
    model = _Model(["The amounts are 120 and 340."])
    buf, ungrounded = syn._ground_answer(_buf("The total is 515."), model, [HumanMessage("q")], st, "q")
    assert buf["text"] == "The amounts are 120 and 340." and ungrounded == ()
    assert len(model.calls) == 1
    assert "515" in str(model.calls[0][-1].content)          # the corrective names the figure
    assert not provenance.human_spans(buf)                    # a model rewrite, not a human edit


def test_a_grounded_answer_costs_no_extra_call():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120", "done")])
    model = _Model([])
    buf, ungrounded = syn._ground_answer(_buf("The amount is 120."), model, [], st, "q")
    assert ungrounded == () and model.calls == [] and buf["text"] == "The amount is 120."


def test_a_figure_that_survives_the_correction_is_disclosed_not_suppressed():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120", "done")])
    model = _Model(["Still 515, honestly."])
    buf, ungrounded = syn._ground_answer(_buf("The total is 515."), model, [], st, "q")
    assert buf["text"] == "Still 515, honestly." and ungrounded == ("515",)


def test_a_failed_correction_keeps_the_original_answer_and_discloses():
    st = base_state(plan=[step(1, "Read", "read_file", "amount: 120", "done")])
    model = _Model([RuntimeError("daemon down")])
    buf, ungrounded = syn._ground_answer(_buf("The total is 515."), model, [], st, "q")
    assert buf["text"] == "The total is 515." and ungrounded == ("515",)


def test_required_figures_is_the_last_computed_value_and_off_on_incidents():
    plan = [step(1, "Read", "read_file", "120 340", "done"),
            step(2, "Sum", "calculate", "460", "done"),
            step(3, "Compare", "calculate", "551", "done")]
    assert syn.required_figures(base_state(plan=plan)) == ("551",)
    plan_inc = plan + [step(4, "Write", "write_file", "skipped write: x", "skipped")]
    assert syn.required_figures(base_state(plan=plan_inc)) == ()
    assert syn.unstated_computed_figures({"text": "beta is larger"}, base_state(plan=plan)) == ("551",)
    assert syn.unstated_computed_figures({"text": "beta is larger at 551"}, base_state(plan=plan)) == ()


def test_final_updates_appends_both_notes_above_the_sources_footer():
    d = syn._final_updates(_buf("answer"), [], [(1, "read_file(a)")], [], tok_per_sec=0.0,
                           context_tokens=0, ungrounded=("515",), dropped=("551",))
    content = d["messages"][-1].content
    assert content.startswith("answer")
    assert syn.COMPUTED_NOTE_HEADER in content and syn.GROUNDING_NOTE_HEADER in content
    assert content.index(syn.GROUNDING_NOTE_HEADER) < content.index("Sources:")
    assert d["answer_buffer"]["text"] == "answer"   # trailers never enter the buffer


# ── the node ────────────────────────────────────────────────────────────────────────────────


def _wire(monkeypatch, first_pass_text, model):
    monkeypatch.setattr(syn, "get_model", lambda role: model)
    monkeypatch.setattr(syn, "model_id", lambda role: "test-model")
    monkeypatch.setattr(syn.continuation, "supports", lambda m: False)
    monkeypatch.setattr(syn, "_stream_first_pass",
                        lambda llm_input, freeze: (_buf(first_pass_text), False, {}, None))


def test_the_node_corrects_a_first_pass_and_records_the_corrected_answer(monkeypatch):
    model = _Model(["The amounts are 120 and 340."])
    _wire(monkeypatch, "The total is 515.", model)
    st = base_state("Read a.csv and tell me the amounts",
                    plan=[step(1, "Read a.csv", "read_file", "amount: 120\namount: 340", "done")],
                    tool_results=["read_file(file_path='a.csv') -> amount: 120\namount: 340"])
    out = syn.synthesize_node(st)
    assert out["messages"][-1].content.startswith("The amounts are 120 and 340.")
    assert out["answer_buffer"]["text"] == "The amounts are 120 and 340."
    assert syn.GROUNDING_NOTE_HEADER not in out["messages"][-1].content


def test_a_pure_reasoning_turn_is_left_alone(monkeypatch):
    model = _Model([])
    _wire(monkeypatch, "SHA-256 produces a 256-bit digest.", model)
    st = base_state("what is sha256?", plan=[step(1, "Think", None, "256 bits", "done")])
    out = syn.synthesize_node(st)
    assert model.calls == []
    assert out["messages"][-1].content.startswith("SHA-256 produces a 256-bit digest.")


def test_the_gate_does_not_fight_a_human_edit_on_resume(monkeypatch):
    """On a resume the human edited the prefix: DETECT and mark, never regenerate."""
    model = _Model(["would be a regeneration"])
    monkeypatch.setattr(syn, "get_model", lambda role: model)
    monkeypatch.setattr(syn, "model_id", lambda role: "test-model")
    monkeypatch.setattr(syn.continuation, "supports", lambda m: True)
    edited = provenance.apply_edit(_buf("The total is 515."), "The total is 999.")
    monkeypatch.setattr(syn, "_stream_continuation",
                        lambda model_name, llm_input, buf, freeze: (dict(buf), False, {}))
    st = base_state("Read a.csv", plan=[step(1, "Read a.csv", "read_file", "amount: 120", "done")],
                    answer_buffer={**edited, "state": "resume"})
    out = syn.synthesize_node(st)
    assert model.calls == []                                     # no regeneration
    assert out["answer_buffer"]["text"] == "The total is 999."   # the human's text stands
    assert "999" in out["messages"][-1].content
    assert syn.GROUNDING_NOTE_HEADER in out["messages"][-1].content   # …but it is marked
