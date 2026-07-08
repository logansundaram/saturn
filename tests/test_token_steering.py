"""
Interrupt-and-correct (token steering) — the offline half of the feature's test net.

Covers the pure/structural surfaces: the raw-mode template registry's byte-fidelity
(core/chat_template — a single wrong special token breaks continuation, so the rendered strings
are pinned as goldens against the sources documented in that module), the provenance buffer's
copy-on-write span math (core/provenance), the freeze latch, the continuation request assembly
(no network — the stream is built lazily), the answer_gate node's resume-value contract, the
synthesize routing, and the rail/replay audit echoes.

The LIVE half — proof that a model actually continues a spliced human prefix seamlessly — is
`utilities/continuation_contract.py` (needs the Ollama daemon), which DEFINES the supported-model
set. Nothing here calls an LLM or the network.
"""

import pytest
from langchain.messages import HumanMessage, SystemMessage

from core import chat_template, continuation, provenance


# --- the template registry (byte-fidelity goldens) ------------------------------------------------

def test_qwen_render_is_byte_faithful():
    msgs = [SystemMessage(content="SYS"), HumanMessage(content="QUESTION")]
    out = chat_template.render_continuation("qwen3.6:27b", msgs, "PREFIX ends mid-tok")
    assert out == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nQUESTION<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nPREFIX ends mid-tok"
    )


def test_gemma_render_is_byte_faithful():
    msgs = [SystemMessage(content="SYS"), HumanMessage(content="QUESTION")]
    out = chat_template.render_continuation("gemma4:e4b", msgs, "PREFIX")
    assert out == (
        "<|turn>system\nSYS<turn|>\n"
        "<|turn>user\nQUESTION<turn|>\n"
        "<|turn>model\nPREFIX"
    )


def test_assistant_turn_is_open_no_end_of_turn_token():
    """The whole feature: the assistant turn is opened but never closed."""
    for model in ("qwen3.5:9b", "gemma4:e4b"):
        t = chat_template.template_for(model)
        out = chat_template.render_continuation(model, [("user", "hi")], "half an ans")
        assert out.endswith("half an ans")
        for stop in t.stop:
            assert not out.endswith(stop)


def test_consecutive_same_role_messages_merge_into_one_turn():
    msgs = [("system", "S"), ("user", "part one"), ("user", "part two"), ("user", "")]
    turns = chat_template.normalize_turns(msgs)
    assert turns == [("system", "S"), ("user", "part one\n\npart two")]


def test_unsupported_model_refuses_and_supported_reports():
    with pytest.raises(chat_template.UnsupportedModel):
        chat_template.template_for("mystery-llm:7b")
    assert not chat_template.supported("mystery-llm:7b")
    assert chat_template.supported("qwen3.6:35b")   # prefix match, any tag
    assert chat_template.supported("gemma4:26b")
    assert not continuation.supports("gpt-oss:20b")  # installed but deliberately outside the set


# --- the provenance buffer (immutable, span-tagged) ------------------------------------------------

def test_append_model_extends_and_merges_spans():
    b = provenance.append_model(provenance.append_model(provenance.new_buffer(), "abc"), "def")
    assert b["text"] == "abcdef"
    assert b["spans"] == [{"start": 0, "end": 6, "author": "model"}]


def test_apply_edit_replace_in_place_records_one_human_span():
    b = provenance.append_model(provenance.new_buffer(), "the capital is Sydney, a city")
    e = provenance.apply_edit(b, "the capital is Canberra, a city")
    assert e["text"] == "the capital is Canberra, a city"
    authors = [(s["author"], e["text"][s["start"]:s["end"]]) for s in e["spans"]]
    assert ("human", "Canberra") in [(a, t.strip(", ")) for a, t in authors] or \
           any(a == "human" and "anberr" in t for a, t in authors)
    assert len(e["edits"]) == 1 and e["edits"][0]["cut"]  # the cut text is on the audit record


def test_apply_edit_truncate_and_append():
    b = provenance.append_model(provenance.new_buffer(), "one, two, three, WRONG")
    e = provenance.apply_edit(b, "one, two, three, ninety-nine,")
    assert e["text"].endswith("ninety-nine,")
    assert provenance.human_spans(e)  # the typed tail is human-authored
    assert provenance.corrected(e)


def test_apply_edit_noop_returns_copy_without_edit_record():
    b = provenance.append_model(provenance.new_buffer(), "unchanged")
    e = provenance.apply_edit(b, "unchanged")
    assert e == b and e is not b
    assert not provenance.corrected(e)


def test_operations_never_mutate_their_input():
    b0 = provenance.append_model(provenance.new_buffer(), "first draft here")
    snapshot = {"text": b0["text"], "spans": [dict(s) for s in b0["spans"]],
                "edits": list(b0["edits"]), "confidence": list(b0["confidence"])}
    provenance.apply_edit(b0, "first CORRECTION here")
    provenance.append_model(b0, " more")
    assert b0 == snapshot  # copy-and-return, never in-place


def test_spans_always_cover_the_text_exactly():
    """The invariant every renderer relies on: spans tile [0, len(text)) in order, gap-free."""
    b = provenance.new_buffer()
    b = provenance.append_model(b, "alpha beta gamma")
    b = provenance.apply_edit(b, "alpha CORRECTED gamma")
    b = provenance.append_model(b, " delta")
    b = provenance.apply_edit(b, "alpha CORRECTED gamma TYPED")
    pos = 0
    for s in b["spans"]:
        assert s["start"] == pos
        pos = s["end"]
    assert pos == len(b["text"])


def test_state_key_survives_provenance_operations():
    b = {**provenance.new_buffer(), "state": "resume"}
    assert provenance.append_model(b, "x")["state"] == "resume"
    assert provenance.apply_edit(b, "y")["state"] == "resume"


# --- the freeze latch -------------------------------------------------------------------------------

def test_freeze_latch_only_fires_while_armed():
    c = continuation.FreezeController()
    assert not c.freeze()          # disarmed: Esc falls through to pause/steer
    c.arm()
    assert c.freeze() and c.requested()
    c.clear()
    assert not c.requested()
    c.freeze()
    c.disarm()                     # disarm clears a stale request too
    assert not c.requested() and not c.freeze()


def test_typeahead_esc_prefers_freeze_then_falls_back_to_pause():
    from core.plan_ops import PauseController
    from tui.typeahead import InputQueue

    froze, paused = [], []
    pc = PauseController()
    q = InputQueue(on_pause=lambda: paused.append(1), on_freeze=lambda: froze.append(1),
                   controller=pc)
    fc = continuation.get_freeze_controller()
    fc.arm()
    try:
        q._on_escape()
        assert froze and not pc.pending()          # consumed as a freeze
    finally:
        fc.disarm()
    q._on_escape()
    assert paused and pc.pending()                 # disarmed: the old pause meaning
    pc.clear()


# --- the continuation request (assembled, never sent) ----------------------------------------------

def test_continue_from_assembles_a_raw_request_without_touching_the_network():
    stream = continuation.continue_from("qwen3.6:27b", [("user", "hi")], "half an answer")
    body = stream._body
    assert body["raw"] is True and body["stream"] is True
    assert body["prompt"].endswith("half an answer")
    assert body["stop"] == ["<|im_end|>"]
    assert body["options"]["num_ctx"] > 0  # §4: explicit, or the daemon silently front-truncates
    stream.close()  # idempotent, never raises
    stream.close()


def test_continue_from_refuses_unsupported_models():
    with pytest.raises(chat_template.UnsupportedModel):
        continuation.continue_from("mystery-llm:7b", [("user", "hi")], "prefix")


# --- the answer_gate node (resume-value contract) ---------------------------------------------------

def _frozen_state(text="draft answer so far"):
    buf = {**provenance.append_model(provenance.new_buffer(), text), "state": "frozen"}
    return {"answer_buffer": buf, "current_query": "q"}


def test_answer_gate_applies_the_edit_and_resumes(monkeypatch):
    import nodes.answer_gate as gate

    monkeypatch.setattr(gate, "interrupt",
                        lambda payload: {"action": "resume", "text": "draft answer CORRECTED"})
    out = gate.answer_gate_node(_frozen_state())
    buf = out["answer_buffer"]
    assert buf["state"] == "resume" and buf["edited"]
    assert buf["text"] == "draft answer CORRECTED"
    assert provenance.human_spans(buf)


def test_answer_gate_tolerates_a_bare_true_resume(monkeypatch):
    """The headless approver answers unknown interrupts with True = continue unchanged."""
    import nodes.answer_gate as gate

    monkeypatch.setattr(gate, "interrupt", lambda payload: True)
    out = gate.answer_gate_node(_frozen_state())
    buf = out["answer_buffer"]
    assert buf["state"] == "resume" and not buf["edited"] and not buf["edits"]


def test_answer_gate_done_accepts_the_text_as_final(monkeypatch):
    import nodes.answer_gate as gate

    monkeypatch.setattr(gate, "interrupt", lambda payload: {"action": "done", "text": "keep this"})
    out = gate.answer_gate_node(_frozen_state())
    assert out["answer_buffer"]["state"] == "done"


# --- synthesize routing + the no-generation finalize path -------------------------------------------

def test_route_after_synthesize():
    from nodes.synthesize import route_after_synthesize

    assert route_after_synthesize({"answer_buffer": {"state": "frozen"}}) == "answer_gate"
    assert route_after_synthesize({"answer_buffer": {"state": "complete"}}) == "end"
    assert route_after_synthesize({"answer_buffer": None}) == "end"
    assert route_after_synthesize({}) == "end"


def test_synthesize_done_buffer_finalizes_without_an_llm(isolated_paths):
    """A 'done' buffer (the user accepted the frozen text) must produce the final AIMessage
    mechanically — no model call, so this runs offline."""
    from nodes.synthesize import synthesize_node

    buf = {**provenance.append_model(provenance.new_buffer(), "the corrected answer"),
           "state": "done"}
    state = {"current_query": "q", "context": "", "plan": [], "tool_results": [],
             "documents_retrieved": [], "messages": [HumanMessage(content="q")],
             "answer_buffer": buf, "tok_per_sec": 0.0, "context_tokens": 0}
    out = synthesize_node(state)
    assert out["messages"][-1].content.startswith("the corrected answer")
    assert out["answer_buffer"]["state"] == "complete"


# --- the audit echoes (live rail + /trace replay share these) ---------------------------------------

def test_rail_echoes_the_freeze_and_the_correction(capsys):
    import importlib

    trace = importlib.import_module("tui.ui.trace")

    trace._render_trust_annotations(
        "synthesize", {"answer_buffer": {"state": "frozen", "text": "x", "spans": []}})
    assert "froze the answer" in capsys.readouterr().out

    trace._render_trust_annotations(
        "answer_gate",
        {"answer_buffer": {"state": "resume", "edited": True,
                           "edits": [{"at": 5, "cut": "Sydney", "typed": "Canberra"}]}})
    out = capsys.readouterr().out
    assert "you corrected the answer" in out and "Canberra" in out

    trace._render_trust_annotations(
        "answer_gate", {"answer_buffer": {"state": "resume", "edited": False, "edits": []}})
    assert "resumed unchanged" in capsys.readouterr().out


def test_receipt_counts_corrections(capsys, monkeypatch):
    import importlib

    response = importlib.import_module("tui.ui.response")
    monkeypatch.setattr(response, "_trust_spans", lambda: [])
    response._print_receipt(corrections=2)
    assert "2 corrections" in capsys.readouterr().out


def test_corrected_body_marks_human_spans_and_never_guesses(capsys):
    import importlib

    response = importlib.import_module("tui.ui.response")
    buf = provenance.apply_edit(
        provenance.append_model(provenance.new_buffer(), "it is Sydney today"),
        "it is Canberra today")
    body = buf["text"] + "\n\nNote — trailing mechanical text"
    assert response._print_marked_body(body, buf)
    assert "Canberra" in capsys.readouterr().out
    # A body the buffer doesn't prefix (a different answer replaced it) must refuse to mark.
    assert not response._print_marked_body("a different answer entirely", buf)


# --- the freeze editor (tui/ui/correction) ---------------------------------------------------------

def test_edit_inline_resolves_the_prompt_module_not_the_function(monkeypatch):
    """Regression: the package __init__ re-exports the prompt() FUNCTION under the module's name,
    so `from . import prompt` hands back the function — the editor crashed the whole turn with
    `'function' object has no attribute '_PTK'`. The editor must reach the real module: with the
    module's _PTK patched False it declines cleanly (None -> wizard fallback) instead of raising."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    prompt_mod = importlib.import_module("tui.ui.prompt")
    monkeypatch.setattr(prompt_mod, "_PTK", False)
    assert correction._edit_inline("some frozen text") is None


def test_edit_answer_returns_the_resume_contract(monkeypatch, capsys):
    """edit_answer's return is the answer_gate resume value: the edited text rides `text`,
    d(one) accepts, anything else resumes."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    monkeypatch.setattr(correction, "_edit_inline", lambda text: "the corrected text")
    answers = iter(["d"])
    monkeypatch.setattr(correction, "ask", lambda _q: next(answers))
    out = correction.edit_answer({"text": "the streamed text", "spans": []})
    assert out == {"action": "done", "text": "the corrected text"}

    monkeypatch.setattr(correction, "_edit_inline", lambda text: None)  # no prompt_toolkit
    answers = iter(["", "", ""])  # wizard: keep everything, no correction; then Enter = resume
    monkeypatch.setattr(correction, "ask", lambda _q: next(answers))
    out = correction.edit_answer({"text": "the streamed text", "spans": []})
    assert out == {"action": "resume", "text": "the streamed text"}
