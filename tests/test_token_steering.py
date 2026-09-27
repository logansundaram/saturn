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


def test_assistant_turn_is_open_no_end_of_turn_token():
    """The whole feature: the assistant turn is opened but never closed."""
    for model in ("qwen3.5:9b",):
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
    """gemma4 is no longer in the registry — the family lock (2026-08-16) retired it, so a
    gemma4 tag is unsupported the same way any other outsider is."""
    with pytest.raises(chat_template.UnsupportedModel):
        chat_template.template_for("mystery-llm:7b")
    assert not chat_template.supported("mystery-llm:7b")
    assert chat_template.supported("qwen3.6:35b")   # prefix match, any tag
    assert not chat_template.supported("gemma4:26b")
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
    from core.pause import PauseController
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


def test_continue_from_asks_for_logprobs_with_drafting_off_only_when_grading_is_on(monkeypatch):
    """With confidence on, the raw request carries `logprobs` AND `draft_num_predict: 0` (a
    speculatively-decoded model reports one logprob per draft batch otherwise); with it off,
    neither rides — nobody asks, the overlay stays empty at the source."""
    from core import confidence

    monkeypatch.setattr(confidence, "enabled", lambda: True)
    on = continuation.continue_from("qwen3.6:27b", [("user", "hi")], "half")._body
    monkeypatch.setattr(confidence, "enabled", lambda: False)
    off = continuation.continue_from("qwen3.6:27b", [("user", "hi")], "half")._body

    assert on["logprobs"] is True and on["options"]["draft_num_predict"] == 0
    assert on["options"]["num_ctx"] > 0
    assert "logprobs" not in off and "draft_num_predict" not in off["options"]


def test_continue_from_refuses_unsupported_models():
    with pytest.raises(chat_template.UnsupportedModel):
        continuation.continue_from("mystery-llm:7b", [("user", "hi")], "prefix")


# --- the answer_gate node (resume-value contract) ---------------------------------------------------

def _frozen_state(text="draft answer so far"):
    buf = {**provenance.append_model(provenance.new_buffer(), text), "state": "frozen"}
    return {"answer_buffer": buf, "current_query": "q"}


def test_provenance_rstrip_trailing_keeps_spans_tiling():
    buf = provenance.append_model(provenance.new_buffer(), "abc ")
    buf = provenance.apply_edit(buf, "abc  xyz \t")
    out = provenance.rstrip_trailing(buf)
    assert out["text"] == "abc  xyz"
    assert out["spans"][-1]["end"] == len(out["text"])
    # an all-whitespace trailing span disappears rather than surviving as an empty span
    buf2 = provenance.apply_edit(provenance.append_model(provenance.new_buffer(), "abc"), "abc   ")
    out2 = provenance.rstrip_trailing(buf2)
    assert out2["text"] == "abc" and all(sp["end"] > sp["start"] for sp in out2["spans"])
    # immutability: the input buffer is untouched
    assert buf["text"] == "abc  xyz \t"


# --- synthesize routing + the no-generation finalize path -------------------------------------------

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
    """edit_answer's return is the answer_gate resume value: the edited text rides `text` and the
    action is what the editor's exit key decided — Esc/Enter resume, Ctrl-D accepts the text as
    final. There is no confirm question after the editor. The wizard floor asks only its two
    cut/correction questions and always resumes."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    monkeypatch.setattr(correction, "_live_start", lambda: None)
    monkeypatch.setattr(correction, "_edit_inline", lambda text: ("the corrected text", "resume"))
    asked: list = []
    monkeypatch.setattr(correction, "ask", lambda q: asked.append(q) or "")
    out = correction.edit_answer({"text": "the streamed text", "spans": []})
    assert out == {"action": "resume", "text": "the corrected text"}
    assert asked == []  # no "resume or done?" confirm after the editor

    # Review 2026-09-06: with the confirm gone, accept-as-final had no key at all — the only
    # way to keep exactly the frozen text was Ctrl-C at the stream, which aborts the turn.
    monkeypatch.setattr(correction, "_edit_inline", lambda text: ("keep this", "done"))
    out = correction.edit_answer({"text": "the streamed text", "spans": []})
    assert out == {"action": "done", "text": "keep this"}

    monkeypatch.setattr(correction, "_edit_inline", lambda text: None)  # no prompt_toolkit
    asked = []
    monkeypatch.setattr(correction, "ask", lambda q: asked.append(q) or "")
    out = correction.edit_answer({"text": "the streamed text", "spans": []})
    assert out == {"action": "resume", "text": "the streamed text"}
    assert len(asked) == 2  # cut-from + correction only; no confirm
    assert not any("resume" in q.lower() and "done" in q.lower() for q in asked)


def test_edit_answer_legend_names_the_three_exit_keys(monkeypatch, capsys):
    """The freeze header teaches the keys that end the edit — esc/enter resume, ctrl-d accepts —
    and nothing else (no [Y]es/[d]one prompt, no ctrl-c hotkey)."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    monkeypatch.setattr(correction, "_live_start", lambda: None)
    monkeypatch.setattr(correction, "_edit_inline", lambda text: (text, "resume"))
    monkeypatch.setattr(correction, "ask", lambda _q: "")
    correction.edit_answer({"text": "frozen", "spans": []})
    out = capsys.readouterr().out.lower()
    assert "esc" in out and "enter" in out and "ctrl-d" in out
    assert "ctrl-c" not in out and "[d]one" not in out


def test_esc_timeout_covers_a_split_alt_enter():
    """Review 2026-09-06: at 50 ms an Alt/Shift+Enter whose ESC and CR arrive in separate reads
    (ssh, mosh, a slow terminal) parsed as a bare Esc and resumed generation from a half-edited
    buffer — with no confirm step, no way back. The wait covers a realistic round-trip while
    staying well under prompt_toolkit's 0.5 s default."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    assert 0.2 <= correction._ESC_TIMEOUT_S < 0.5


def test_freeze_editor_binds_escape_to_submit():
    """Esc inside the freeze editor submits the buffer (= resume). The binding lives in the
    editor's OWN key bindings, merged over the prompt's, so the `»` prompt is untouched."""
    import importlib

    prompt_mod = importlib.import_module("tui.ui.prompt")
    if not prompt_mod._PTK:
        import pytest

        pytest.skip("prompt_toolkit not installed")
    correction = importlib.import_module("tui.ui.correction")
    from prompt_toolkit.keys import Keys

    kb = correction._freeze_key_bindings({"action": "resume"})
    esc = [b for b in kb.bindings if tuple(b.keys) == (Keys.Escape,)]
    assert esc, "no bare Esc binding in the freeze editor"
    # Ctrl-D is the accept-as-final key — bound here, never in the prompt's own set.
    assert [b for b in kb.bindings if tuple(b.keys) == (Keys.ControlD,)]
    assert not [b for b in prompt_mod._PTK_KB.bindings if tuple(b.keys) == (Keys.ControlD,)]

    # The prompt's own bindings never gained a bare Esc (that would break Alt+Enter there).
    assert not [b for b in prompt_mod._PTK_KB.bindings if tuple(b.keys) == (Keys.Escape,)]
    # The Shift/Alt+Enter newline binding still rides along in the merged set.
    assert [b for b in kb.bindings if tuple(b.keys) == (Keys.Escape, Keys.ControlM)]


def test_edit_answer_shows_the_tail_when_the_editor_fails_at_runtime(monkeypatch, capsys):
    """`show_tail` is decided up front from `_PTK`, but `_edit_inline` ALSO drops to the wizard
    from its generic except (a PromptSession/_make_ptk_input failure on an odd terminal). In that
    path the wizard asked "cut from (a fragment of the text…)" for text that was never printed —
    exactly the case the tail exists to cover. Predicted-unavailable stays a single print."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    monkeypatch.setattr(correction, "_live_start", lambda: None)
    monkeypatch.setattr(correction, "ask", lambda _q: "")

    # prompt_toolkit LOOKS available, then the editor blows up on the way in.
    monkeypatch.setattr(correction, "_inline_available", lambda: True)
    monkeypatch.setattr(correction, "_edit_inline", lambda text: None)
    correction.edit_answer({"text": "the frozen answer body", "spans": []})
    assert "the frozen answer body" in capsys.readouterr().out

    # Predicted unavailable: _print_frozen already showed it — the tail must not print twice.
    monkeypatch.setattr(correction, "_inline_available", lambda: False)
    correction.edit_answer({"text": "the frozen answer body", "spans": []})
    assert capsys.readouterr().out.count("the frozen answer body") == 1


def test_edit_answer_repins_the_status_bar_on_the_way_out(monkeypatch):
    """The graph resumes the moment the editor returns and the model re-primes its context before
    the first continued token — seconds of a completely static screen, right after the most
    interactive moment in the product. The sibling blocking editors (approval.ask_approval,
    plan.review_plan) both restart the bar on exit; this one never did (it did not even import
    _live_start)."""
    import importlib

    correction = importlib.import_module("tui.ui.correction")
    started: list = []
    monkeypatch.setattr(correction, "_live_start", lambda: started.append(True))
    monkeypatch.setattr(correction, "_edit_inline", lambda text: (text, "resume"))

    monkeypatch.setattr(correction, "ask", lambda _q: "")
    correction.edit_answer({"text": "frozen", "spans": []})
    assert started == [True]


def test_resumed_stream_stops_the_bar_before_reopening_its_live(monkeypatch):
    """Rich allows exactly one live region. The freeze editor leaves the status bar up, so the
    resumed answer tail must drop it before opening its own — and finish() must too, for the
    `done` path where no token ever resumes."""
    import importlib

    response = importlib.import_module("tui.ui.response")
    stops: list = []
    monkeypatch.setattr(response, "_live_stop", lambda: stops.append(True))
    monkeypatch.setattr(response, "_RICH", False)  # no real Live in the test

    stream = response.ResponseStream()
    stream._started = True
    stream.freeze_display()
    assert stream._froze and stream._reopen_pending

    stream.feed("resumed tokens")
    assert stops == [True]
    assert not stream._reopen_pending  # one-shot: a later token must not re-part the block


def test_a_frozen_answer_gets_its_response_rule_back(monkeypatch, capsys):
    """`_begin()`'s `── response` rule scrolled up above the freeze editor's own block, so the
    final answer landed bare, directly under the editor's output. The reopened rule names what
    happened — and never claims an edit over a resume that changed nothing."""
    import importlib

    response = importlib.import_module("tui.ui.response")
    monkeypatch.setattr(response, "_RICH", False)
    monkeypatch.setattr(response, "_live_stop", lambda: None)
    monkeypatch.setattr(response, "_trust_spans", lambda: [])

    def _finish_with(buffer):
        monkeypatch.setattr(response, "_turn_buffer", buffer)
        stream = response.ResponseStream()
        stream._started = True
        stream._chars = ["the answer"]
        stream.freeze_display()
        stream.finish("the answer")
        return capsys.readouterr().out

    out = _finish_with({"text": "the answer", "edits": [{"at": 3, "cut": "x", "typed": "y"}]})
    assert "── response" in out and "resumed after your edit" in out

    out = _finish_with({"text": "the answer", "edits": []})
    assert "── response" in out
    assert "resumed after your edit" not in out   # nothing was edited — don't claim it was
    assert "kept the text unchanged" in out

    # A turn that never froze keeps exactly one header (the one _begin printed).
    monkeypatch.setattr(response, "_turn_buffer", None)
    stream = response.ResponseStream()
    stream._started = True
    stream._chars = ["the answer"]
    stream.finish("the answer")
    assert "── response" not in capsys.readouterr().out


# --- word-boundary freeze grace (transplanted from the token_steering isolate) ------------------------
#
# Esc mid-word used to cut the prefix inside a word — a tail that retokenizes onto boundaries the
# model never produces (the daemon has no token healing), so the continuation could stumble. The
# freeze now SEEKS a boundary: after Esc, up to FREEZE_GRACE more chunks may land until the buffer
# ends outside a word (or the next chunk begins with whitespace — dropped, not appended); a second
# Esc forces the immediate cut. The seek is bounded so a freeze can never be starved.


def test_ends_mid_word():
    assert continuation.ends_mid_word("The quick bro")
    assert continuation.ends_mid_word("snake_cas")
    assert not continuation.ends_mid_word("The quick brown ")
    assert not continuation.ends_mid_word("done.")
    assert not continuation.ends_mid_word("")


def test_freeze_seeker_lands_on_a_word_boundary():
    c = continuation.FreezeController()
    c.arm()
    seek = continuation.FreezeSeeker(c)
    assert seek.before_chunk("The quick bro", " brown") is False  # nothing requested yet
    c.freeze()
    # after the chunk that completed a word lands, the buffer ends mid-word: keep pulling
    assert seek.after_chunk("The quick bro") is False
    assert seek.after_chunk("The quick brown") is False   # still alnum-terminal
    # the next chunk begins with whitespace: the word is over — freeze WITHOUT appending it
    assert seek.before_chunk("The quick brown", " fox") is True
    assert seek.dropped == " fox"


def test_freeze_seeker_stops_immediately_at_a_boundary():
    c = continuation.FreezeController()
    c.arm()
    c.freeze()
    seek = continuation.FreezeSeeker(c)
    assert seek.after_chunk("Sentence one. ") is True


def test_freeze_seeker_grace_is_bounded():
    c = continuation.FreezeController()
    c.arm()
    c.freeze()
    seek = continuation.FreezeSeeker(c)
    text = "abc"
    decisions = []
    for _ in range(continuation.FREEZE_GRACE + 2):
        text += "d"  # never reaches a boundary
        decisions.append(seek.after_chunk(text))
    assert True in decisions
    assert decisions.index(True) < continuation.FREEZE_GRACE  # cut within the grace budget


def test_second_esc_forces_the_cut():
    c = continuation.FreezeController()
    c.arm()
    assert c.freeze() and not c.forced()
    seek = continuation.FreezeSeeker(c)
    assert seek.after_chunk("mid-wor") is False
    assert c.freeze() and c.forced()          # Esc again while seeking
    assert seek.after_chunk("mid-word") is True
    c.clear()
    assert not c.forced()


def test_qwen38_resolves_to_the_qwen3x_family():
    """qwen3.8 shipped 2026-08 and passed utilities/continuation_contract.py (the one definition
    of "supported") on 2026-08-16 — same ChatML shape as qwen3.5/3.6."""
    assert chat_template.template_for("qwen3.8:27b").family == "qwen3.x"
    assert chat_template.supported("qwen3.8:27b")
