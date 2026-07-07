"""
Token-confidence grading (2026-07-06) — the offline test net.

Covers the pure surfaces: chunk alignment (core/confidence.align_chunk — token strings onto
character offsets, the mean-entry fallback, both entry shapes), run grading (low_runs — the
threshold, the min-run floor, neutral-token bridging, ledger-gap breaks, whitespace trimming),
the provenance buffer's confidence overlay (append shift, apply_edit's keep/drop/shift, gotcha-#4
plain dicts + immutability), the ResponseStream live ledger, and the freeze-editor/interrupt
plumbing that carries the overlay. Nothing here calls an LLM or the network; the LIVE half
(a daemon actually answering logprobs on both paths) belongs to utilities/continuation_contract.
"""

from types import SimpleNamespace

from core import confidence, provenance


def _lp(token: str, logprob: float) -> dict:
    return {"token": token, "logprob": logprob}


# --- align_chunk ----------------------------------------------------------------------------------


def test_align_perfect_tokenization_yields_per_token_entries():
    text = "the cat"
    entries = confidence.align_chunk(text, [_lp("the", -0.1), _lp(" cat", -2.0)])
    assert entries == [
        {"start": 0, "end": 3, "logprob": -0.1},
        {"start": 3, "end": 7, "logprob": -2.0},
    ]


def test_align_honors_offset():
    entries = confidence.align_chunk("ab", [_lp("a", -1.0), _lp("b", -2.0)], offset=10)
    assert [(e["start"], e["end"]) for e in entries] == [(10, 11), (11, 12)]


def test_align_mismatch_falls_back_to_one_mean_entry():
    # Token strings that don't reassemble the chunk: one whole-chunk entry, mean logprob —
    # coarse, but never mis-attributed character offsets.
    entries = confidence.align_chunk("hello", [_lp("hel", -1.0), _lp("LO", -3.0)])
    assert entries == [{"start": 0, "end": 5, "logprob": -2.0}]


def test_align_tolerates_object_shaped_entries():
    # The chat path forwards the ollama client's attribute-shaped objects untouched.
    objs = [SimpleNamespace(token="hi", logprob=-0.5)]
    assert confidence.align_chunk("hi", objs) == [{"start": 0, "end": 2, "logprob": -0.5}]


def test_align_empty_or_garbage_is_no_entries():
    assert confidence.align_chunk("text", None) == []
    assert confidence.align_chunk("", [_lp("x", -1.0)]) == []
    assert confidence.align_chunk("text", [{"nope": 1}]) == []


# --- low_runs -------------------------------------------------------------------------------------

# exp(-3) ≈ 0.05 — comfortably under any sane threshold; exp(-0.01) ≈ 0.99 — comfortably over.
LOW, HIGH = -3.0, -0.01


def _entries(text: str, tokens: list[tuple[str, float]]) -> list[dict]:
    return confidence.align_chunk(text, [_lp(t, lp) for t, lp in tokens])


def test_three_low_tokens_make_a_run():
    text = "aa bb cc"
    ents = _entries(text, [("aa", LOW), (" bb", LOW), (" cc", LOW)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == [(0, 8)]


def test_fewer_than_min_run_is_never_marked():
    text = "aa bb cc"
    ents = _entries(text, [("aa", LOW), (" bb", LOW), (" cc", HIGH)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == []


def test_high_token_breaks_the_run():
    text = "aa bb XX cc dd ee"
    ents = _entries(text, [("aa", LOW), (" bb", LOW), (" XX", HIGH),
                           (" cc", LOW), (" dd", LOW), (" ee", LOW)])
    # Only the second group reaches the floor; edges trimmed to non-whitespace.
    assert confidence.low_runs(ents, text, threshold_p=0.35) == [(9, 17)]


def test_neutral_tokens_bridge_but_do_not_count():
    # Punctuation/whitespace tokens ride along a run (their probability says nothing about
    # content) without counting toward the floor or breaking the streak.
    text = "aa, bb, cc"
    ents = _entries(text, [("aa", LOW), (",", HIGH), (" bb", LOW), (",", HIGH), (" cc", LOW)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == [(0, 10)]
    # …but neutrals alone never form a run.
    ents2 = _entries("...", [(".", LOW), (".", LOW), (".", LOW)])
    assert confidence.low_runs(ents2, "...", threshold_p=0.35) == []


def test_ledger_gap_breaks_a_run():
    # Two low pairs separated by UNMEASURED text (a chunk that carried no logprobs) must not
    # fuse into one marked run across the gap.
    text = "aa bb ???? cc dd"
    left = _entries("aa bb", [("aa", LOW), (" bb", LOW)])
    right = confidence.align_chunk(" cc dd", [_lp(" cc", LOW), _lp(" dd", LOW)], offset=10)
    assert confidence.low_runs(left + right, text, threshold_p=0.35) == []


def test_threshold_parameter_is_respected():
    text = "aa bb cc"
    p_half = -0.7  # exp ≈ 0.5
    ents = _entries(text, [("aa", p_half), (" bb", p_half), (" cc", p_half)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == []
    assert confidence.low_runs(ents, text, threshold_p=0.6) == [(0, 8)]


def test_buffer_runs_tolerates_garbage():
    assert confidence.buffer_runs(None) == []
    assert confidence.buffer_runs({"text": 5, "confidence": "junk"}) == []


# --- the provenance buffer's confidence overlay ----------------------------------------------------


def test_new_buffer_carries_the_overlay_key():
    assert provenance.new_buffer()["confidence"] == []


def test_append_model_shifts_chunk_relative_entries():
    b = provenance.append_model(provenance.new_buffer(), "abc",
                                confidence.align_chunk("abc", [_lp("abc", -1.0)]))
    b = provenance.append_model(b, "def",
                                confidence.align_chunk("def", [_lp("def", -2.0)]))
    assert b["confidence"] == [
        {"start": 0, "end": 3, "logprob": -1.0},
        {"start": 3, "end": 6, "logprob": -2.0},
    ]
    # Entries are plain dicts with plain floats — gotcha #4: the buffer rides the checkpointer.
    assert all(type(e) is dict for e in b["confidence"])


def test_apply_edit_keeps_prefix_drops_edited_shifts_suffix():
    # "one two three" tokenized one-word-per-entry; the human replaces "two" with "TWENTY".
    b = provenance.new_buffer()
    b = provenance.append_model(
        b, "one two three",
        confidence.align_chunk("one two three",
                               [_lp("one", -0.1), _lp(" two", -3.0), _lp(" three", -0.2)]),
    )
    e = provenance.apply_edit(b, "one TWENTY three")
    starts = {(c["start"], c["end"]): c["logprob"] for c in e["confidence"]}
    assert (0, 3) in starts                       # the untouched prefix survives as-is
    assert not any(s < 10 for s, _ in [k for k in starts if k != (0, 3)])
    # The suffix entry (" three": was 7..13) shifts by the +3 length delta -> 10..16.
    assert (10, 16) in starts and starts[(10, 16)] == -0.2
    # The edited region's entry (" two") is gone — human text has no model confidence.
    assert len(e["confidence"]) == 2


def test_apply_edit_never_mutates_the_input_overlay():
    b = provenance.append_model(provenance.new_buffer(), "abcdef",
                                confidence.align_chunk("abcdef", [_lp("abcdef", -1.0)]))
    snapshot = [dict(c) for c in b["confidence"]]
    provenance.apply_edit(b, "abcXYZ")
    assert b["confidence"] == snapshot


def test_old_buffers_without_overlay_still_edit_cleanly():
    # A pre-feature buffer (a replayed old record) lacks the key entirely.
    old = {"text": "hello world", "spans": [{"start": 0, "end": 11, "author": "model"}],
           "edits": []}
    e = provenance.apply_edit(old, "hello there")
    assert e["text"] == "hello there" and e["confidence"] == []


# --- the live UI ledger ----------------------------------------------------------------------------


def test_response_stream_ledger_accumulates_and_reset_reseeds(capsys):
    from tui.ui.response import ResponseStream

    rs = ResponseStream()
    rs.feed("abc", [_lp("abc", -1.0)])
    rs.feed("def")                       # no logprobs: a ledger gap, honest absence
    rs.feed("ghi", [_lp("ghi", -2.0)])
    assert rs._conf == [
        {"start": 0, "end": 3, "logprob": -1.0},
        {"start": 6, "end": 9, "logprob": -2.0},
    ]
    rs.reset_to("abcX", [{"start": 0, "end": 3, "logprob": -1.0}])
    assert rs._len == 4 and rs._conf == [{"start": 0, "end": 3, "logprob": -1.0}]
    rs.feed("jk", [_lp("jk", -0.5)])
    assert rs._conf[-1] == {"start": 4, "end": 6, "logprob": -0.5}
    capsys.readouterr()  # swallow the plain-path/section output


def test_final_body_marks_uncertain_runs_without_a_correction():
    # The user-reported bug: an uncorrected answer with low-confidence runs must render red on
    # the PERMANENT body (not only the transient live tail / the receipt count). Drive the real
    # final-render path against a color-capturing console and assert the red ANSI is emitted.
    import importlib
    import io

    from rich.console import Console

    import tui.ui._base as base

    cap = io.StringIO()
    saved = base._console
    R = importlib.import_module("tui.ui.response")
    try:
        base._console = Console(file=cap, force_terminal=True, color_system="standard", width=100)
        R._console = base._console

        text = "The capital is Canberra and the population is roughly four million people."
        buf = provenance.append_model(provenance.new_buffer(), text)
        i, j = text.index("roughly four million"), text.index("roughly four million") + 20
        buf["confidence"] = (
            [{"start": 0, "end": i, "logprob": HIGH}]
            + [{"start": p, "end": p + 1, "logprob": LOW} for p in range(i, j)]
            + [{"start": j, "end": len(text), "logprob": HIGH}]
        )
        buf["state"] = "complete"

        R.set_turn_buffer({"answer_buffer": buf})
        R._final_render(text, plain_body=text)
    finally:
        base._console = saved
        R._console = saved
    out = cap.getvalue()
    assert "\x1b[31m" in out          # red foreground for the low-confidence run
    assert "uncertain" in out          # …and the receipt still counts it


def test_uncertain_answer_keeps_markdown_and_shows_red():
    # The core of this feature's UX fix: an uncorrected uncertain answer must keep its markdown
    # (bold/heading) AND redden the low-confidence run — the reddening rides the rendered SEGMENT
    # stream by content, so markdown reflow survives. Drive the real _render_answer_body path.
    import importlib
    import io

    from rich.console import Console

    import tui.ui._base as base

    cap = io.StringIO()
    saved = base._console
    R = importlib.import_module("tui.ui.response")
    try:
        base._console = Console(file=cap, force_terminal=True, color_system="standard", width=80)
        R._console = base._console

        body = ("# Report\n\nThe capital is **Canberra** and the population is roughly four "
                "million people, a number that may be stale.")
        buf = provenance.append_model(provenance.new_buffer(), body)
        i = body.index("roughly four million")
        j = i + len("roughly four million")
        buf["confidence"] = (
            [{"start": 0, "end": i, "logprob": HIGH}]
            + [{"start": p, "end": p + 1, "logprob": LOW} for p in range(i, j)]
            + [{"start": j, "end": len(body), "logprob": HIGH}]
        )
        buf["state"] = "complete"
        R._render_answer_body(body, buf)
    finally:
        base._console = saved
        R._console = saved
    out = cap.getvalue()
    assert "\x1b[31m" in out   # the low-confidence run is red …
    assert "\x1b[1m" in out    # … and the markdown bold ("Canberra") survived


def test_redden_segments_preserves_style_and_marks_by_content():
    # The mechanism: reddening a Segment stream by phrase, combining red into the existing style
    # (a bold phrase stays bold-red), and leaving non-matching segments untouched.
    import importlib

    from rich.segment import Segment
    from rich.style import Style

    R = importlib.import_module("tui.ui.response")
    segs = [Segment("plain ", Style()),
            Segment("roughly four million", Style(bold=True)),
            Segment(" tail", Style())]
    out = list(R._redden_segments(segs, ["roughly four million"], "red"))
    red = [s for s in out if s.style and s.style.color and s.style.color.name == "red"]
    assert red and red[0].text == "roughly four million"
    assert red[0].style.bold  # existing bold combined with the red
    # A phrase absent from any single segment (here split by a boundary) is simply not marked.
    split = [Segment("roughly ", Style()), Segment("four", Style(bold=True)),
             Segment(" million", Style())]
    out2 = list(R._redden_segments(split, ["roughly four million"], "red"))
    assert not any(s.style and s.style.color and s.style.color.name == "red" for s in out2)


def test_frozen_tail_render_survives_confidence(capsys):
    # The freeze editor's tail print with an overlay present — smoke both paths' tolerance.
    from tui.ui.correction import _print_frozen

    text = "aa bb cc"
    conf = confidence.align_chunk(text, [_lp("aa", LOW), _lp(" bb", LOW), _lp(" cc", LOW)])
    _print_frozen(text, [], conf)
    out = capsys.readouterr().out
    assert "aa bb cc" in out


def test_answer_gate_payload_carries_the_overlay(monkeypatch):
    # The interrupt payload hands the freeze editor the overlay, and the edited buffer written
    # back keeps it (shifted by THE one edit-diff, provenance.apply_edit).
    from nodes import answer_gate

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "resume", "text": payload["text"]}

    monkeypatch.setattr(answer_gate, "interrupt", fake_interrupt)
    buf = provenance.append_model(provenance.new_buffer(), "abc",
                                  confidence.align_chunk("abc", [_lp("abc", -1.0)]))
    out = answer_gate.answer_gate_node(
        {"answer_buffer": {**buf, "state": "frozen"}, "current_query": "q"}
    )
    assert captured["confidence"] == buf["confidence"]
    assert out["answer_buffer"]["confidence"] == buf["confidence"]
