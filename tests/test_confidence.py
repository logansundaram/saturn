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

import math
from types import SimpleNamespace

import pytest

from core import confidence, provenance


def _lp(token: str, logprob: float) -> dict:
    return {"token": token, "logprob": logprob}


def _sgr(style: str) -> str:
    """The opening SGR sequence a rich style renders as — DERIVED from the palette constant, never
    a hard-coded escape. A palette change then updates these assertions instead of breaking them,
    which is the point: the tests care that the run is marked, not what color it is."""
    from rich.style import Style

    return Style.parse(style).render("X").split("X")[0]


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
    from tui.ui._base import _LOW_CONF_STYLE

    out = cap.getvalue()
    assert _sgr(_LOW_CONF_STYLE) in out  # the low-confidence run is marked on the PERMANENT body
    assert "uncertain" in out            # …and the receipt still counts it


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
    from tui.ui._base import _LOW_CONF_STYLE

    out = cap.getvalue()
    assert _sgr(_LOW_CONF_STYLE) in out  # the low-confidence run is marked …
    assert "\x1b[1m" in out              # … and the markdown bold ("Canberra") survived


def test_low_confidence_marking_survives_no_color():
    """The point of moving off plain red. `NO_COLOR=1` (and monochrome terminals, and GIF color
    compression) erases a pure hue completely — the receipt would say `◌ 3 uncertain spans` with
    nothing marked in the body to look at. The marking must carry a non-color attribute, and must
    stay distinguishable from the human-correction style, which also survives."""
    import io

    from rich.console import Console
    from rich.text import Text

    from tui.ui._base import _HUMAN_STYLE, _LOW_CONF_STYLE

    for style in (_LOW_CONF_STYLE, _HUMAN_STYLE):
        buf = io.StringIO()
        Console(file=buf, force_terminal=True, no_color=True, width=40).print(
            Text("marked", style=style))
        assert "\x1b[" in buf.getvalue(), style   # still marked with no color at all

    # …and the two remain telling apart without color: red-vs-cyan alone would not.
    stripped = [_sgr(s).replace("\x1b[", "").rstrip("m")
                for s in (_LOW_CONF_STYLE, _HUMAN_STYLE)]
    assert stripped[0] != stripped[1]
    # Low confidence is a caution, not a failure: it must not borrow the risk vocabulary.
    assert "red" not in _LOW_CONF_STYLE


def test_mark_segments_preserves_style_and_marks_by_content():
    # The mechanism: reddening a Segment stream by phrase, combining red into the existing style
    # (a bold phrase stays bold-red), and leaving non-matching segments untouched.
    import importlib

    from rich.segment import Segment
    from rich.style import Style

    R = importlib.import_module("tui.ui.response")
    segs = [Segment("plain ", Style()),
            Segment("roughly four million", Style(bold=True)),
            Segment(" tail", Style())]
    out = list(R._mark_segments(segs, ["roughly four million"], "red"))
    red = [s for s in out if s.style and s.style.color and s.style.color.name == "red"]
    assert red and red[0].text == "roughly four million"
    assert red[0].style.bold  # existing bold combined with the red
    # A phrase absent from any single segment (here split by a boundary) is simply not marked.
    split = [Segment("roughly ", Style()), Segment("four", Style(bold=True)),
             Segment(" million", Style())]
    out2 = list(R._mark_segments(split, ["roughly four million"], "red"))
    assert not any(s.style and s.style.color and s.style.color.name == "red" for s in out2)


def test_frozen_tail_render_survives_confidence(capsys):
    # The freeze editor's print with an overlay present — smoke both paths' tolerance.
    from tui.ui.correction import _print_frozen

    text = "aa bb cc"
    conf = confidence.align_chunk(text, [_lp("aa", LOW), _lp(" bb", LOW), _lp(" cc", LOW)])

    # Default: the one-line summary only — the pre-filled editor is the single presentation of
    # the frozen text, so re-printing the body here just made the answer move again.
    _print_frozen(text, [], conf)
    out = capsys.readouterr().out
    assert "✂ frozen" in out and "8 chars" in out
    assert "low-confidence run" in out
    assert "aa bb cc" not in out

    # The wizard floor has no editor, so it still gets the body.
    _print_frozen(text, [], conf, show_tail=True)
    assert "aa bb cc" in capsys.readouterr().out


def test_frozen_summary_counts_earlier_corrections():
    from tui.ui.correction import _frozen_summary

    spans = [{"start": 0, "end": 2, "author": "model"},
             {"start": 2, "end": 5, "author": "human"}]
    line = _frozen_summary("abcdefgh", spans, None)
    assert "8 chars" in line and "1 earlier correction" in line
    # Garbage spans are tolerated — the summary is additive, never the freeze's problem.
    assert "8 chars" in _frozen_summary("abcdefgh", ["nope", None], None)


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


# --- hysteresis + the closed-class stoplist (transplanted from the confidence_coloring isolate) ------
#
# Two-threshold hysteresis: a run OPENS on tokens under the enter threshold (the min_run onset
# floor is unchanged) and, once building, EXTENDS through tokens under the looser exit threshold
# instead of one p=0.21 token closing it mid-phrase. Closed-class words (the, of, is, …) draw low
# mass from many valid continuations, so they are NEUTRAL like punctuation: they ride along a run
# without counting toward the floor or breaking it, and never form a run on their own.

MID = math.log(0.25)  # above enter=0.20, below the default exit (0.30)


def test_hysteresis_extends_an_open_run_through_a_mid_token():
    text = "aa bb cc dd ee"
    ents = _entries(text, [("aa", LOW), (" bb", LOW), (" cc", LOW), (" dd", MID), (" ee", LOW)])
    # Without hysteresis: [aa..cc] then a lone ee (< min_run) — the phrase is cut at dd.
    assert confidence.low_runs(ents, text, threshold_p=0.20, exit_p=0.20) == [(0, 8)]
    # With the exit threshold, dd rides along and ee joins the run.
    assert confidence.low_runs(ents, text, threshold_p=0.20, exit_p=0.30) == [(0, 14)]


def test_hysteresis_never_opens_a_run_by_itself():
    text = "aa bb cc"
    ents = _entries(text, [("aa", MID), (" bb", MID), (" cc", MID)])
    assert confidence.low_runs(ents, text, threshold_p=0.20, exit_p=0.30) == []


def test_default_exit_threshold_is_looser_than_enter_and_bounded():
    assert confidence.exit_threshold(0.20) > 0.20
    assert confidence.exit_threshold(0.90) <= 0.95
    assert confidence.exit_threshold(0.0) == 0.0


def test_stopwords_are_neutral_bridges():
    text = "aa the bb of cc"
    ents = _entries(text, [("aa", LOW), (" the", HIGH), (" bb", LOW), (" of", HIGH), (" cc", LOW)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == [(0, 15)]
    # …but stopwords alone never form a run, however unlikely the model found them.
    text2 = "the of and"
    ents2 = _entries(text2, [("the", LOW), (" of", LOW), (" and", LOW)])
    assert confidence.low_runs(ents2, text2, threshold_p=0.35) == []


def test_stopword_match_is_casefolded_and_punctuation_trimmed():
    assert confidence.is_stopword(" The")
    assert confidence.is_stopword("(and")
    assert confidence.is_stopword("'s")
    assert confidence.is_stopword(" it's")
    assert not confidence.is_stopword(" theory")
    assert not confidence.is_stopword("Paris")


def test_a_content_high_token_still_breaks_the_run():
    text = "aa bb Paris cc dd ee"
    ents = _entries(text, [("aa", LOW), (" bb", LOW), (" Paris", HIGH),
                           (" cc", LOW), (" dd", LOW), (" ee", LOW)])
    assert confidence.low_runs(ents, text, threshold_p=0.35) == [(12, 20)]


# --- per-model calibrated thresholds (C3, from the confidence_coloring isolate) -----------------------
#
# The isolate's calibration data showed a 6-10x spread in clean-output surprisal across model sizes,
# so one fixed threshold is wrong for most bindings. `runtime.confidence_threshold: auto` (the new
# default) looks the synthesizer's model up in the tracked calibration table (generated by
# utilities/confidence_calibrate.py under main's OWN token scoring — the isolate's taus were
# word/min-p/top-k numbers and don't transfer); a NUMBER in config is an explicit fixed override;
# an uncalibrated model falls back to the built-in default.


def _cfg(monkeypatch, value, exit_value=None):
    from config import get_config

    monkeypatch.setitem(get_config()._data["runtime"], "confidence_threshold", value)
    monkeypatch.setitem(get_config()._data["runtime"], "confidence_exit_threshold", exit_value)


def test_auto_threshold_uses_the_calibration_table_for_the_synthesizer(monkeypatch):
    from core import confidence_calibration as table

    _cfg(monkeypatch, "auto")
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "fake-model:1b")
    monkeypatch.setattr(table, "CALIBRATION",
                        {"fake-model:1b": {"enter": 0.07, "exit": 0.11, "tokens": 999}})
    assert confidence.threshold() == 0.07
    assert confidence.exit_threshold() == 0.11
    assert confidence.calibration_for("FAKE-MODEL:1B")["enter"] == 0.07   # tag case-insensitive


def test_auto_threshold_falls_back_for_an_uncalibrated_model(monkeypatch):
    from core import confidence_calibration as table

    _cfg(monkeypatch, "auto")
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "nobody:0b")
    monkeypatch.setattr(table, "CALIBRATION", {})
    assert confidence.threshold() == confidence._DEFAULT_THRESHOLD
    assert confidence.calibration_for("nobody:0b") is None
    assert confidence.exit_threshold() == min(confidence._EXIT_CAP,
                                              confidence._DEFAULT_THRESHOLD * confidence._EXIT_FACTOR)


def test_a_numeric_config_threshold_is_an_explicit_override(monkeypatch):
    from core import confidence_calibration as table

    _cfg(monkeypatch, 0.33)
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "fake-model:1b")
    monkeypatch.setattr(table, "CALIBRATION", {"fake-model:1b": {"enter": 0.07, "exit": 0.11}})
    assert confidence.threshold() == 0.33
    # a numeric enter override also disables the table's exit (the pair belongs together)
    assert confidence.exit_threshold() == min(confidence._EXIT_CAP, 0.33 * confidence._EXIT_FACTOR)


def test_garbage_config_threshold_falls_back_safely(monkeypatch):
    _cfg(monkeypatch, "lots")
    monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "nobody:0b")
    assert confidence.threshold() == confidence._DEFAULT_THRESHOLD


def test_shipped_calibration_table_is_well_formed():
    from core import confidence_calibration as table

    for tag, rec in table.CALIBRATION.items():
        assert tag == tag.lower(), tag
        assert 0.0 < rec["enter"] <= rec["exit"] < 1.0, (tag, rec)
        # measured (tokens > 0) — or explicitly marked as an ESTIMATE naming what it came from,
        # never a silent guess wearing a measurement's clothes.
        assert rec["tokens"] > 0 or (
            str(rec.get("source", "")).lower() == "estimated"
            and str(rec.get("estimated_from", "")).strip()
        ), (tag, rec)


class TestOverlayResolutionOrder:
    """pin > user overlay > shipped table > built-in default."""

    @pytest.fixture
    def bound(self, isolated_paths, monkeypatch):
        from core import confidence, confidence_store

        confidence_store.store_path().parent.mkdir(parents=True, exist_ok=True)
        confidence_store._reset_cache()
        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "qwen3.8:27b")
        yield confidence_store
        confidence_store._reset_cache()

    def _no_pin(self, monkeypatch):
        from core import confidence

        monkeypatch.setattr(confidence, "_configured_threshold", lambda: None)

    def test_the_shipped_table_is_used_when_there_is_no_overlay(self, bound, monkeypatch):
        from core import confidence, confidence_calibration

        self._no_pin(monkeypatch)
        shipped = confidence_calibration.CALIBRATION["qwen3.8:27b"]["enter"]
        assert confidence.threshold() == pytest.approx(shipped)

    def test_the_overlay_wins_over_the_shipped_table(self, bound, monkeypatch):
        from core import confidence

        self._no_pin(monkeypatch)
        bound.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        assert confidence.threshold() == pytest.approx(0.41)
        assert confidence.exit_threshold() == pytest.approx(0.62)

    def test_an_explicit_numeric_pin_still_wins_over_the_overlay(self, bound, monkeypatch):
        from core import confidence

        bound.write_entry("qwen3.8:27b", 0.41, 0.62, source="manual")
        monkeypatch.setattr(confidence, "_configured_threshold", lambda: 0.11)
        assert confidence.threshold() == pytest.approx(0.11)

    def test_an_uncalibrated_model_falls_back_to_the_builtin_default(self, monkeypatch,
                                                                    isolated_paths):
        from core import confidence, confidence_store

        confidence_store._reset_cache()
        self._no_pin(monkeypatch)
        monkeypatch.setattr(confidence, "_synthesizer_model", lambda: "nothing:known")
        assert confidence.threshold() == pytest.approx(confidence._DEFAULT_THRESHOLD)

    def test_a_garbled_overlay_degrades_to_the_shipped_table(self, bound, monkeypatch):
        from core import confidence, confidence_calibration

        self._no_pin(monkeypatch)
        bound.store_path().write_text("{ not json", encoding="utf-8")
        bound._reset_cache()
        shipped = confidence_calibration.CALIBRATION["qwen3.8:27b"]["enter"]
        assert confidence.threshold() == pytest.approx(shipped)
