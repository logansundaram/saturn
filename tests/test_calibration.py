"""The measurement core behind /confidence tune — offline: the daemon is faked, never called."""

import math

import pytest

from core import calibration


class TestQuantile:
    def test_picks_the_expected_point(self):
        ps = [round(0.01 * i, 2) for i in range(1, 101)]
        assert calibration.quantile(ps, 0.05) == pytest.approx(0.05, abs=0.02)
        assert calibration.quantile(ps, 0.10) == pytest.approx(0.10, abs=0.02)

    def test_empty_is_zero(self):
        assert calibration.quantile([], 0.05) == 0.0

    def test_single_value(self):
        assert calibration.quantile([0.4], 0.05) == 0.4


class TestSummarize:
    def test_enter_is_the_5pct_point_and_exit_the_10pct(self):
        # The EXACT points, not merely enter < exit — p25/p75 would satisfy that ordering while
        # marking ~5x as much text. ps = 0.01 … 1.00, so quantile picks ordered[round(q * 99)]:
        # 0.05 -> index 5 -> 0.06, 0.10 -> index 10 -> 0.11.
        ps = [round(0.01 * i, 2) for i in range(1, 101)]
        out = calibration.summarize(ps)
        assert out["enter"] == 0.06
        assert out["exit"] == 0.11
        assert out["p50"] == 0.51   # round(0.5 * 99) = 50 (banker's rounding) -> ordered[50]
        assert out["tokens"] == 100

    def test_no_samples_yields_no_thresholds(self):
        out = calibration.summarize([])
        assert out["tokens"] == 0
        assert out["enter"] == 0.0
        assert out["exit"] == 0.0

    def test_the_result_is_sorted_independent(self):
        ps = [0.9, 0.1, 0.5, 0.3, 0.7]
        assert calibration.summarize(ps) == calibration.summarize(sorted(ps, reverse=True))

    def test_exit_never_falls_below_enter_on_a_flat_distribution(self):
        # p05 and p10 collide when every probability is identical — the clamp exists for exactly
        # this case; an inverted pair silently defeats core/confidence.exit_threshold's guard.
        out = calibration.summarize([0.4] * 100)
        assert out["exit"] >= out["enter"]


class TestPrompts:
    def test_the_prompt_set_is_non_empty_and_stable(self):
        assert len(calibration.PROMPTS) >= 40
        assert all(isinstance(p, str) and p.strip() for p in calibration.PROMPTS)


class TestShippedTable:
    """The shipped baseline and the size ladder must not drift apart: a bindable tag with no
    CALIBRATION row silently falls back to the built-in 0.20, i.e. marks that are NOT the
    per-model claim the feature makes. Nothing else guards this."""

    def _table(self):
        from core import confidence_calibration

        return confidence_calibration.CALIBRATION

    def test_every_ladder_tag_has_a_calibration_entry(self):
        from core import model_family as mf

        table = self._table()
        for _key, tag in mf.SIZE_LADDER:
            assert tag.lower() in table, tag

    def test_every_entry_has_a_sane_threshold_pair(self):
        # 0 < enter <= exit < 1: an inverted or out-of-range pair silently defeats
        # core/confidence.exit_threshold's sanity guard and the model behaves as uncalibrated.
        for tag, rec in self._table().items():
            enter, exit_ = float(rec["enter"]), float(rec["exit"])
            assert 0.0 < enter <= exit_ < 1.0, tag

    def test_an_unmeasured_entry_is_marked_as_an_estimate(self):
        """A row that carries no measurement must say `source: 'estimated'` and name what it was
        estimated from — otherwise a guess reads as a measurement in every readout."""
        from core import calibration as core_calibration

        for tag, rec in self._table().items():
            if int(rec.get("tokens", 0)) >= core_calibration.MIN_TOKENS:
                continue
            assert str(rec.get("source", "")).lower() == "estimated", tag
            assert str(rec.get("estimated_from", "")).strip(), tag


# ── measure(): the only new logic that decides a threshold ───────────────────────────────────
# Every other test monkeypatches it away, which left the load-bearing line — the isalnum /
# is_stopword filter that must MIRROR core/confidence.low_runs — with no guard at all. The
# "measured under main's own scoring unit" claim rests entirely on it, and it can drift silently.

class _Chunk:
    """One streamed chunk as langchain_ollama yields it: text + the daemon's logprob entries."""

    def __init__(self, content, tokens=None):
        self.content = content
        self.response_metadata = {}
        if tokens:
            self.response_metadata["logprobs"] = [
                {"token": tok, "logprob": lp} for tok, lp in tokens
            ]


@pytest.fixture
def daemon(monkeypatch):
    """A fake Ollama: `stream` replays scripted chunks per prompt, `_build` returns a stub."""
    from core import llms
    from trust import egress

    monkeypatch.setattr(egress, "ollama_is_local", lambda: True)
    monkeypatch.setattr(egress, "airgap_on", lambda: False)
    monkeypatch.setattr(llms, "_build", lambda provider, tag: object())

    script = {"chunks": [], "calls": [], "kwargs": []}

    def fake_stream(model, messages, *, tag="", **kwargs):
        script["calls"].append(messages)
        script["kwargs"].append(kwargs)
        return iter(script["chunks"])

    monkeypatch.setattr(llms, "stream", fake_stream)
    return script


class TestMeasure:
    # "Paris is the capital." split over two chunks; the second chunk's entries are only at the
    # right character offsets because measure passes offset=len(text) into align_chunk.
    _CHUNKS = [
        _Chunk("Paris is", [("Paris", -0.5), (" is", -4.0)]),
        _Chunk(" the capital.", [(" the", -3.0), (" capital", 0.25), (".", -6.0)]),
    ]

    def test_scores_only_the_tokens_low_runs_would_grade(self, daemon):
        """The filter must mirror core/confidence.low_runs exactly: punctuation-only tokens and
        closed-class stopwords are neutral there, so they are neutral here too. 'is'/'the' are
        stopwords and '.' has no alnum — only 'Paris' and 'capital' are content."""
        daemon["chunks"] = self._CHUNKS
        out = calibration.measure("fake:1b", ["q"])

        assert out["tokens"] == 2
        assert out["prompts"] == 1

    def test_offsets_are_global_not_chunk_relative(self, daemon):
        """A second chunk aligned at offset 0 would slice 'Pari' / 's is the' out of the text —
        different tokens, a different filter verdict, and probabilities attributed to the wrong
        words. The exact quantiles pin the two tokens that survived."""
        daemon["chunks"] = self._CHUNKS
        out = calibration.measure("fake:1b", ["q"])

        # ps = [exp(-0.5) for "Paris", 1.0 for " capital"]; p05 and p10 of a 2-sample both land
        # on the lower one.
        assert out["enter"] == round(math.exp(-0.5), 4)
        assert out["exit"] == round(math.exp(-0.5), 4)

    def test_a_positive_logprob_is_clamped_to_probability_one(self, daemon):
        """math.exp(min(logprob, 0.0)): daemons do emit tiny positive logprobs, and exp() of one
        is > 1 — a "probability" above certainty would drag a quantile upward."""
        daemon["chunks"] = [_Chunk("Paris", [("Paris", 0.25)])]
        out = calibration.measure("fake:1b", ["q"])

        assert out["tokens"] == 1
        assert out["enter"] == 1.0          # not exp(0.25) == 1.284

    def test_a_chunk_without_logprobs_contributes_nothing(self, daemon):
        daemon["chunks"] = [_Chunk("Paris is the capital.")]
        out = calibration.measure("fake:1b", ["q"])

        assert out["tokens"] == 0

    def test_on_progress_reports_the_per_prompt_delta(self, daemon):
        daemon["chunks"] = self._CHUNKS
        seen = []
        calibration.measure("fake:1b", ["q1", "q2"], on_progress=lambda *a: seen.append(a))

        # Each prompt replays the same script, so each adds the same 2 content tokens — the
        # 4th field is the DELTA, not the running total.
        assert seen == [(1, 2, "q1", 2), (2, 2, "q2", 2)]

    def test_it_asks_the_daemon_for_per_token_logprobs(self, daemon):
        daemon["chunks"] = self._CHUNKS
        calibration.measure("fake:1b", ["q"])

        assert daemon["kwargs"][0]["logprobs"] is True
        assert daemon["kwargs"][0]["reasoning"] is False   # a rationale is not answer prose

    def test_the_air_gap_refuses_an_off_machine_daemon(self, monkeypatch):
        """The measurement streams through core.llms' Ollama boundary; with the seal on and
        OLLAMA_HOST off-machine it must refuse BEFORE building a model, not leak the prompts."""
        from core import llms
        from trust import egress

        monkeypatch.setattr(egress, "ollama_is_local", lambda: False)
        monkeypatch.setattr(egress, "airgap_on", lambda: True)
        monkeypatch.setattr(llms, "_build",
                            lambda *a, **k: pytest.fail("built a model past the air-gap"))

        with pytest.raises(RuntimeError, match="air-gap"):
            calibration.measure("fake:1b", ["q"])
