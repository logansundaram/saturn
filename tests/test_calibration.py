"""The measurement core behind /confidence tune — pure parts only, no daemon."""

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
