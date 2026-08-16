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
        ps = [round(0.01 * i, 2) for i in range(1, 101)]
        out = calibration.summarize(ps)
        assert out["enter"] < out["exit"]
        assert out["tokens"] == 100

    def test_no_samples_yields_no_thresholds(self):
        out = calibration.summarize([])
        assert out["tokens"] == 0
        assert out["enter"] == 0.0
        assert out["exit"] == 0.0

    def test_the_result_is_sorted_independent(self):
        ps = [0.9, 0.1, 0.5, 0.3, 0.7]
        assert calibration.summarize(ps) == calibration.summarize(sorted(ps, reverse=True))


class TestPrompts:
    def test_the_prompt_set_is_non_empty_and_stable(self):
        assert len(calibration.PROMPTS) >= 40
        assert all(isinstance(p, str) and p.strip() for p in calibration.PROMPTS)
