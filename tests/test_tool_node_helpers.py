"""nodes/tools.py helpers — the observation clamp (gotcha #5: one big tool result must
never overflow the context window) and the call/preview formatters the trace rides on."""

from nodes.tools import (
    _clamp_observation,
    _fmt_call,
    _preview,
    _MAX_OBSERVATION,
    _MAX_RESULT_PREVIEW,
)


def test_clamp_short_passthrough():
    s = "x" * _MAX_OBSERVATION
    assert _clamp_observation(s) is s


def test_clamp_keeps_head_and_tail_with_marker():
    head_sentinel = "HEAD-SENTINEL"
    tail_sentinel = "TAIL-SENTINEL"
    s = head_sentinel + "x" * (_MAX_OBSERVATION * 3) + tail_sentinel
    out = _clamp_observation(s)
    assert out.startswith(head_sentinel)
    assert out.endswith(tail_sentinel)
    assert "truncated" in out
    # Bounded: the clamped text can't itself be unboundedly large.
    assert len(out) < _MAX_OBSERVATION + 200


def test_preview_single_capped_line():
    out = _preview("line one\nline   two\n" + "y" * 500)
    assert "\n" not in out
    assert len(out) <= _MAX_RESULT_PREVIEW
    assert out.endswith("…")
    assert _preview("short") == "short"


def test_fmt_call_pairs_and_truncates_args():
    out = _fmt_call("calculate", {"expression": "1+1"})
    assert out == "calculate(expression='1+1')"
    long = _fmt_call("write_file", {"content": "z" * 1000})
    assert len(long) < 1000  # the arg repr was capped
    assert long.startswith("write_file(content=")
    assert _fmt_call("list_directory", {}) == "list_directory()"
