"""
utilities/confidence_calibrate.py's `--inherit` path must produce a row the committed contract
tests accept — `tests/test_calibration.py`'s `test_an_unmeasured_entry_is_marked_as_an_estimate`
and `tests/test_confidence.py`'s `test_shipped_calibration_table_is_well_formed` both require an
unmeasured row (`tokens == 0`) to carry `source: 'estimated'` plus a non-empty `estimated_from`
string. This pins the generator's `inherit_record()` helper (the `--inherit` code path, factored
out so it's testable without hitting the daemon or writing core/confidence_calibration.py) to
that same vocabulary, so the generator and the contract can't drift apart again.
"""

from utilities.confidence_calibrate import inherit_record

_SRC_REC = {
    "enter": 0.2229, "exit": 0.4247, "tokens": 1279, "prompts": 55, "at": "2026-08-16",
}


def _assert_estimate_contract(tag, rec):
    """The exact predicate tests/test_calibration.py::test_an_unmeasured_entry_is_marked_as_an_estimate
    and tests/test_confidence.py::test_shipped_calibration_table_is_well_formed apply."""
    assert 0.0 < float(rec["enter"]) <= float(rec["exit"]) < 1.0, tag
    assert int(rec.get("tokens", 0)) == 0
    assert str(rec.get("source", "")).lower() == "estimated", tag
    assert str(rec.get("estimated_from", "")).strip(), tag


def test_inherit_record_satisfies_the_estimated_row_contract():
    rec = inherit_record("qwen3.8:27b", "qwen3.6:27b", _SRC_REC, today="2026-08-16")
    _assert_estimate_contract("qwen3.8:27b", rec)


def test_inherit_record_keeps_the_source_thresholds():
    rec = inherit_record("some-tag:1b", "qwen3.6:27b", _SRC_REC, today="2026-08-16")
    assert rec["enter"] == _SRC_REC["enter"]
    assert rec["exit"] == _SRC_REC["exit"]


def test_inherit_record_measured_nothing_of_its_own():
    # An inherited row is a borrowed estimate, not a measurement — tokens/prompts must say so.
    rec = inherit_record("some-tag:1b", "qwen3.6:27b", _SRC_REC, today="2026-08-16")
    assert rec["tokens"] == 0
    assert rec["prompts"] == 0


def test_inherit_record_names_the_source_and_its_basis():
    rec = inherit_record("qwen3.8:27b", "qwen3.6:27b", _SRC_REC, today="2026-08-16")
    basis = rec["estimated_from"]
    assert "qwen3.6:27b" in basis
    assert str(_SRC_REC["tokens"]) in basis
    assert _SRC_REC["at"] in basis
    assert "/confidence tune" in basis


def test_inherit_record_never_carries_the_old_bare_marker():
    # The pre-2026-08-16 shape (`inherited_from`, no `source`) predates the estimated-row
    # vocabulary the contract tests now enforce; the generator must never re-emit it.
    rec = inherit_record("qwen3.8:27b", "qwen3.6:27b", _SRC_REC, today="2026-08-16")
    assert "inherited_from" not in rec
