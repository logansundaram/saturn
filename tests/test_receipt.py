"""
Trust receipt (receipt.py) — the pure trust_parts builder over synthetic egress events, plus the
arg-scan helper the gate's secret warning uses (redaction.scan_args).
"""

from trust import egress
from trust import receipt
from trust import redaction


def _ev(host="api.tavily.com", n_bytes=0, status=egress.SENT):
    return egress.EgressEvent(ts="t", channel="web_search", host=host,
                              n_bytes=n_bytes, status=status)


# --- trust_parts -----------------------------------------------------------------------------

def test_local_only_when_nothing_sent():
    assert receipt.trust_parts([], 0) == ["local-only"]


def test_blocked_only_still_reads_local():
    parts = receipt.trust_parts([_ev(status=egress.BLOCKED)], 0)
    assert parts[0] == "local-only"
    assert "⛔ 1 blocked" in parts


def test_send_summary_bytes_and_host():
    parts = receipt.trust_parts([_ev(n_bytes=2048)], 0)
    assert parts == ["⇅ 1 send · 2.0KB → api.tavily.com"]


def test_multiple_hosts_collapse_to_plus_n():
    evs = [_ev(), _ev(host="api.anthropic.com"), _ev(host="api.tavily.com")]
    (label,) = receipt.trust_parts(evs, 0)
    assert label.startswith("⇅ 3 sends")
    assert "api.tavily.com +1" in label


def test_gate_count_appends():
    assert receipt.trust_parts([], 3)[-1] == "3 calls gated"
    assert receipt.trust_parts([], 1)[-1] == "1 call gated"
    assert "gated" not in " ".join(receipt.trust_parts([], 0))


def test_trust_spans_kinds():
    # The styled receipt keys its semantic colors off the kinds; the plain path renders the
    # identical bare text — the two views must never disagree.
    spans = receipt.trust_spans([_ev(n_bytes=2048), _ev(status=egress.BLOCKED)], 2)
    assert [k for _, k in spans] == ["sent", "blocked", "gated"]
    assert receipt.trust_spans([], 0) == [("local-only", "local")]
    assert receipt.trust_parts([], 2) == [t for t, _ in receipt.trust_spans([], 2)]


def test_turn_parts_slices_from_mark(isolated_paths):
    egress.clear()
    try:
        egress.record("web_search", "old.example.com")
        mark = receipt.mark()
        egress.record("web_search", "new.example.com", n_bytes=10)
        (label,) = receipt.turn_parts(mark, 0)
        assert "new.example.com" in label and "old.example.com" not in label
    finally:
        egress.clear()


def test_turn_parts_not_fooled_by_ledger_trim(isolated_paths, monkeypatch):
    # The mark is a seq, not a list index: when the capped ledger trims its oldest entries
    # mid-turn, the receipt must still report the turn's sends — never 'local-only' over a turn
    # that sent (the one lie the trust receipt must never tell).
    egress.clear()
    monkeypatch.setattr(egress, "_MAX_EVENTS", 2)
    try:
        egress.record("web_search", "old.example.com")
        mark = receipt.mark()
        egress.record("web_search", "new1.example.com", n_bytes=10)
        egress.record("web_search", "new2.example.com", n_bytes=10)  # trim shifts indexes
        (label,) = receipt.turn_parts(mark, 0)
        assert label.startswith("⇅ 2 sends")
        assert "local-only" not in label
    finally:
        egress.clear()


def test_turn_parts_unknown_when_no_mark():
    # Mark 0 = no turn recorded (headless / before the first turn): the slice is UNKNOWN, and the
    # receipt must say so rather than summarizing the whole ledger as this turn's egress — and it
    # must never assert 'local-only' over a slice it can't vouch for.
    parts = receipt.turn_parts(0, 2)
    assert parts[0] == "egress unknown"
    assert "local-only" not in parts
    assert parts[-1] == "2 calls gated"
    assert [k for _, k in receipt.turn_spans(0, 0)] == ["unknown"]


def test_turn_parts_unknown_after_clear_past_mark(isolated_paths):
    # A `/privacy egress clear` that wiped events recorded AFTER the mark means the slice may be
    # missing real sends — same unknown presentation, never 'local-only' (the contract /trace
    # answer applies via egress.cleared_since before trusting the live slice).
    egress.clear()
    try:
        mark = receipt.mark()
        egress.record("web_search", "x.example.com", n_bytes=10)
        egress.clear()  # wipes an event recorded after the mark
        assert receipt.turn_parts(mark, 0) == ["egress unknown"]
    finally:
        egress.clear()


def test_turn_parts_after_mid_session_clear(isolated_paths):
    # `/privacy egress clear` empties the ledger but must not corrupt an outstanding turn mark.
    egress.clear()
    try:
        egress.record("web_search", "before.example.com")
        mark = receipt.mark()
        egress.clear()
        assert receipt.turn_parts(mark, 0) == ["local-only"]  # nothing sent since the mark
        egress.record("web_search", "after.example.com", n_bytes=10)
        (label,) = receipt.turn_parts(mark, 0)
        assert "after.example.com" in label
    finally:
        egress.clear()


# --- the gate's secret-in-args scan ----------------------------------------------------------

def test_scan_args_walks_nested_values():
    args = {
        "url": "https://api.example.com",
        "headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456"},
        "extra": [{"note": "key sk-ant-abcdefghijklmnopqrstuvwx leaked"}],
    }
    kinds = {f.kind for f in redaction.scan_args(args)}
    assert "bearer-token" in kinds
    assert "anthropic-key" in kinds


def test_scan_args_clean_and_non_string_tolerant():
    assert redaction.scan_args({"a": 1, "b": [True, None], "c": {"d": 2.5}}) == []
    assert redaction.scan_args(None) == []
