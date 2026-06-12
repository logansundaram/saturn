"""
Persistent, hash-chained egress log (egress.py) — the durable twin of the in-memory ledger.

Covers: disk append on record(), the chain verifying intact, tamper detection (an edited middle
line breaks the chain), the disable toggle, and the cross-session summary aggregation.
"""

import json

from trust import egress


def _reset():
    egress.clear()


def test_record_appends_to_disk(isolated_paths):
    _reset()
    egress.record("web_search", "api.example.com", "a query", n_bytes=42)
    rows = egress.read_log()
    assert len(rows) == 1
    assert rows[0]["host"] == "api.example.com"
    assert rows[0]["n_bytes"] == 42
    assert rows[0]["session"] == egress._SESSION_ID
    assert rows[0]["prev"] == ""  # first line links to the empty tip
    assert "h" in rows[0]


def test_chain_verifies_intact(isolated_paths):
    _reset()
    for i in range(5):
        egress.record("http_request", f"h{i}.example.com", n_bytes=i)
    v = egress.verify_log()
    assert v["ok"] is True
    assert v["lines"] == 5
    assert v["broken_at"] is None
    # Each line's prev links to the previous line's h.
    rows = egress.read_log()
    for a, b in zip(rows, rows[1:]):
        assert b["prev"] == a["h"]


def test_tampered_middle_line_breaks_chain(isolated_paths):
    _reset()
    for i in range(4):
        egress.record("web_search", f"h{i}.example.com", n_bytes=i)
    path = isolated_paths / "database" / "egress.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["host"] = "evil.example.com"  # edit the payload, leave the stored hash
    lines[1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v = egress.verify_log()
    assert v["ok"] is False
    assert v["broken_at"] == 2  # 1-based index of the first bad line


def test_disable_toggle_skips_disk(isolated_paths, monkeypatch):
    _reset()
    from config import get_config

    get_config().set("runtime.egress_log", False)
    try:
        egress.record("web_search", "api.example.com")
        assert egress.read_log() == []
    finally:
        get_config().set("runtime.egress_log", True)


def test_record_clips_unbounded_labels(isolated_paths):
    # detail/host are display labels — unclipped they bloat every render AND can push a durable
    # line past the tail window the chain appender reads.
    _reset()
    egress.record("http_request", "h" * 5000 + ".example.com", "y" * 20000)
    e = egress.events()[-1]
    assert len(e.detail) <= 500
    assert len(e.host) <= 200
    assert len(egress.read_log()[0]["detail"]) <= 500


def test_tail_tip_survives_oversize_legacy_line(isolated_paths):
    # A line longer than the 8KB tail window (e.g. written before detail clipping existed) must
    # not parse as a mid-line fragment: that would chain the next append from prev="" and break
    # verification forever. The window grows until it provably spans a whole line.
    _reset()
    path = isolated_paths / "database" / "egress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": "t", "channel": "http_request", "host": "h.example.com",
               "detail": "x" * 20000, "provider": "", "n_bytes": 0, "redactions": 0,
               "status": "sent", "session": "legacy"}
    line = dict(payload, prev="", h=egress._entry_hash("", payload))
    path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")

    egress.record("web_search", "api.example.com")  # must chain from the REAL tip
    v = egress.verify_log()
    assert v["ok"] is True
    assert v["lines"] == 2


def test_garbled_tail_line_breaks_chain(isolated_paths):
    # An unparseable line is a broken chain, never silently skipped — destroying the newest
    # entry must not verify as intact.
    _reset()
    for i in range(3):
        egress.record("web_search", f"h{i}.example.com")
    path = isolated_paths / "database" / "egress.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1][: len(lines[-1]) // 2]  # truncate the last record mid-JSON
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v = egress.verify_log()
    assert v["ok"] is False
    assert v["broken_at"] == 3


def test_garbage_file_does_not_verify(isolated_paths):
    _reset()
    path = isolated_paths / "database" / "egress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a json record at all\n", encoding="utf-8")
    v = egress.verify_log()
    assert v["ok"] is False
    assert v["broken_at"] == 1


def test_events_since_survives_trim_and_clear(isolated_paths, monkeypatch):
    # Seq-keyed turn slices: neither the _MAX_EVENTS trim nor a mid-session clear() may shift a
    # turn-start mark onto the wrong events (the trust receipt reads these).
    _reset()
    monkeypatch.setattr(egress, "_MAX_EVENTS", 3)
    egress.record("web_search", "old1.example.com")
    egress.record("web_search", "old2.example.com")
    mark = egress.next_seq()
    egress.record("web_search", "new1.example.com")
    egress.record("web_search", "new2.example.com")  # trims old1 — indexes shift, seqs don't
    assert [e.host for e in egress.events_since(mark)] == [
        "new1.example.com", "new2.example.com"]

    egress.clear()
    assert egress.events_since(mark) == []  # cleared events never resurface
    egress.record("web_search", "after.example.com")
    assert [e.host for e in egress.events_since(mark)] == ["after.example.com"]


def test_locked_log_skips_append_never_forks(isolated_paths, monkeypatch):
    # If the cross-process lock can't be had, the append must be SKIPPED — writing unlocked
    # would chain from a tip another session is moving, fork the chain, and turn every later
    # verify into a permanent false tamper alarm.
    _reset()
    egress.record("web_search", "first.example.com")
    # A scoped context, NOT monkeypatch.undo(): undo would also revert isolated_paths'
    # redirections and the next record() would hit the REAL database/egress.log.
    with monkeypatch.context() as m:
        m.setattr(egress, "_lock_handle", lambda fh: False)
        egress.record("web_search", "skipped.example.com")
    egress.record("web_search", "third.example.com")

    rows = egress.read_log()
    assert [r["host"] for r in rows] == ["first.example.com", "third.example.com"]
    assert egress.verify_log()["ok"] is True  # the chain stays linear
    # the in-memory ledger still has the event — only the durable line was skipped
    assert "skipped.example.com" in [e.host for e in egress.events()]


def test_cleared_since_marks_wiped_slices(isolated_paths):
    # The Glass Box treats a possibly-clear-emptied slice as UNKNOWN; cleared_since is how it
    # tells "nothing sent" from "the events were wiped".
    _reset()
    egress.record("web_search", "a.example.com")
    mark = egress.next_seq()
    egress.record("web_search", "b.example.com")
    assert not egress.cleared_since(mark)
    egress.clear()
    assert egress.cleared_since(mark)          # b's event is gone — the slice lies
    later = egress.next_seq()
    assert not egress.cleared_since(later)     # a fresh mark after the clear is trustworthy
    assert not egress.cleared_since(0)         # 0 = no mark, handled by callers as unknown


def test_read_log_limit_reads_tail(isolated_paths):
    _reset()
    for i in range(20):
        egress.record("web_search", f"h{i:02d}.example.com")
    rows = egress.read_log(3)
    assert [r["host"] for r in rows] == [
        "h17.example.com", "h18.example.com", "h19.example.com"]


def test_log_tip_matches_chain_walk(isolated_paths, monkeypatch):
    # log_tip is the anchor source for signed artifacts: its tip must be exactly the last
    # line's `h` (the same chain head verify_log's walk ends on) and its count the raw line
    # count — whether served from the append cache or a fresh tail read.
    _reset()
    for i in range(4):
        egress.record("web_search", f"h{i}.example.com")
    tip = egress.log_tip()
    rows = egress.read_log()
    assert tip == {"tip_hash": rows[-1]["h"], "line_count": 4}
    v = egress.verify_log()
    assert v["ok"] is True and v["lines"] == tip["line_count"]
    # Drop the append cache — the tail-read path must agree with the cached one.
    monkeypatch.setattr(egress, "_TIP_CACHE", None)
    assert egress.log_tip() == tip


def test_log_tip_absent_cases(isolated_paths):
    # No durable log / knob off → None: the anchor field must be ABSENT, never a fake value.
    _reset()
    assert egress.log_tip() is None  # no log file yet in this fresh tree
    from config import get_config

    egress.record("web_search", "a.example.com")
    get_config().set("runtime.egress_log", False)
    try:
        assert egress.log_tip() is None  # knob off — even though a file exists
    finally:
        get_config().set("runtime.egress_log", True)
    assert egress.log_tip() is not None


def test_log_tip_never_fakes_a_garbled_tail(isolated_paths):
    # An unreadable tail line yields NO tip — an anchor must only commit a real chain head.
    _reset()
    egress.record("web_search", "a.example.com")
    path = isolated_paths / "database" / "egress.log"
    with open(path, "ab") as fh:
        fh.write(b"garbage not json\n")
    assert egress.log_tip() is None


def test_summarize_events_shared_accounting():
    evs = [
        egress.EgressEvent(ts="t", channel="web_search", host="a.example.com", n_bytes=10),
        egress.EgressEvent(ts="t", channel="llm", host="b.example.com", n_bytes=20),
        egress.EgressEvent(ts="t", channel="web_search", host="a.example.com", n_bytes=5,
                           status=egress.BLOCKED),
    ]
    agg = egress.summarize_events(evs)
    assert agg["sent"] == 2 and agg["blocked"] == 1
    assert agg["bytes"] == 30                      # blocked bytes never count
    assert agg["hosts"] == ["a.example.com", "b.example.com"]  # first-seen order
    assert agg["channels"] == ["web_search", "llm"]


def test_log_summary_aggregates(isolated_paths):
    _reset()
    egress.record("web_search", "a.example.com", n_bytes=10)
    egress.record("http_request", "b.example.com", n_bytes=20, status=egress.BLOCKED)
    egress.record("web_search", "a.example.com", n_bytes=30)
    s = egress.log_summary()
    assert s["lines"] == 3
    assert s["sent"] == 2
    assert s["blocked"] == 1
    assert s["bytes"] == 40  # only SENT events count toward bytes
    assert s["hosts"] == ["a.example.com"]
    assert egress._SESSION_ID in s["sessions"]
