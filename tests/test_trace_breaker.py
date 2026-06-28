"""stores/trace.Tracer's circuit breaker: a per-delta write failure trips it (later deltas
no-op instead of stalling on sqlite's busy timeout), but `end_run` is EXEMPT — the run's
terminal status/answer row must not be lost to a transient early lock — and the breaker
re-arms per run. `broken` is the surface agent.py's per-turn warning reads."""

import types

from stores.trace import Tracer


class _FlakyConn:
    """Wraps the real sqlite connection; fails INSERTs into `events` while `fail_events` is on.
    (sqlite3.Connection attributes are C-level and can't be monkeypatched directly.)"""

    def __init__(self, real):
        self._real = real
        self.fail_events = True

    def execute(self, sql, *args):
        if self.fail_events and "INSERT INTO events" in sql:
            raise RuntimeError("database is locked")
        return self._real.execute(sql, *args)

    def commit(self):
        return self._real.commit()


def test_breaker_trips_on_delta_failure_but_end_run_still_lands(tmp_path):
    tr = Tracer(str(tmp_path / "t.sqlite"))
    rid = tr.start_run("th", "the query")
    real_conn = tr.conn
    tr.conn = _FlakyConn(real_conn)

    tr.log_event(rid, "agent", {"x": 1})  # first failure trips the breaker
    assert tr.broken
    tr.log_event(rid, "agent", {"x": 2})  # no-ops silently — no stall per delta

    # The lock clears before turn end; end_run is exempt from the breaker, so the run's
    # terminal row still lands (previously it no-opped and the run stayed 'running' forever).
    tr.conn.fail_events = False
    tr.end_run(rid, "ok", "the answer")
    row = real_conn.execute(
        "SELECT status, response FROM runs WHERE run_id = ?", (rid,)
    ).fetchone()
    assert row == ("ok", "the answer")

    # The breaker re-arms on the next run — one retry per turn, never a permanently dead trace.
    tr.conn = real_conn
    rid2 = tr.start_run("th2", "next query")
    assert not tr.broken
    tr.log_event(rid2, "agent", {"y": 1})
    n = real_conn.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", (rid2,)).fetchone()[0]
    assert n == 1


def test_loop_warning_reads_broken(tmp_path):
    # agent._trace_warning is the documented consumer of `broken`: silent degradation must
    # surface to the user once per affected turn.
    import agent

    tr = Tracer(str(tmp_path / "t.sqlite"))
    assert agent._trace_warning(tr) is None
    tr._broken = True
    note = agent._trace_warning(tr)
    assert note and "degraded" in note

    # Tolerates any tracer-shaped object (the warning must never be a new crash source).
    assert agent._trace_warning(types.SimpleNamespace(broken=False)) is None
