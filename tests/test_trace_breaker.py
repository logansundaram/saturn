"""stores/trace.Tracer's circuit breaker: a per-delta write failure trips it (later deltas
no-op instead of stalling on sqlite's busy timeout), but `end_run` is EXEMPT — the run's
terminal status/answer row must not be lost to a transient early lock — and the breaker
re-arms per run. `broken` is the surface agent.py's per-turn warning reads. Plus the tracer's
other open-time and run-time seams: the current-run id, and the legacy FTS triggers it drops."""

import types

import pytest

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


def test_current_run_id_is_set_during_a_run_and_cleared_after(tmp_path, monkeypatch):
    """The seam that stamps `run=<id>` provenance on a fact `remember` stores mid-turn."""
    from stores import trace

    monkeypatch.setattr(trace, "_CURRENT_RUN_ID", None)  # an earlier test's run left open
    t = trace.Tracer(str(tmp_path / "db.sqlite"))
    assert trace.current_run_id() is None
    rid = t.start_run("thread", "q")
    assert trace.current_run_id() == rid
    t.end_run(rid, "ok", "a")
    assert trace.current_run_id() is None
    t.conn.close()


# ── a DB written while `/trace search` existed (cut 2026-09-30) ─────────────────────────────
# Its FTS5 triggers live in the DB file and fire on every INSERT/UPDATE of `runs`; on a SQLite
# build without fts5 they fail every start_run. The tracer drops them at open; the orphaned
# index table is left alone (no destructive migration of the user's db).

_LEGACY_FTS = """
CREATE VIRTUAL TABLE runs_fts USING fts5(query, response, content='runs', content_rowid='run_id');
CREATE TRIGGER runs_fts_ai AFTER INSERT ON runs BEGIN
  INSERT INTO runs_fts(rowid, query, response) VALUES (new.run_id, new.query, new.response);
END;
CREATE TRIGGER runs_fts_ad AFTER DELETE ON runs BEGIN
  INSERT INTO runs_fts(runs_fts, rowid, query, response)
    VALUES ('delete', old.run_id, old.query, old.response);
END;
CREATE TRIGGER runs_fts_au AFTER UPDATE ON runs BEGIN
  INSERT INTO runs_fts(runs_fts, rowid, query, response)
    VALUES ('delete', old.run_id, old.query, old.response);
  INSERT INTO runs_fts(rowid, query, response) VALUES (new.run_id, new.query, new.response);
END;
"""


def _legacy_db(path, *, missing_module=False):
    import sqlite3

    from stores import trace

    conn = sqlite3.connect(str(path))
    conn.executescript(trace._SCHEMA)
    conn.executescript(_LEGACY_FTS)
    conn.execute("INSERT INTO runs (thread_id, query, status) VALUES ('t', 'old run', 'ok')")
    if missing_module:
        # Simulate a build without fts5: point the virtual table at a module that doesn't exist.
        conn.execute("PRAGMA writable_schema=1")
        conn.execute("UPDATE sqlite_master SET sql=replace(sql, 'USING fts5', 'USING no_such_module') "
                     "WHERE name='runs_fts'")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("missing_module", [False, True])
def test_legacy_fts_triggers_are_dropped_and_recording_works(tmp_path, missing_module):
    db = tmp_path / "db.sqlite"
    _legacy_db(db, missing_module=missing_module)
    t = Tracer(str(db))
    rid = t.start_run("thread", "does recording still work")
    assert rid == 2 and not t.broken
    t.end_run(rid, "ok", "yes")
    assert not t.broken
    names = {r[0] for r in t.conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert not names & {"runs_fts_ai", "runs_fts_ad", "runs_fts_au"}
    assert "runs_fts" in names                      # the table itself is left alone
    assert t.conn.execute("SELECT status, response FROM runs WHERE run_id = 2").fetchone() == ("ok", "yes")
    t.conn.close()
