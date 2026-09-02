"""Opening the trace to the agent (2026-09-02): an FTS5 index over `runs.query` / `runs.response`
behind the read-only `recall_runs` tool and `/trace search`, plus the current-run seam that
stamps provenance on facts stored mid-turn. Offline: a throwaway sqlite file per test.
"""

from commands._framework import CommandContext
from stores import trace


def _seed(db_path):
    from pathlib import Path

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    t = trace.Tracer(str(db_path))
    for q, a in [
        ("rename the Q3 reports", "Renamed 4 files under reports/ with the q3_ prefix."),
        ("summarize the invoice export", "The invoice export totals 12 rows; the date filter is missing."),
        ("what is the weather", "It is sunny."),
    ]:
        rid = t.start_run("thread", q)
        t.end_run(rid, "ok", a)
    t.conn.close()


def test_search_runs_matches_query_and_answer_words(tmp_path):
    db = tmp_path / "db.sqlite"
    _seed(db)
    hits = trace.search_runs(db, "invoice export")
    assert [h["run_id"] for h in hits] == [2]
    assert "date filter" in hits[0]["response"]
    # Words from the ANSWER hit too; every content term must match first (AND) ...
    assert [h["run_id"] for h in trace.search_runs(db, "q3_ prefix")] == [1]
    # ... and when no run has them all, any-term matches come back instead of nothing.
    assert {h["run_id"] for h in trace.search_runs(db, "invoice weather")} == {2, 3}
    assert trace.search_runs(db, "") == []
    # FTS syntax in user text is quoted, never interpreted: "OR" is filler, `"` is dropped, so
    # this is the plain "invoice" search — not an operator.
    assert [h["run_id"] for h in trace.search_runs(db, 'invoice OR "')] == [2]
    # The natural-language shapes the tool is advertised for find the run.
    assert [h["run_id"] for h in trace.search_runs(db, "the invoice export we did last week")] == [2]
    assert trace.search_runs(db, "what did we do last week") == []  # all filler: no false hit


def test_search_runs_like_fallback_without_fts(tmp_path, monkeypatch):
    db = tmp_path / "db.sqlite"
    _seed(db)
    monkeypatch.setattr(trace, "ensure_fts", lambda _conn: False)
    hits = trace.search_runs(db, "Invoice Export")
    assert [h["run_id"] for h in hits] == [2]


def test_fts_index_backfills_existing_rows(tmp_path):
    import sqlite3

    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(trace._SCHEMA)
    conn.execute("INSERT INTO runs (thread_id, query, started_at, status, response) VALUES "
                 "('t', 'old run about kittens', '2026-01-01', 'ok', 'cute')")
    conn.commit()
    conn.close()
    assert [h["run_id"] for h in trace.search_runs(db, "kittens")] == [1]


def test_current_run_id_is_set_during_a_run_and_cleared_after(tmp_path):
    t = trace.Tracer(str(tmp_path / "db.sqlite"))
    assert trace.current_run_id() is None
    rid = t.start_run("thread", "q")
    assert trace.current_run_id() == rid
    t.end_run(rid, "ok", "a")
    assert trace.current_run_id() is None
    t.conn.close()


def test_recall_runs_tool_reads_the_configured_db(isolated_paths):
    from config import get_config
    from tools.knowledge import recall_runs

    _seed(get_config().path("db_sqlite"))
    out = recall_runs.invoke({"query": "reports q3", "limit": 3})
    assert "run #1" in out and "request: rename the Q3 reports" in out and "q3_ prefix" in out
    assert recall_runs.invoke({"query": "zebra"}) == "No past runs match those words."


def test_trace_search_command(tmp_path, capsys):
    from commands import dispatch

    db = tmp_path / "db.sqlite"
    _seed(db)
    ctx = CommandContext(state={}, make_initial_state=dict, db_path=str(db))
    dispatch("/trace search invoice", ctx)
    dispatch("/trace search", ctx)
    dispatch("/trace search zebra", ctx)
    out = capsys.readouterr().out
    assert "#2" in out and "summarize the invoice export" in out and "date filter" in out
    assert "usage: /trace search" in out and "no recorded run matches 'zebra'" in out


def test_like_fallback_escapes_wildcards(tmp_path, monkeypatch):
    db = tmp_path / "db.sqlite"
    _seed(db)
    monkeypatch.setattr(trace, "ensure_fts", lambda _conn: False)
    # `_` is a LIKE wildcard: unescaped, "invoice_export" would match "invoice export" (run 2).
    assert trace.search_runs(db, "invoice_export") == []
    assert [h["run_id"] for h in trace.search_runs(db, "q3_")] == [1]


def test_fts_triggers_are_dropped_when_the_module_is_missing(tmp_path):
    """A DB indexed by an fts5-capable build, opened by one without it: the leftover triggers
    would fail every INSERT on runs ("no such module"). The tracer must keep recording."""
    import sqlite3

    db = tmp_path / "db.sqlite"
    _seed(db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'runs_fts_a_'").fetchone()[0] == 3
    # Simulate the missing module: point the virtual table at a module that does not exist.
    conn.execute("PRAGMA writable_schema=1")
    conn.execute("UPDATE sqlite_master SET sql=replace(sql, 'USING fts5', 'USING no_such_module') "
                 "WHERE name='runs_fts'")
    conn.commit()
    conn.close()
    t = trace.Tracer(str(db))  # ensure_fts fails -> triggers dropped
    rid = t.start_run("thread", "does recording still work")
    assert rid > 0 and not t._broken
    t.end_run(rid, "ok", "yes")
    assert t.conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'runs_fts_a_'").fetchone()[0] == 0
    assert [h["run_id"] for h in trace.search_runs(db, "recording")] == [4]  # LIKE fallback
    t.conn.close()
