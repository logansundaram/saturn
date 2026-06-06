from __future__ import annotations

from typing import Optional

from commands._framework import command, _print


def _to_int(s) -> Optional[int]:
    """Parse a run selector token to an int, tolerating a leading '#'. None if not a number."""
    try:
        return int(str(s).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


@command(
    "trace",
    "Expanded drill-down of a recorded run (default: the last run).",
    usage="/trace [#id | -r id] | -l [n]",
    details="""
Expands one recorded run from the trace database (database/db.sqlite) into the full replay the
live trace abbreviates: the query, every node with its step time and metrics, the plan as it
advanced, the agent's reasoning and tool-call decisions at each step (the execution detail the
live trace omits), each tool call WITH its output (the live trace hides that too), and — last and
de-emphasized — the recorded final answer. This is the execution log, not a reprint of the
response.

With no argument it expands the MOST RECENT run. Select another run by id, or list runs to find
one:

  /trace            expand the last run
  /trace #7         expand run 7   (also: -r 7, --run 7, or just: /trace 7)
  /trace -l         list recent runs at a glance — the run ids live here
  /trace -l 20      list the last 20

Every turn is one run. Unlike /history (in-memory, cleared by /reset) this is the durable record
that survives restarts; /calls is the cross-run tool-call/output view.
""",
)
def _trace(ctx, args):
    import sqlite3
    from tui import ui

    list_mode = False
    run_id: Optional[int] = None
    bare: Optional[int] = None
    it = iter(args)
    for a in it:
        low = a.lower()
        if low in ("-l", "--list", "list"):
            list_mode = True
        elif low in ("-r", "--run"):
            rid = _to_int(next(it, ""))
            if rid is not None:
                run_id = rid
        elif a.startswith("#"):
            rid = _to_int(a)
            if rid is not None:
                run_id = rid
        elif a.lstrip("+-").isdigit():
            bare = int(a)
        else:
            _print(f"  ignoring unrecognized argument: {a!r}")

    n: Optional[int] = bare if list_mode else None
    if bare is not None and not list_mode and run_id is None:
        run_id = bare

    conn = sqlite3.connect(ctx.db_path)
    try:
        if list_mode:
            rows = conn.execute(
                "SELECT run_id, started_at, status, query, "
                "(SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS n_events "
                "FROM runs r ORDER BY run_id DESC LIMIT ?",
                (max(1, n or 10),),
            ).fetchall()
            if not rows:
                _print("  (no runs recorded yet)")
                return
            _print(f"  last {len(rows)} run(s) — newest first  (/trace #<id> to expand one):")
            for rid, started_at, status, query, n_events in rows:
                when = (started_at or "")[:19].replace("T", " ")
                q = " ".join(str(query or "").split())
                if len(q) > 56:
                    q = q[:55] + "…"
                _print(f"    #{rid:<4} {when}  {str(status):<7} {n_events:>2}ev  {q}")
            return

        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
            run_id = row[0] if row else None
            if run_id is None:
                _print("  (no runs recorded yet)")
                return
        run = conn.execute(
            "SELECT run_id, query, started_at, ended_at, status, response FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            _print(f"  no run #{run_id} — try /trace -l to list recorded runs.")
            return
        events = conn.execute(
            "SELECT seq, ts, node, summary, data FROM events WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    ui.show_run(run, events)
