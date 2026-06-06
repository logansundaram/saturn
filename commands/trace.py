from __future__ import annotations

import json
import sqlite3
from typing import Optional

from commands._framework import command, _print


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_count(n: int) -> str:
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def _calls(ctx, args):
    _MAX_CALL_OUTPUT = 600
    n = 10
    if args:
        try:
            n = max(1, int(args[0]))
        except ValueError:
            _print(f"  ignoring non-numeric count: {args[0]!r}")

    conn = sqlite3.connect(ctx.db_path)
    try:
        rows = conn.execute(
            "SELECT run_id, data FROM events WHERE node = 'tools' ORDER BY id DESC LIMIT ?",
            (n * 5,),
        ).fetchall()
    finally:
        conn.close()

    calls: list[tuple[int, dict, str]] = []
    for run_id, data in rows:
        try:
            delta = json.loads(data or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        events = delta.get("tool_events") or []
        results = delta.get("tool_results") or []
        for i, ev in enumerate(events):
            full = results[i] if i < len(results) else ""
            calls.append((run_id, ev, full))
        if len(calls) >= n:
            break

    if not calls:
        _print("  (no tool calls recorded yet)")
        return

    calls = calls[:n]
    _print(f"  last {len(calls)} tool call(s) — newest first:")
    for run_id, ev, full in calls:
        glyph = "✓" if ev.get("ok", True) else "⨯"
        dur = ev.get("dur")
        dur_s = f"{dur:.2f}s" if isinstance(dur, (int, float)) else "  -  "
        call_repr, _, observation = (full or "").partition(" -> ")
        if not call_repr:
            call_repr = ev.get("name", "?")
            observation = ev.get("result", "")
        _print(f"    #{run_id:<4} {glyph} {dur_s:>6}  {call_repr}")
        out = " ".join(str(observation).split())
        if len(out) > _MAX_CALL_OUTPUT:
            out = out[: _MAX_CALL_OUTPUT - 1] + "…"
        _print(f"             -> {out}" if out else "             -> (no output)")


def _cost(ctx, args):
    all_time = any(a.lower() in ("--all", "-a", "all") for a in args)
    scope = "" if all_time else (ctx.session_started_at or "")

    conn = sqlite3.connect(ctx.db_path)
    try:
        if scope:
            runs = conn.execute(
                "SELECT run_id, query, started_at, ended_at, status FROM runs "
                "WHERE started_at >= ? ORDER BY run_id",
                (scope,),
            ).fetchall()
        else:
            runs = conn.execute(
                "SELECT run_id, query, started_at, ended_at, status FROM runs ORDER BY run_id"
            ).fetchall()
        if not runs:
            _print("  (no runs recorded yet this session)" if scope else "  (no runs recorded yet)")
            return
        ev_rows = conn.execute(
            "SELECT run_id, data FROM events WHERE run_id >= ?", (runs[0][0],)
        ).fetchall()
    finally:
        conn.close()

    from datetime import datetime

    def _parse(ts):
        try:
            return datetime.fromisoformat(ts) if ts else None
        except (TypeError, ValueError):
            return None

    total_wall = 0.0
    timed = 0
    slowest = (0.0, "")
    status_mix = {"ok": 0, "error": 0, "interrupted": 0, "other": 0}
    for _rid, query, started_at, ended_at, status in runs:
        s, e = _parse(started_at), _parse(ended_at)
        if s and e:
            secs = (e - s).total_seconds()
            total_wall += secs
            timed += 1
            if secs > slowest[0]:
                slowest = (secs, query or "")
        status_mix[status if status in status_mix else "other"] += 1

    total_tools = 0
    total_prompt_tokens = 0
    peak_ctx = 0
    max_iter = {}
    for run_id, data in ev_rows:
        try:
            delta = json.loads(data or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        total_tools += len(delta.get("tools_called") or [])
        ct = delta.get("context_tokens") or 0
        if ct:
            total_prompt_tokens += ct
            peak_ctx = max(peak_ctx, ct)
        if "iteration" in delta:
            max_iter[run_id] = max(max_iter.get(run_id, 0), delta["iteration"] or 0)
    total_iters = sum(max_iter.values())

    turns = len(runs)
    avg = total_wall / timed if timed else 0.0
    mix = " · ".join(f"{v} {k}" for k, v in status_mix.items() if v)

    _print("")
    _print(
        f"  session totals — {turns} turn{'' if turns == 1 else 's'}"
        + ("" if scope else "  (all recorded runs)")
    )
    _print(f"    turns        {turns}" + (f"   ({mix})" if mix else ""))
    if timed:
        _print(f"    wall time    {_fmt_secs(total_wall)}   (avg {_fmt_secs(avg)}/turn)")
    _print(f"    iterations   {total_iters}")
    _print(f"    tool calls   {total_tools}")
    _print(
        f"    prompt tok   {_fmt_count(total_prompt_tokens)} processed"
        + (f"   (peak ctx {_fmt_count(peak_ctx)})" if peak_ctx else "")
    )
    if slowest[0]:
        q = " ".join(str(slowest[1]).split())
        if len(q) > 48:
            q = q[:47] + "…"
        _print(f"    slowest      {_fmt_secs(slowest[0])}  \"{q}\"")
    _print("")


def _to_int(s) -> Optional[int]:
    """Parse a run selector token to an int, tolerating a leading '#'. None if not a number."""
    try:
        return int(str(s).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _verbosity(ctx, args):
    from tui import ui

    arg = args[0].lower() if args else ""
    if arg in ("off", "quiet", "compact", "false", "no", "0"):
        ctx.show_ui = False
    elif arg in ("on", "normal", "true", "yes", "1"):
        ctx.show_ui = True
        ui.set_verbosity("normal")
    elif arg in ("full", "verbose", "detailed", "all", "debug"):
        ctx.show_ui = True
        ui.set_verbosity("verbose")
    else:
        _print(
            f"  usage: /trace off|on|full   (trace {'on' if ctx.show_ui else 'off'}, "
            f"detail {ui.verbosity()})"
        )
        return

    if not ctx.show_ui:
        _print("  live trace off — only the final response prints.")
    else:
        level = ui.verbosity()
        detail = (
            "every node + full timings" if level == "verbose"
            else "plan · agent · tools · synthesize (plumbing folded)"
        )
        _print(f"  live trace on — {level}: {detail}.")


def _state(ctx, args):
    s = ctx.state
    _print("  agent state:")
    _print(f"    messages      : {len(s.get('messages', []))}")
    _print(f"    current_query : {s.get('current_query', '')!r}")
    _print(f"    iteration     : {s.get('iteration', 0)}")
    _print(f"    plan steps    : {len(s.get('plan', []))}")
    _print(f"    tools_called  : {s.get('tools_called', [])}")
    _print(f"    docs_retrieved: {len(s.get('documents_retrieved', []))}")
    if "--full" in args:
        _print("    ---- raw ----")
        _print(f"    {s}")


@command(
    "trace",
    "Observability hub: drill-down of recorded runs + live trace control.",
    usage="/trace [#id | -r id | -l [n] | on | off | full]",
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

Every turn is one run. This is the durable record that survives restarts.
Subviews:

  /trace invoke [#id]  the LLM calls of a run: each model call's INPUT messages + OUTPUT, with
                       timing + token counts. Defaults to the most recent run with LLM calls; add
                       --full to show whole messages, -l to list runs that have them.
  /trace calls [n]     recent tool calls + their outputs
  /trace cost [--all]  session totals: turns, time, tokens
  /trace state         dump the live AgentState (message count, plan steps, tools called, etc.)
                       pass --full to also dump the raw state dict

Live trace verbosity (controls what scrolls during a turn; recording is always on):

  /trace off    only the final response prints — runs quietly
  /trace on     normal: plan · agent · tools · synthesize (plumbing nodes folded)  [default]
  /trace full   verbose: every node line, including folded plumbing + full timings
""",
)
def _trace(ctx, args):
    from tui import ui

    if args and args[0].lower() in ("invoke", "--invoke", "llm", "--llm", "model", "models"):
        return _show_llm_calls(ctx, args[1:])
    if args and args[0].lower() in ("calls", "io"):
        return _calls(ctx, args[1:])
    if args and args[0].lower() in ("cost", "session", "usage"):
        return _cost(ctx, args[1:])
    if args and args[0].lower() in ("state", "--state"):
        return _state(ctx, args[1:])
    if args and args[0].lower() in ("on", "off", "full", "normal", "quiet", "verbose",
                                     "detailed", "all", "debug", "compact",
                                     "true", "false", "yes", "no", "0", "1"):
        return _verbosity(ctx, args)

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


def _show_llm_calls(ctx, args):
    """`/trace invoke` — replay one run's LLM calls (input messages + output). Default: the most
    recent run that has any; `#id`/`-r id`/bare int to pick one; `-l` to list runs with LLM calls;
    `--full` to show whole messages instead of the clipped preview."""
    from tui import ui

    full = False
    list_mode = False
    run_id: Optional[int] = None
    it = iter(args)
    for a in it:
        low = a.lower()
        if low in ("--full", "-f", "full"):
            full = True
        elif low in ("-l", "--list", "list"):
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
            run_id = int(a)
        else:
            _print(f"  ignoring unrecognized argument: {a!r}")

    conn = sqlite3.connect(ctx.db_path)
    try:
        # The llm_calls table is created by the Tracer at startup; guard anyway for a stale DB.
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'"
        ).fetchone()
        if not has_table:
            _print("  (no LLM calls recorded yet — run a query first)")
            return

        if list_mode:
            rows = conn.execute(
                "SELECT c.run_id, COUNT(*) AS n, COALESCE(SUM(c.dur), 0), r.query "
                "FROM llm_calls c LEFT JOIN runs r ON r.run_id = c.run_id "
                "GROUP BY c.run_id ORDER BY c.run_id DESC LIMIT ?",
                (max(1, run_id or 10),),
            ).fetchall()
            if not rows:
                _print("  (no LLM calls recorded yet)")
                return
            _print(f"  runs with LLM calls — newest first  (/trace invoke #<id> to expand one):")
            for rid, n, dur, query in rows:
                q = " ".join(str(query or "").split())
                if len(q) > 50:
                    q = q[:49] + "…"
                _print(f"    #{rid:<4} {n:>2} call(s)  {float(dur or 0):>6.1f}s  {q}")
            return

        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM llm_calls").fetchone()
            run_id = row[0] if row else None
            if run_id is None:
                _print("  (no LLM calls recorded yet — run a query first)")
                return

        run = conn.execute(
            "SELECT run_id, query, started_at, ended_at, status, response FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            _print(f"  no run #{run_id} — try /trace invoke -l to list runs with LLM calls.")
            return
        calls = conn.execute(
            "SELECT seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status "
            "FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    ui.show_llm_calls(run, calls, full=full)
