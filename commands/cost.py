from commands._framework import command, _print


def _fmt_secs(s: float) -> str:
    """Compact wall-clock: 8.4s, 1m23s, 2h05m."""
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_count(n: int) -> str:
    """Compact integer: 980, 1.8k, 2.34M."""
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


@command(
    "cost",
    "Session totals: turns, time, tokens, tools.",
    aliases=("session", "usage"),
    usage="/cost [--all]",
    details="""
Cumulative accounting for the current session: how many turns you've run, total + average wall
time, the slowest turn, total agent iterations and tool calls, and the prompt tokens processed
(with the peak context fill reached). The per-turn receipt under each answer shows one turn; this
is the running total across the whole session.

The numbers are aggregated from the trace database (database/db.sqlite), so they're exact and
survive /reset (which clears the in-memory conversation, not the trace). By default the scope is
THIS session (runs started since the process launched); pass --all to total every recorded run
across all past sessions.

"prompt tok processed" sums the prompt tokens ingested across every LLM call this session (agent
passes + synthesis) — the bulk of local-model cost; output tokens aren't separately recorded. Only
Ollama models report token counts, so this reads 0 on providers that don't.

Examples:
  /cost        this session's totals
  /cost --all  every recorded run, all sessions
""",
)
def _cost(ctx, args):
    import json
    import sqlite3
    from datetime import datetime

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
            _print("  (no runs recorded yet this session)" if scope
                   else "  (no runs recorded yet)")
            return
        ev_rows = conn.execute(
            "SELECT run_id, data FROM events WHERE run_id >= ?", (runs[0][0],)
        ).fetchall()
    finally:
        conn.close()

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
