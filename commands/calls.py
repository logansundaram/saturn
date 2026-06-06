from commands._framework import command, _print

_MAX_CALL_OUTPUT = 600


@command(
    "calls",
    "Show recent tool calls and their outputs.",
    aliases=("io",),
    usage="/calls [n]",
    details="""
Shows the most recent tool calls the agent made — each with its arguments, the result the tool
returned, how long it took, and whether it succeeded (✓) or errored (⨯). Defaults to the last 10.

The data comes from the trace database (database/db.sqlite), so it survives /reset and restarts
and spans every run — unlike /history (in-memory conversation only). This is the I/O complement
to /tools (which lists the tools that *exist*) and /trace (which lists runs at a glance).

Long outputs are truncated for readability; the full observation rides the message history the
model sees. For the per-run event breakdown, see /trace.

Examples:
  /calls       last 10 calls
  /calls 25    last 25 calls
""",
)
def _calls(ctx, args):
    import json
    import sqlite3

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
