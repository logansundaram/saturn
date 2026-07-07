from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from commands._framework import command, _print
from stores.trace import decode_json, parse_ts
from textutil import CALL_RESULT_SEP, clip as _clip, fmt_args


@contextmanager
def _connect(db_path):
    """The trace DB connection for one read, closed on exit."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


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

    with _connect(ctx.db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, data FROM events WHERE node = 'tools' ORDER BY id DESC LIMIT ?",
            (n * 5,),
        ).fetchall()

    calls: list[tuple[int, dict, str]] = []
    for run_id, data in rows:
        delta = decode_json(data, {})
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
        # CALL_RESULT_SEP — the one constant nodes/tools.py builds these entries with; a
        # hand-typed " -> " literal here silently stops splitting if the separator ever changes.
        call_repr, _, observation = (full or "").partition(CALL_RESULT_SEP)
        if not call_repr:
            call_repr = ev.get("name", "?")
            observation = ev.get("result", "")
        _print(f"    #{run_id:<4} {glyph} {dur_s:>6}  {call_repr}")
        out = _clip(observation, _MAX_CALL_OUTPUT)
        _print(f"             -> {out}" if out else "             -> (no output)")


def _cost(ctx, args):
    all_time = any(a.lower() in ("--all", "-a", "all") for a in args)
    scope = "" if all_time else (ctx.session_started_at or "")

    with _connect(ctx.db_path) as conn:
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

    total_wall = 0.0
    timed = 0
    slowest = (0.0, "")
    status_mix = {"ok": 0, "error": 0, "interrupted": 0, "other": 0}
    for _rid, query, started_at, ended_at, status in runs:
        s, e = parse_ts(started_at), parse_ts(ended_at)
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
        delta = decode_json(data, {})
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
        _print(f"    slowest      {_fmt_secs(slowest[0])}  \"{_clip(slowest[1], 48)}\"")
    _print("")


def _to_int(s) -> Optional[int]:
    """Parse a run selector token to an int, tolerating a leading '#'. None if not a number."""
    try:
        return int(str(s).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _parse_run_selector(args, *, consume=None):
    """THE run-selector grammar, shared by every /trace subview (was five hand-kept copies that
    had already drifted: _why/_answer didn't take -r, and invoke reused run_id as the list
    count). Recognized everywhere:

        -r/--run <id> · #<id> · bare integer   -> run_id  (bare digits are RUN IDS — except in
                                                  list mode, where a bare digit is the COUNT)
        -l/--list/list/ls                      -> list_mode

    `consume(low, arg, it)` is an optional hook for command-specific tokens (--md, -o <path>,
    --full); return True when the hook handled the token (it may pull a value from `it`).
    Anything unrecognized prints the shared "ignoring" note. Returns (run_id, count, list_mode);
    `count` is only ever set in list mode."""
    run_id: Optional[int] = None
    bare: Optional[int] = None
    list_mode = False
    it = iter(args)
    for a in it:
        low = a.lower()
        if consume is not None and consume(low, a, it):
            continue
        if low in ("-l", "--list", "list", "ls"):
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
    if bare is not None and not list_mode and run_id is None:
        run_id = bare
    return run_id, (bare if list_mode else None), list_mode


def _load_run(conn, run_id, *,
              columns="run_id, query, started_at, ended_at, status, response",
              latest_from="runs",
              empty_msg="  (no runs recorded yet)",
              hint="/trace -l"):
    """THE latest-run fallback + row loader (was five hand-kept copies of MAX(run_id) + the
    per-id SELECT + the two error prints). `latest_from` lets /trace invoke default to the
    newest run that HAS llm_calls. Returns (run_id, row); row is None (after printing why)
    when there is nothing to show. `columns`/`latest_from` are code-controlled literals, never
    user input."""
    if run_id is None:
        row = conn.execute(f"SELECT MAX(run_id) FROM {latest_from}").fetchone()
        run_id = row[0] if row else None
        if run_id is None:
            _print(empty_msg)
            return None, None
    run = conn.execute(
        f"SELECT {columns} FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not run:
        _print(f"  no run #{run_id} — try {hint} to list recorded runs.")
        return run_id, None
    return run_id, run


# --- /trace export ------------------------------------------------------------------------------
# One run's complete record (run + events + LLM calls) written to a self-contained file. JSON is
# the record format /trace replay renders offline; --md renders a human-readable report instead.
# (The sha256 integrity digest + the verify flows — /trace verify, saturn verify — were CUT
# 2026-07-03: a digest stored inside the file it protects verifies after any edit that recomputes
# it, so it only ever caught accidental corruption; real verification returns in Phase 3 with
# signing. Legacy exports still carry `integrity`/`signature` blocks — replay ignores them.)

# The versioned artifact-format marker embedded in every export (layout versioning).
ARTIFACT_FORMAT = "saturn-artifact/1"


def _saturn_version() -> str:
    """The running Saturn version for stamping exports — read off the already-loaded agent module
    (importing agent.py here would be heavy and double-imports under `python agent.py`)."""
    for name in ("__main__", "agent"):
        v = getattr(sys.modules.get(name), "__version__", None)
        if v:
            return str(v)
    return "unknown"


def _export_payload(run, events, calls) -> dict:
    run_id, query, started_at, ended_at, status, response = run
    payload = {
        "saturn_trace_export": 1,
        "format": ARTIFACT_FORMAT,
        "saturn_version": _saturn_version(),
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": {
            "run_id": run_id,
            "query": query,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
            "response": response,
        },
        "events": [
            {
                "seq": seq,
                "ts": ts,
                "node": node,
                "summary": summary,
                # keep undecodable deltas verbatim — an audit record drops nothing
                "data": decode_json(data, None) if data else None,
            }
            for seq, ts, node, summary, data in events
        ],
        "llm_calls": [
            {
                "seq": seq,
                "ts": ts,
                "node": node,
                "model": model,
                "dur": dur,
                "prompt_tokens": p_tok,
                "output_tokens": o_tok,
                "input": decode_json(inp, None) if inp else None,
                "output": decode_json(out, None) if out else None,
                "status": call_status,
            }
            for seq, ts, node, model, dur, p_tok, o_tok, inp, out, call_status in calls
        ],
    }
    return payload


def _export_markdown(payload: dict) -> str:
    """The human-readable rendering of an export payload — a report, not the audit format."""
    run = payload["run"]
    lines = [
        f"# Saturn run record — run #{run['run_id']}",
        "",
        f"- **query:** {run['query'] or '(none)'}",
        f"- **started:** {run['started_at'] or '?'}   **ended:** {run['ended_at'] or '?'}",
        f"- **status:** {run['status'] or '?'}",
        f"- **exported:** {payload['exported_at']}  (saturn {payload['saturn_version']})",
        "",
        "## Timeline",
        "",
    ]
    for ev in payload["events"]:
        ts = (ev["ts"] or "")[11:19]
        lines.append(f"- `{ts}` **{ev['node']}** — {ev['summary'] or ''}")
        delta = ev["data"] if isinstance(ev["data"], dict) else {}
        for result in delta.get("tool_results") or []:
            lines.append("")
            lines.append("  ```")
            lines.extend(f"  {ln}" for ln in str(result).splitlines())
            lines.append("  ```")
    calls = payload["llm_calls"]
    if calls:
        lines += ["", f"## LLM calls ({len(calls)})", ""]
        for c in calls:
            dur = f"{c['dur']:.1f}s" if isinstance(c["dur"], (int, float)) else "?"
            toks = f"{c['prompt_tokens'] or 0}→{c['output_tokens'] or 0} tok"
            lines.append(f"- `{c['node']}` {c['model'] or '?'} — {dur}, {toks}, {c['status']}")
    lines += ["", "## Final response", "", run["response"] or "(none)", ""]
    return "\n".join(lines)


def export_run(
    db_path,
    run_id: Optional[int] = None,
    dest: Optional[Path] = None,
    fmt_md: bool = False,
) -> "tuple[Path, dict]":
    """THE one export-payload builder + writer — shared by the /trace export handler and the
    headless `saturn -p ... --export FILE` flag, so the two surfaces can never drift onto
    different payloads. `run_id=None` exports the latest run; `dest=None` writes the default
    logging/exports/run_<id>.<ext>. Returns (path written, payload as written). Raises
    LookupError (no such run) / OSError (write failed) — each caller renders those its own
    way (REPL note vs. stderr + exit code)."""
    with _connect(db_path) as conn:
        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
            run_id = row[0] if row else None
            if run_id is None:
                raise LookupError("(no runs recorded yet)")
        run = conn.execute(
            "SELECT run_id, query, started_at, ended_at, status, response FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            raise LookupError(f"no run #{run_id} — try /trace -l to list recorded runs.")
        events = conn.execute(
            "SELECT seq, ts, node, summary, data FROM events WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()
        has_calls = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'"
        ).fetchone()
        calls = conn.execute(
            "SELECT seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status "
            "FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall() if has_calls else []

    payload = _export_payload(run, events, calls)

    from config import get_config

    if dest is None:
        ext = "md" if fmt_md else "json"
        dest = get_config().path("exports") / f"run_{run_id}.{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt_md:
        dest.write_text(_export_markdown(payload), encoding="utf-8")
    else:
        dest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return dest, payload


def _export(ctx, args):
    fmt_md = False
    out_path: Optional[str] = None
    bad_out = False

    def consume(low, a, it):
        nonlocal fmt_md, out_path, bad_out
        if low in ("--md", "-m", "md", "markdown"):
            fmt_md = True
            return True
        if low in ("-o", "--out", "--output"):
            out_path = next(it, None)
            # A dangling -o must not silently write the default, and a flag-shaped "path"
            # (-o --md) is a swallowed flag, not a destination — refuse both before any
            # DB/file work.
            if out_path is None or out_path.startswith("-"):
                bad_out = True
            return True
        return False

    run_id, _count, _list = _parse_run_selector(args, consume=consume)
    if bad_out:
        _print("  usage: /trace export [#id] [--md] [-o <path>] — -o needs a path; "
               "nothing written")
        return

    try:
        dest, payload = export_run(
            ctx.db_path,
            run_id,
            dest=Path(out_path).expanduser() if out_path else None,
            fmt_md=fmt_md,
        )
    except LookupError as e:
        _print(f"  {e}")
        return
    except OSError as e:
        _print(f"  could not write export: {e}")
        return

    run_id = payload["run"]["run_id"]
    _print(f"  run #{run_id} exported -> {dest}")
    _print(f"    {len(payload['events'])} event(s), {len(payload['llm_calls'])} LLM call(s)")
    if not fmt_md:
        _print("    (replayable offline: /trace replay <file>, or saturn --replay <file>)")


# --- /trace replay · saturn --replay -----------------------------------------------------------
# Render an exported run record OFFLINE, through the exact same drill-down view /trace uses on the
# live DB — what makes an export not just inspectable but SHAREABLE: attach a .json to a bug report
# and the recipient replays the full rail (plan, reasoning, tool I/O, answer) with no database.

def export_rows(payload: dict):
    """Rebuild (run_tuple, event_rows) from an export payload, in the shapes ui.show_run expects
    (event `data` re-encoded to JSON — the export stores it decoded). Pure, for tests."""
    run = payload.get("run") or {}
    run_tuple = (
        run.get("run_id"), run.get("query"), run.get("started_at"),
        run.get("ended_at"), run.get("status"), run.get("response"),
    )
    rows = [
        (
            ev.get("seq"), ev.get("ts"), ev.get("node"), ev.get("summary"),
            json.dumps(ev.get("data")) if ev.get("data") is not None else None,
        )
        for ev in (payload.get("events") or [])
    ]
    return run_tuple, rows


def render_export(path_str: str) -> bool:
    """Load an exported run record and replay it via ui.show_run. Used by
    `/trace replay <file>` and the `saturn --replay <file>` CLI flag. Diagnostics (unreadable
    file, not an export) go to STDERR so a piped stdout stays the
    rendered run; returns False on a file that can't be rendered (the CLI exits non-zero on it)."""
    from tui import ui

    path = Path(path_str.strip('"')).expanduser()
    try:
        # utf-8-sig: a BOM (PowerShell 5.1 redirection writes one) must not fail the read.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  could not read {path}: {e}", file=sys.stderr)
        return False
    if not isinstance(payload, dict) or payload.get("saturn_trace_export") != 1:
        print(f"  {path.name} is not a Saturn trace export (see /trace export).",
              file=sys.stderr)
        return False

    run_tuple, rows = export_rows(payload)
    _print(f"  replaying exported record: {path.name}  "
           f"(saturn {payload.get('saturn_version', '?')}, exported {payload.get('exported_at', '?')})")
    _print("")
    ui.show_run(run_tuple, rows)
    return True


def _replay(ctx, args):
    if not args:
        _print("  usage: /trace replay <exported .json file>")
        return
    render_export(" ".join(args))


def _verbosity(ctx, args):
    from tui import ui

    arg = args[0].lower() if args else ""
    if arg in ("off", "quiet", "compact", "false", "no"):
        ctx.show_ui = False
    elif arg in ("on", "normal", "true", "yes"):
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
            else "plan · execute · tools · synthesize (plumbing folded)"
        )
        _print(f"  live trace on — {level}: {detail}.")


# --- /trace why — decision provenance ----------------------------------------------------------
# /trace shows WHAT happened; this subview reconstructs WHY: the causal chain from the recorded
# plan, per-step agent reasoning + chosen tool calls, the evidence relied on, the groundedness
# verdict, and the cited sources. (Folded in from the old standalone /why, June 2026.)

def _why(ctx, args):
    from tui import ui

    run_id, _count, _list = _parse_run_selector(args)

    with _connect(ctx.db_path) as conn:
        run_id, run = _load_run(
            conn, run_id, empty_msg="  (no runs recorded yet — ask something first)"
        )
        if run is None:
            return
        events = conn.execute(
            "SELECT seq, node, summary, data FROM events WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()
        has_calls = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'"
        ).fetchone()
        calls = conn.execute(
            "SELECT seq, node, output FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall() if has_calls else []

    _render_why(ui, run, events, calls)


def _final_plan(events) -> list:
    """The plan as it stood at the end of the run — the last event delta that carried one."""
    plan = []
    for _seq, _node, _summary, data in events:
        delta = decode_json(data, {})
        if delta.get("plan"):
            plan = delta["plan"]
    return plan


def _collect_tools(events):
    """Flatten tools_called + tool_results across the run, in order."""
    results = []
    for _seq, node, _summary, data in events:
        if node != "tools":
            continue
        delta = decode_json(data, {})
        for r in delta.get("tool_results") or []:
            results.append(str(r))
        for d in delta.get("documents_retrieved") or []:
            results.append("knowledge base: " + _clip(d, 80))
    return results


def _render_why(ui, run, events, calls):
    run_id, query, _started, _ended, status, response = run
    _GLYPH = {"pending": "○", "active": "▸", "done": "✓", "skipped": "—",
              "blocked": "⊘", "error": "✗", "cancelled": "−"}

    ui.section(f"why · run #{run_id}", f"status: {status or '?'}")

    _print("  the request")
    _print(f"    {_clip(query, 120) or '(none)'}")
    _print("")

    # What it set out to do — the plan.
    plan = _final_plan(events)
    if plan:
        _print("  what it set out to do")
        for s in plan:
            glyph = _GLYPH.get(s.get("status"), "○")
            tool = f"  [{s['intended_tool']}]" if s.get("intended_tool") else ""
            _print(f"    {glyph} {s.get('step_id')}. {s.get('label')}{tool}")
        _print("")

    # How it reasoned — the execute steps + the rectify verdicts, from the recorded LLM I/O.
    step = 0
    verdicts: list[str] = []
    for _seq, node, output in calls:
        out = decode_json(output, {})
        if node == "execute":
            step += 1
            content = _clip(out.get("content", ""), 240)
            tcs = out.get("tool_calls") or []
            if step == 1:
                _print("  how it reasoned")
            if content:
                _print(f"    step {step}: {content}")
            if tcs:
                names = ", ".join(
                    f"{c.get('name')}({_fmt_call_args(c.get('args'))})" for c in tcs
                )
                _print(f"      → chose to call: {names}")
            elif not content:
                _print(f"    step {step}: (finished — no further action)")
        elif node in ("rectify", "replan"):
            content = str(out.get("content", "") or "").strip()
            if content:
                verdicts.append(f"{node}: {content}")
    if step:
        _print("")

    # What it relied on — the evidence the answer was built from.
    evidence = _collect_tools(events)
    if evidence:
        _print("  what it relied on")
        for e in evidence[:12]:
            _print(f"    • {_clip(e, 110)}")
        if len(evidence) > 12:
            _print(f"    … and {len(evidence) - 12} more (see /trace #%s)" % run_id)
        _print("")
    else:
        _print("  what it relied on")
        _print("    (no tools ran — answered from the model's own knowledge + context)")
        _print("")

    # Self-correction — ALWAYS printed: the negative case is information too (the Glass Box's
    # "rectified" row says the same thing, and the two must agree). Silence here used to read as
    # "maybe checked, maybe not" — a trust surface can't leave that ambiguous. Named for what the
    # state records (rectify verdicts); "verification" overpromised — nothing verifies the answer.
    _print("  self-correction")
    if verdicts:
        for v in verdicts[-3:]:
            _print(f"    {_clip(v, 160)}")
    else:
        _print("    rectify judge did not run — every step resolved mechanically.")
    _print("")

    # Provenance footer of the answer, if the synthesizer attached one (the [n] → source map).
    if response and "Sources:" in str(response):
        tail = str(response).split("Sources:", 1)[1].strip()
        if tail:
            _print("  cited sources (from the answer)")
            for line in tail.splitlines():
                if line.strip():
                    _print(f"    {line.strip()}")
            _print("")

    _print(f"  full step-by-step record: /trace #{run_id}   ·   model I/O: /trace invoke #{run_id}")


def _fmt_call_args(args) -> str:
    return fmt_args(args, 41) if isinstance(args, dict) else ""


# --- /trace answer — the Glass Box (answer-level provenance) ------------------------------------
# /trace why explains HOW the agent worked; this shows whether you can trust WHAT it told you: each
# cited source's origin (local vs network) and trust, and what left the machine. Bare reads the
# live last turn (exact egress); #id reconstructs from the recorded run (egress inferred from
# source tools).

def _answer(ctx, args):
    from tui import ui
    from trust import glassbox

    run_id, _count, _list = _parse_run_selector(args)

    state = ctx.state or {}
    # "Live" means THIS process ran the last turn: the per-turn accumulators (or current_query)
    # are populated. Messages alone do NOT count — a /resume-restored conversation carries only
    # messages, and rendering it against a fresh process's empty egress ledger would produce a
    # false 'local-only, 0 sources' box for an answer that may have been cloud-composed from web
    # sources last session. Such turns reconstruct from the recorded run below instead.
    live = bool(state.get("tool_results") or state.get("documents_retrieved")
                or state.get("tool_events") or state.get("current_query"))
    # Bare + a live last turn → the live Glass Box. glassbox.build_live owns the turn-mark guard
    # (the exact egress slice passes only when trustworthy — the same contract the native
    # post-answer provenance applies), so this path can't drift from it.
    if run_id is None and live:
        from tui.ui import _base
        gated = _base._status.get("gates", 0) if isinstance(_base._status, dict) else 0
        ui.show_glassbox(glassbox.build_live(state, gated=gated))
        return

    # Otherwise reconstruct from the recorded run (last, or the requested #id).
    with _connect(ctx.db_path) as conn:
        run_id, run = _load_run(
            conn, run_id, columns="run_id, query, response",
            empty_msg="  (no runs recorded yet — ask something first)",
        )
        if run is None:
            return
        events = conn.execute(
            "SELECT data FROM events WHERE run_id = ? ORDER BY seq, id", (run_id,)
        ).fetchall()

    _rid, query, response = run
    # Decode with an explicit failure sentinel: a fat delta is stored truncated at the trace's
    # _DATA_CAP and comes back undecodable — the Glass Box must know its inputs are incomplete
    # rather than assert 'sources: 0 · no untrusted content' over data it silently dropped.
    deltas = []
    truncated = False
    for (data,) in events:
        d = decode_json(data, None)
        if d is None and data:
            truncated = True
            continue
        if isinstance(d, dict):
            deltas.append(d)
    gb = glassbox.build_from_record(query, response, deltas, complete=not truncated)
    from tui import ui as _ui
    _ui.show_glassbox(gb)


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
    usage="/trace [#id | -l [n] | why | answer | invoke | calls | cost | state"
          " | export | replay | on|off|full]",
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

  /trace why [#id]     decision provenance: not WHAT happened but WHY — the plan it drafted, the
                       model's recorded reasoning + tool choice at each step, the evidence the
                       answer was built from, the rectify verdicts (plan revisions and why), and
                       the cited sources. Defaults to the last run.
  /trace answer [#id]  the Glass Box (also: /glass): answer-level provenance — each cited source's
                       origin (local vs network) and trust, and what left the machine. Bare = the
                       live last turn; #id reconstructs a recorded run. (/source <n> prints the
                       full text behind citation [n].)
  /trace invoke [#id]  the LLM calls of a run: each model call's INPUT messages + OUTPUT, with
                       timing + token counts. Defaults to the most recent run with LLM calls; add
                       --full to show whole messages, -l to list runs that have them.
  /trace calls [n]     recent tool calls + their outputs
  /trace cost [--all]  session totals: turns, time, tokens
  /trace state         dump the live AgentState (message count, plan steps, tools called, etc.)
                       pass --full to also dump the raw state dict
  /trace export [#id]  write a run's complete record (events + tool I/O + LLM calls) to a
                       self-contained file under logging/exports/ — JSON is the replayable
                       record; --md for a readable markdown report; -o <path> to choose the
                       destination. The record you can hand to someone else (also: saturn -p
                       "..." --export <file> writes the same artifact after a headless turn).
  /trace replay <file> replay an exported record OFFLINE through the same drill-down view —
                       no database needed. What makes an export shareable: anyone can replay
                       a run you hand them (also: saturn --replay <file> straight from the
                       shell).

Live trace verbosity (controls what scrolls during a turn; recording is always on):

  /trace off    only the final response prints — runs quietly
  /trace on     normal: plan · execute · tools · synthesize (plumbing nodes folded)  [default]
  /trace full   verbose: every node line, including folded plumbing + full timings
""",
)
def _trace(ctx, args):
    from tui import ui

    if args and args[0].lower() in ("why", "--why"):
        return _why(ctx, args[1:])
    if args and args[0].lower() in ("answer", "--answer", "glass", "glassbox"):
        return _answer(ctx, args[1:])
    if args and args[0].lower() in ("invoke", "--invoke", "llm", "--llm", "model", "models"):
        return _show_llm_calls(ctx, args[1:])
    if args and args[0].lower() in ("export", "--export"):
        return _export(ctx, args[1:])
    if args and args[0].lower() in ("replay", "--replay"):
        return _replay(ctx, args[1:])
    if args and args[0].lower() in ("calls", "io"):
        return _calls(ctx, args[1:])
    if args and args[0].lower() in ("cost", "session", "usage"):
        return _cost(ctx, args[1:])
    if args and args[0].lower() in ("state", "--state"):
        return _state(ctx, args[1:])
    # NOTE: no "0"/"1" verbosity aliases here — a bare digit is a RUN ID (`/trace 1` drills into
    # run #1, same as `/trace #1`); the digit aliases used to eat it and toggle verbosity instead.
    if args and args[0].lower() in ("on", "off", "full", "normal", "quiet", "verbose",
                                     "detailed", "all", "debug", "compact",
                                     "true", "false", "yes", "no"):
        return _verbosity(ctx, args)

    run_id, count, list_mode = _parse_run_selector(args)

    with _connect(ctx.db_path) as conn:
        if list_mode:
            rows = conn.execute(
                "SELECT run_id, started_at, status, query, "
                "(SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS n_events "
                "FROM runs r ORDER BY run_id DESC LIMIT ?",
                (max(1, count or 10),),
            ).fetchall()
            if not rows:
                _print("  (no runs recorded yet)")
                return
            _print(f"  last {len(rows)} run(s) — newest first  (/trace #<id> to expand one):")
            for rid, started_at, status, query, n_events in rows:
                when = (started_at or "")[:19].replace("T", " ")
                _print(f"    #{rid:<4} {when}  {str(status):<7} {n_events:>2}ev  {_clip(query, 56)}")
            return

        run_id, run = _load_run(conn, run_id)
        if run is None:
            return
        events = conn.execute(
            "SELECT seq, ts, node, summary, data FROM events WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()

    ui.show_run(run, events)


def _show_llm_calls(ctx, args):
    """`/trace invoke` — replay one run's LLM calls (input messages + output). Default: the most
    recent run that has any; `#id`/`-r id`/bare int to pick one; `-l` to list runs with LLM calls;
    `--full` to show whole messages instead of the clipped preview."""
    from tui import ui

    full = False

    def consume(low, a, it):
        nonlocal full
        if low in ("--full", "-f", "full"):
            full = True
            return True
        return False

    run_id, count, list_mode = _parse_run_selector(args, consume=consume)

    with _connect(ctx.db_path) as conn:
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
                (max(1, count or 10),),
            ).fetchall()
            if not rows:
                _print("  (no LLM calls recorded yet)")
                return
            _print("  runs with LLM calls — newest first  (/trace invoke #<id> to expand one):")
            for rid, n, dur, query in rows:
                _print(f"    #{rid:<4} {n:>2} call(s)  {float(dur or 0):>6.1f}s  {_clip(query, 50)}")
            return

        run_id, run = _load_run(
            conn, run_id, latest_from="llm_calls",
            empty_msg="  (no LLM calls recorded yet — run a query first)",
            hint="/trace invoke -l",
        )
        if run is None:
            return
        calls = conn.execute(
            "SELECT seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status "
            "FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()

    ui.show_llm_calls(run, calls, full=full)


# ── /glass — the brandable front door for /trace answer ──────────────────────────────────────
# A flagship surface (trust the ANSWER, the companion to /trace's trust-the-PROCESS), so it gets
# its own top-level name; all the logic lives in _answer above.


@command(
    "glass",
    "The Glass Box: answer-level provenance (origin · trust · what left the machine).",
    aliases=("glassbox",),
    usage="/glass [#id]",
    details="""
The answer-level companion to /trace. For one answer it shows, per cited source: where it came
from (local disk vs the network) and whether its origin is trusted. Plus the trust label: how many
sources, what left the machine this turn, whether rectify revised the plan mid-run, and how many
calls faced the approval gate.

  /glass         the live last turn (exact egress slice)
  /glass #7      reconstruct run #7 from the trace record

Identical to `/trace answer`. Scope: bare reads the live last turn (the accumulators reset when a
new turn starts); #id reconstructs any recorded run (egress is inferred from the source tools, the
one facet history can't carry exactly).

Companion: /source <n> prints the FULL text behind citation [n] — same numbering as the facets
here; /glass is whether to trust a source, /source is what it actually said.
""",
)
def _glass(ctx, args):
    return _answer(ctx, args)


# ── /source — the raw material behind a citation, registered with the other provenance views ──
# The citations footer maps each inline [n] to a one-line label; this shows the FULL tool
# result / retrieved passage behind that number, rebuilt with the same numbering the synthesizer
# saw (nodes.synthesize.build_sources over the turn's accumulators), so [3] here is exactly the
# [3] in the answer. Closes the provenance loop in one keystroke instead of a /trace drill-down.


def lookup_source(state: dict, n: int) -> "tuple[str, str] | None":
    """(label, full_text) for citation number `n` of the last turn, or None when out of range.
    Pure over the state accumulators so it's testable without a turn."""
    from nodes.synthesize import build_sources

    tool_results = (state or {}).get("tool_results") or []
    docs = (state or {}).get("documents_retrieved") or []
    numbered_tools, numbered_docs, sources = build_sources(tool_results, docs)
    entries = numbered_tools + numbered_docs
    if not (1 <= n <= len(entries)):
        return None
    label = sources[n - 1][1]
    # Strip the `[n] ` numbering prefix build_sources added for the prompt.
    text = entries[n - 1]
    prefix = f"[{n}] "
    if text.startswith(prefix):
        text = text[len(prefix):]
    return label, text


@command(
    "source",
    "Show the full material behind a citation [n] of the last answer.",
    aliases=("sources",),
    usage="/source [n]",
    details="""
Answers cite their evidence inline ([1], [2], …) with a Sources footer mapping each number to the
tool call or document behind it. This command shows the FULL text behind a number — the complete
tool observation or retrieved passage the synthesizer actually read — using the same numbering
the answer used.

  /source        list the last answer's sources (numbered labels)
  /source 3      print everything behind citation [3]

Scope: the most recent turn (the accumulators reset when a new turn starts; /clear empties them).
For older runs, /trace #<id> replays the full tool I/O of any recorded run.

Companion: /glass shows the trust/provenance facets of these SAME citations (origin · trust)
under the same numbering — /source is the raw material, /glass is whether to trust it.
""",
)
def _source(ctx, args):
    from nodes.synthesize import build_sources

    state = ctx.state or {}
    tool_results = state.get("tool_results") or []
    docs = state.get("documents_retrieved") or []
    _, _, sources = build_sources(tool_results, docs)

    if not sources:
        _print("  (the last answer drew on no gathered sources — nothing to cite)")
        return

    if not args:
        _print("  sources of the last answer  (/source <n> for the full text):")
        for n, label in sources:
            _print(f"    [{n}] {label}")
        return

    try:
        n = int(args[0].lstrip("[").rstrip("]"))
    except ValueError:
        _print(f"  usage: /source [n]   (n is a citation number, 1–{len(sources)})")
        return

    found = lookup_source(state, n)
    if found is None:
        _print(f"  no source [{n}] — the last answer has {len(sources)} source(s); /source lists them.")
        return
    label, text = found
    _print(f"  [{n}] {label}")
    _print("")
    for line in text.splitlines() or [""]:
        _print(f"  {line}")
    _print("")
