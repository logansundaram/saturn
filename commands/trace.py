from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from app import __version__
from commands._framework import command, _print
from stores.trace import decode_json
from textutil import clip as _clip, fmt_args, json_terminal_safe, split_sources_footer


@contextmanager
def _connect(db_path):
    """The trace DB connection for one read, closed on exit."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _to_int(s) -> Optional[int]:
    """Parse a run selector token to an int, tolerating a leading '#'. None if not a number."""
    try:
        return int(str(s).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _parse_run_selector(args, *, consume=None):
    """THE run-selector grammar, shared by every /trace subview. Recognized everywhere:

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
              hint="/trace -l",
              report=_print):
    """THE latest-run fallback + row loader. `latest_from` lets /trace invoke default to the
    newest run that HAS llm_calls. Returns (run_id, row); row is None (after `report`ing why)
    when there is nothing to show. `columns`/`latest_from` are code-controlled literals, never
    user input."""
    if run_id is None:
        row = conn.execute(f"SELECT MAX(run_id) FROM {latest_from}").fetchone()
        run_id = row[0] if row else None
        if run_id is None:
            report(empty_msg)
            return None, None
    run = conn.execute(
        f"SELECT {columns} FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not run:
        report(f"  no run #{run_id} — try {hint} to list recorded runs.")
        return run_id, None
    return run_id, run


def _raise_lookup(msg: str):
    """A `_load_run` reporter for callers that render the miss themselves (export_run)."""
    raise LookupError(msg.strip())


# --- /trace export ------------------------------------------------------------------------------
# One run's complete record (run + events + LLM calls) written to a self-contained JSON file —
# the record format /trace replay renders offline. Older exports may carry `integrity`/`signature`
# blocks — replay ignores them.

# The versioned artifact-format marker embedded in every export (layout versioning).
ARTIFACT_FORMAT = "saturn-artifact/1"


def _export_payload(run, events, calls) -> dict:
    run_id, query, started_at, ended_at, status, response = run
    payload = {
        "saturn_trace_export": 1,
        "format": ARTIFACT_FORMAT,
        "saturn_version": __version__,
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


def export_run(
    db_path,
    run_id: Optional[int] = None,
    dest: Optional[Path] = None,
) -> "tuple[Path, dict]":
    """THE one export-payload builder + writer — shared by the /trace export handler and the
    headless `saturn -p ... --export FILE` flag, so the two surfaces can never drift onto
    different payloads. `run_id=None` exports the latest run; `dest=None` writes the default
    logging/exports/run_<id>.json. Returns (path written, payload as written). Raises
    LookupError (no such run) / OSError (write failed) — each caller renders those its own
    way (REPL note vs. stderr + exit code)."""
    with _connect(db_path) as conn:
        run_id, run = _load_run(conn, run_id, report=_raise_lookup)
        events = conn.execute(
            "SELECT seq, ts, node, summary, data FROM events WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()
        calls = conn.execute(
            "SELECT seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status "
            "FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()

    payload = _export_payload(run, events, calls)

    from config import get_config

    if dest is None:
        dest = get_config().path("exports") / f"run_{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json_terminal_safe(json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8"
    )
    return dest, payload


def _export(ctx, args):
    out_path: Optional[str] = None
    bad_out = False

    def consume(low, a, it):
        nonlocal out_path, bad_out
        if low in ("-o", "--out", "--output"):
            out_path = next(it, None)
            # A dangling -o must not silently write the default, and a flag-shaped "path"
            # (-o --foo) is a swallowed flag, not a destination — refuse both before any
            # DB/file work.
            if out_path is None or out_path.startswith("-"):
                bad_out = True
            return True
        return False

    run_id, _count, _list = _parse_run_selector(args, consume=consume)
    if bad_out:
        _print("  usage: /trace export [#id] [-o <path>] — -o needs a path; "
               "nothing written")
        return

    try:
        dest, payload = export_run(
            ctx.db_path,
            run_id,
            dest=Path(out_path).expanduser() if out_path else None,
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
    _print("    (replayable offline: /trace replay <file>, or saturn --replay <file>)")


# --- /trace replay · saturn --replay -----------------------------------------------------------
# Render an exported run record OFFLINE, through the exact same drill-down view /trace uses on the
# live DB — what makes an export not just inspectable but SHAREABLE: attach a .json to a bug report
# and the recipient replays the full rail (plan, reasoning, tool I/O, answer) with no database.

def export_rows(payload: dict):
    """Rebuild (run_tuple, event_rows) from an export payload, in the shapes ui.show_run expects
    (event `data` re-encoded to JSON — the export stores it decoded). Pure, for tests."""
    # Every shape is CHECKED, not assumed. An export is the attach-it-to-a-bug-report path, so
    # the payload is untrusted by design: a "run" that decodes to a list, or an event whose
    # "data" is a list, must not raise out of render_export (and, via saturn --replay, out of
    # main() as a traceback). A malformed record renders as much as it can.
    run = payload.get("run")
    run = run if isinstance(run, dict) else {}
    run_tuple = (
        run.get("run_id"), run.get("query"), run.get("started_at"),
        run.get("ended_at"), run.get("status"), run.get("response"),
    )
    events = payload.get("events")
    rows = []
    for ev in events if isinstance(events, list) else []:
        if not isinstance(ev, dict):
            continue
        data = ev.get("data")
        try:
            encoded = json.dumps(data) if data is not None else None
        except (TypeError, ValueError):
            encoded = None
        rows.append((ev.get("seq"), ev.get("ts"), ev.get("node"), ev.get("summary"), encoded))
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

    try:
        run_tuple, rows = export_rows(payload)
    except Exception as e:  # the contract is a bool, never a traceback out of --replay
        print(f"  {path.name} could not be rendered: {e}", file=sys.stderr)
        return False
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
            else "agent · tools · gate decisions (plumbing folded)"
        )
        _print(f"  live trace on — {level}: {detail}.")


# --- /trace why — decision provenance ----------------------------------------------------------
# /trace shows WHAT happened; this subview reconstructs WHY: the causal chain from the model's
# checklist (if it wrote one), each agent pass's thought + chosen tool calls, the evidence relied
# on, and the cited sources.

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
        calls = conn.execute(
            "SELECT seq, node, output FROM llm_calls WHERE run_id = ? ORDER BY seq, id",
            (run_id,),
        ).fetchall()

    _render_why(ui, run, events, calls)


def _final_plan(events) -> list:
    """The plan as it stood at the end of the run — the last event delta that carried one."""
    plan = []
    for _seq, _node, _summary, data in events:
        delta = decode_json(data, {})
        if delta.get("plan"):
            plan = delta["plan"]
    return plan


def _call_names(tcs) -> str:
    return ", ".join(f"{c.get('name')}({_fmt_call_args(c.get('args'))})" for c in tcs)


def _group_passes(rows: list, entries: list) -> "list | None":
    """Pair the agent's recorded model calls with its passes: [(pass number, the call that
    STANDS, the pass's other call or None)]. A pass is one call unless thinking made it two
    (core/think.entry): a discarded draft then its rethink (`draft`), or a thought that came
    to nothing beside the think-off call that stands — the rerun after it, or the draft
    before it. None when the records do not line up (no think records: an older run; extra
    calls from a malformed-output retry or the unbound hard stop), so the caller falls back
    to one line per call rather than guess."""
    if not entries:
        return None
    out, i = [], 0
    for e in entries:
        two = bool(e.get("draft")) or (bool(e.get("asked")) and e.get("outcome") != "thought")
        chunk = rows[i:i + (2 if two else 1)]
        i += len(chunk)
        if len(chunk) != (2 if two else 1):
            return None
        if not two:
            out.append((e.get("pass"), chunk[0], None))
        elif e.get("draft") or chunk[0].get("think"):
            out.append((e.get("pass"), chunk[1], chunk[0]))  # the later call stands
        else:
            out.append((e.get("pass"), chunk[0], chunk[1]))  # the draft stands
    return out if i == len(rows) else None


def _think_entries(events) -> list:
    """Every agent pass's think record this run, in order (the `think` key of agent deltas)."""
    out = []
    for _seq, node, _summary, data in events:
        if node != "agent":
            continue
        out.extend(e for e in (decode_json(data, {}).get("think") or []) if isinstance(e, dict))
    return out


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
    ui.section(f"why · run #{run_id}", f"status: {status or '?'}")

    _print("  the request")
    _print(f"    {_clip(query, 120) or '(none)'}")
    _print("")

    # What it set out to do — the plan.
    plan = _final_plan(events)
    if plan:
        _print("  what it set out to do")
        for s in plan:
            glyph = ui.status_glyph(s.get("status"))
            tool = f"  [{s['intended_tool']}]" if s.get("intended_tool") else ""
            _print(f"    {glyph} {s.get('step_id')}. {s.get('label')}{tool}")
        _print("")

    # How it reasoned — every agent pass, from the recorded LLM I/O: the pre-call thought and the
    # calls it chose, or the answer. A pass can hold two model calls because of thinking (a
    # thought that was dropped and the rerun; a think-off draft and its rethink): the think
    # records say which call STANDS, so a call the model drafted and never issued is shown as
    # that, not as a choice it made.
    rows = [decode_json(output, {}) for _seq, node, output in calls if node == "agent"]
    grouped = _group_passes(rows, _think_entries(events))
    if grouped is None:  # an older run, or a pass with retries the records cannot place
        grouped = [(i, row, None) for i, row in enumerate(rows, 1)]
    if grouped:
        _print("  how it reasoned")
    for step, out, other in grouped:
        content = _clip(out.get("content", ""), 240)
        tcs = out.get("tool_calls") or []
        if tcs:
            _print(f"    pass {step}: {content or '(no preamble)'}")
        else:
            _print(f"    pass {step}: answered" + (f" — {content}" if content else ""))
        # A thinking pass's reasoning (stores/trace._llm_output) — why it chose what follows.
        reasoning = " ".join(str(out.get("reasoning") or "").split())
        if reasoning:
            _print(f"      thought: {_clip(reasoning, 400)}")
        if other is not None:
            dropped = " ".join(str(other.get("reasoning") or "").split())
            if other.get("think"):
                _print("      a thought was dropped (empty, cut or stopped)"
                       + (f": {_clip(dropped, 200)}" if dropped else ""))
            elif other.get("tool_calls"):
                _print(f"      drafted without thinking, then rethought — not issued: "
                       f"{_call_names(other['tool_calls'])}")
        if tcs:
            _print(f"      → chose to call: {_call_names(tcs)}")
    if grouped:
        _print("")

    # When it thought — one line per agent pass (core/think.entry, off the agent's events):
    # the kind of step each pass was and what came of its thought. Shown only for a turn where
    # thinking came up at all; a turn of plain passes has nothing to explain.
    passes = _think_entries(events)
    if any(e.get("outcome") not in (None, "none") for e in passes):
        from core import think

        _print("  when it thought")
        for e in passes:
            _print(f"    pass {e.get('pass')}: {think.describe(e)}")
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
        _print("    (nothing gathered — answered from the model's own knowledge + context)")
        _print("")

    # Provenance footer of the answer, if the agent attached one (the [n] → source map), through
    # THE one parser of the footer nodes/agent.py builds — never a split on the first "Sources:",
    # which an answer's own prose can contain.
    _, entries = split_sources_footer(response)
    if entries:
        _print("  cited sources (from the answer)")
        for line in entries:
            _print(f"    {line.strip()}")
        _print("")

    _print(f"  full step-by-step record: /trace #{run_id}   ·   model I/O: /trace invoke #{run_id}")


def _fmt_call_args(args) -> str:
    return fmt_args(args, 41) if isinstance(args, dict) else ""


@command(
    "trace",
    "Observability hub: drill-down of recorded runs + live trace control.",
    usage="/trace [#id | -l [n] | why | source | invoke"
          " | export | replay | on|off|full]",
    details="""
Expands one recorded run from the trace database (database/db.sqlite) into the full replay the
live trace abbreviates: the query, every node of the loop with its step time and metrics, the
checklist (if the model wrote one), the agent's reasoning and tool-call decisions at each pass,
each tool call WITH its output (the live trace hides that), and — last and de-emphasized — the
recorded final answer. This is the execution log, not a reprint of the
response.

With no argument it expands the MOST RECENT run. Select another run by id, or list runs to find
one:

  /trace            expand the last run
  /trace #7         expand run 7   (also: -r 7, --run 7, or just: /trace 7)
  /trace -l         list recent runs at a glance — the run ids live here
  /trace -l 20      list the last 20

Every turn is one run. This is the durable record that survives restarts.
Subviews:

  /trace why [#id]     decision provenance: not WHAT happened but WHY — the checklist it wrote
                       (if any), each pass's recorded thought + tool choice, the evidence the
                       answer was built from, and the cited sources. Defaults to the last run.
  /trace source [n]    the FULL material behind a citation [n] of the last answer — the complete
                       tool observation or retrieved passage the agent read, under the same
                       numbering the answer used. Bare lists the numbered sources.
  /trace invoke [#id]  the LLM calls of a run: each model call's INPUT messages + OUTPUT, with
                       timing + token counts. Defaults to the most recent run with LLM calls; add
                       --full to show whole messages — literally what your machine sent the model
                       (`/trace context` is the same view, full) — -l to list runs that have them.
  /trace export [#id]  write a run's complete record (events + tool I/O + LLM calls) to a
                       self-contained replayable JSON file under logging/exports/; -o <path>
                       to choose the destination. The record you can hand to someone else
                       (also: saturn -p "..." --export <file> writes the same artifact after
                       a headless turn).
  /trace replay <file> replay an exported record OFFLINE through the same drill-down view —
                       no database needed. What makes an export shareable: anyone can replay
                       a run you hand them (also: saturn --replay <file> straight from the
                       shell).

Live trace verbosity (controls what scrolls during a turn; recording is always on):

  /trace off    only the final response prints — runs quietly
  /trace on     normal: agent · tools · gate decisions (plumbing nodes folded)  [default]
  /trace full   verbose: every node line, including folded plumbing + full timings
""",
)
def _trace(ctx, args):
    from tui import ui

    if args and args[0].lower() in ("why", "--why"):
        return _why(ctx, args[1:])
    if args and args[0].lower() in ("source", "sources", "src"):
        return _source(ctx, args[1:])
    if args and args[0].lower() in ("invoke", "--invoke", "llm", "--llm", "model", "models"):
        return _show_llm_calls(ctx, args[1:])
    if args and args[0].lower() in ("context", "--context", "ctx", "prompt", "prompts"):
        # The context view is invoke, whole.
        return _show_llm_calls(ctx, ["--full", *args[1:]])
    if args and args[0].lower() in ("export", "--export"):
        return _export(ctx, args[1:])
    if args and args[0].lower() in ("replay", "--replay"):
        return _replay(ctx, args[1:])
    # NOTE: no "0"/"1" verbosity aliases here — a bare digit is a RUN ID (`/trace 1` drills into
    # run #1, same as `/trace #1`).
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

    # The llm_calls table always exists: the Tracer (constructed before any command runs) creates
    # it, IF NOT EXISTS, on every DB it opens.
    with _connect(ctx.db_path) as conn:
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


# ── /trace source — the raw material behind a citation ────────────────────────────────────────
# The Sources footer gives each thing the answer drew on a number and a one-line label; this
# shows the FULL tool result / retrieved passage behind that number, rebuilt with the same
# numbering (core.sources.build_sources over the turn's accumulators), so [3] here is exactly
# the [3] under the answer. Closes the provenance loop in one keystroke instead of a /trace drill-down.


def _source(ctx, args):
    """`/trace source [n]` — the FULL text behind a citation [n] of the last answer."""
    from core.sources import build_sources

    state = ctx.state or {}
    sources = build_sources(state.get("tool_results"), state.get("documents_retrieved"))

    if not sources:
        _print("  (the last answer drew on no gathered sources — nothing to cite)")
        return

    if not args:
        _print("  sources of the last answer  (/trace source <n> for the full text):")
        for n, label, _text in sources:
            _print(f"    [{n}] {label}")
        return

    try:
        n = int(args[0].lstrip("[").rstrip("]"))
    except ValueError:
        _print(f"  usage: /trace source [n]   (n is a citation number, 1–{len(sources)})")
        return

    if not 1 <= n <= len(sources):
        _print(f"  no source [{n}] — the last answer has {len(sources)} source(s); /trace source lists them.")
        return
    _n, label, text = sources[n - 1]
    _print(f"  [{n}] {label}")
    _print("")
    for line in text.splitlines() or [""]:
        _print(f"  {line}")
    _print("")
