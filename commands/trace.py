from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import signing
from commands._framework import command, _print
from stores.trace import decode_json, parse_ts
from textutil import clip as _clip, fmt_args


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
        call_repr, _, observation = (full or "").partition(" -> ")
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


# --- /trace export · /trace verify ------------------------------------------------------------
# The seed of the audit layer: one run's complete record (run + events + LLM calls) written to a
# self-contained file. JSON is the audit format — canonical, with a tamper-evident sha256 digest
# that `/trace verify` recomputes. --md renders a human-readable report instead (no digest).

# The canonical-digest scheme and version stamp live in signing.py (the one home for the byte
# stream every verification reproduces) — these aliases keep this module's call sites readable.
_saturn_version = signing.saturn_version
_canonical_digest = signing.canonical_digest


def _export_payload(run, events, calls) -> dict:
    run_id, query, started_at, ended_at, status, response = run
    return {
        "saturn_trace_export": 1,
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


def _export(ctx, args):
    fmt_md = False
    run_id: Optional[int] = None
    out_path: Optional[str] = None
    it = iter(args)
    for a in it:
        low = a.lower()
        if low in ("--md", "-m", "md", "markdown"):
            fmt_md = True
        elif low in ("-o", "--out", "--output"):
            out_path = next(it, None)
        elif low in ("-r", "--run"):
            rid = _to_int(next(it, ""))
            if rid is not None:
                run_id = rid
        elif a.startswith("#") or a.lstrip("+-").isdigit():
            rid = _to_int(a)
            if rid is not None:
                run_id = rid
        else:
            _print(f"  ignoring unrecognized argument: {a!r}")

    with _connect(ctx.db_path) as conn:
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

    if out_path:
        dest = Path(out_path).expanduser()
    else:
        ext = "md" if fmt_md else "json"
        dest = get_config().path("exports") / f"run_{run_id}.{ext}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if fmt_md:
            dest.write_text(_export_markdown(payload), encoding="utf-8")
        else:
            digest = _canonical_digest(payload)
            payload["integrity"] = {"algorithm": "sha256", "digest": digest}
            # Sign the digest (provenance, on top of the digest's tamper-evidence). The signature
            # block is NOT part of the canonical bytes the digest covers — it's added after — so
            # /trace verify recomputes the same digest, then checks the signature over it.
            signed = None
            if bool(get_config().get("runtime.sign_exports", True)):
                signed = signing.sign_digest(digest)
                if signed:
                    payload["signature"] = signed
            dest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except OSError as e:
        _print(f"  could not write {dest}: {e}")
        return

    _print(f"  run #{run_id} exported -> {dest}")
    _print(f"    {len(payload['events'])} event(s), {len(payload['llm_calls'])} LLM call(s)")
    if not fmt_md:
        _print(f"    sha256 {payload['integrity']['digest']}")
        if payload.get("signature"):
            _print(f"    signed   ed25519 by key {payload['signature'].get('key_id', '?')}")
            _print("    (anyone can verify content + signer offline: /trace verify <file>)")
        else:
            _print("    (anyone can re-check it later: /trace verify <file>)")
            if bool(get_config().get("runtime.sign_exports", True)) and not signing.available():
                _print("    (unsigned — install `cryptography` to sign exports: pip install cryptography)")


# --- /trace replay · saturn --replay -----------------------------------------------------------
# Render an exported run record OFFLINE, through the exact same drill-down view /trace uses on the
# live DB — what makes an export not just verifiable but SHAREABLE: attach a .json to a bug report
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
    """Load an exported run record, verify its digest, and replay it via ui.show_run. Used by
    `/trace replay <file>` and the `saturn --replay <file>` CLI flag. Returns False on a file
    that can't be rendered (the CLI exits non-zero on it)."""
    from tui import ui

    path = Path(path_str.strip('"')).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _print(f"  could not read {path}: {e}")
        return False
    if not isinstance(payload, dict) or payload.get("saturn_trace_export") != 1:
        _print(f"  {path.name} is not a Saturn trace export (see /trace export).")
        return False

    # Integrity first — a replay states up front whether the record it renders is intact. A
    # failed digest still renders (the content may be exactly what you need to inspect), loudly.
    # signing.verify_payload is THE one verify flow (it pops integrity+signature off a copy
    # before recomputing — the rule every verifier must share).
    v = signing.verify_payload(payload)
    if v["has_integrity"]:
        if v["digest_ok"]:
            _print(f"  ✓ integrity verified — sha256 {v['computed_digest'][:16]}…")
        else:
            _print(f"  ⨯ INTEGRITY FAILURE — {path.name} was modified after export "
                   "(rendering anyway; do not treat it as an authentic record).")
        if v["signed"]:
            if v.get("signature_ok"):
                mine = " (this machine)" if v.get("signer_is_local") else ""
                _print(f"  ✓ signed — ed25519 key {v.get('key_id', '?')}{mine}")
            else:
                _print(f"  ⨯ SIGNATURE INVALID — key {v.get('key_id', '?')} does not "
                       "verify this record.")
    else:
        _print("  (no integrity digest — an unverifiable record; JSON exports carry one)")

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


def _verify(ctx, args):
    if not args:
        _print("  usage: /trace verify <exported .json file>")
        return
    path = Path(" ".join(args).strip('"')).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _print(f"  could not read {path}: {e}")
        return
    is_export = isinstance(payload, dict) and payload.get("saturn_trace_export") == 1
    # /privacy report -o writes the SAME digest+signature layering — verify it here too instead
    # of bouncing the user ("verifiable like a trace export" must have a verify path).
    is_report = isinstance(payload, dict) and payload.get("saturn_trust_report") is not None
    if not is_export and not is_report:
        _print(f"  {path.name} is not a Saturn trace export or trust report.")
        return
    # signing.verify_payload owns the fragile rule: BOTH the integrity and signature blocks are
    # popped (off a copy) before recomputing — neither is part of the canonical digest bytes.
    v = signing.verify_payload(payload)
    if not v["has_integrity"]:
        _print(f"  {path.name} carries no integrity digest (a --md report? only JSON exports do).")
        return
    kind = "trace export" if is_export else "trust report"
    if v["digest_ok"]:
        _print(f"  ✓ {path.name} verifies ({kind}) — sha256 {v['stored_digest']}")
        if is_export:
            _print(f"    run #{payload['run']['run_id']}, recorded {payload['run']['started_at']}")
        else:
            _print(f"    generated {payload.get('generated_at', '?')}, "
                   f"session {payload.get('session_id', '?')}")
    else:
        _print(f"  ⨯ {path.name} DOES NOT verify — the record was modified after export.")
        _print(f"    stored   {v['stored_digest']}")
        _print(f"    computed {v['computed_digest']}")

    # Signature (provenance) — independent of the digest check. Verified over the STORED digest
    # (what was signed); combined with the digest check above, both passing means authentic AND
    # untampered AND provably from the signer's key.
    if v["signed"]:
        if v.get("signature_ok"):
            mine = " (this machine's key)" if v.get("signer_is_local") else ""
            _print(f"  ✓ signature valid — ed25519 key {v.get('key_id', '?')}{mine}")
            _print(f"    public key {v.get('public_key', '?')}")
        else:
            _print(f"  ⨯ signature INVALID — does not match the embedded key {v.get('key_id', '?')} "
                   "(forged, corrupted, or the digest was altered).")
    elif not signing.available():
        _print("  (no signature in this export; `cryptography` is not installed here to check one)")
    else:
        _print("  (no signature in this export — digest-only; was runtime.sign_exports off?)")


def _key(ctx, args):
    """`/trace key` — show this machine's audit-signing public key + fingerprint, the value you
    publish so others can verify your signed exports. Creates the keypair on first call."""
    from tui import ui

    info = signing.key_info()
    if not info.get("available"):
        ui.section("signing key", "unavailable — the `cryptography` library is not installed")
        _print("  exports fall back to a sha256 integrity digest only (still tamper-evident,")
        _print("  but not provenance-signed). Install it to sign:  pip install cryptography")
        return

    ui.section("signing key", f"ed25519 · key id {info.get('key_id') or '?'}")
    rows = [
        ("key id", info.get("key_id") or "?"),
        ("public key", info.get("public_key") or "?"),
        ("created", info.get("created_at") or "(just now)"),
        ("key file", (info.get("path") or "?", "dim")),
    ]
    ui.table(rows)
    _print("  the PUBLIC key above is safe to publish — share it so anyone can confirm a signed")
    _print("  export came from you (/trace verify reports the signer's fingerprint).")
    _print("  the private key never leaves this machine and is never embedded in an export.")


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
            else "plan · agent · tools · synthesize (plumbing folded)"
        )
        _print(f"  live trace on — {level}: {detail}.")


# --- /trace why — decision provenance ----------------------------------------------------------
# /trace shows WHAT happened; this subview reconstructs WHY: the causal chain from the recorded
# plan, per-step agent reasoning + chosen tool calls, the evidence relied on, the groundedness
# verdict, and the cited sources. (Folded in from the old standalone /why, June 2026.)

def _why(ctx, args):
    from tui import ui

    run_id: Optional[int] = None
    for a in args:
        rid = _to_int(a) if (a.startswith("#") or a.lstrip("+-").isdigit()) else None
        if rid is not None:
            run_id = rid

    with _connect(ctx.db_path) as conn:
        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
            run_id = row[0] if row else None
            if run_id is None:
                _print("  (no runs recorded yet — ask something first)")
                return
        run = conn.execute(
            "SELECT run_id, query, started_at, ended_at, status, response FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            _print(f"  no run #{run_id} — try /trace -l to list recorded runs.")
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
    _GLYPH = {"pending": "○", "active": "▸", "done": "✓", "skipped": "—"}

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

    # How it reasoned — the agent steps + the groundedness judge, from the recorded LLM I/O.
    step = 0
    judged = None
    for _seq, node, output in calls:
        out = decode_json(output, {})
        if node == "agent":
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
        elif node == "replan":
            judged = out.get("content", "")
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

    # Verification — did the groundedness judge weigh in?
    if judged:
        _print("  verification")
        _print(f"    groundedness judge: {_clip(judged, 160)}")
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
# cited source's origin (local vs network), trust, and — the headline — whether a span of an
# untrusted source bled verbatim into the answer (the data→answer channel). Bare reads the live last
# turn (exact egress); #id reconstructs from the recorded run (egress inferred from source tools).

def _answer(ctx, args):
    from tui import ui
    import egress
    import glassbox

    run_id: Optional[int] = None
    for a in args:
        rid = _to_int(a) if (a.startswith("#") or a.lstrip("+-").isdigit()) else None
        if rid is not None:
            run_id = rid

    state = ctx.state or {}
    # "Live" means THIS process ran the last turn: the per-turn accumulators (or current_query)
    # are populated. Messages alone do NOT count — a /resume-restored conversation carries only
    # messages, and rendering it against a fresh process's empty egress ledger would produce a
    # false 'local-only, 0 sources' box for an answer that may have been cloud-composed from web
    # sources last session. Such turns reconstruct from the recorded run below instead.
    live = bool(state.get("tool_results") or state.get("documents_retrieved")
                or state.get("tool_events") or state.get("current_query"))
    # Bare + a live last turn → the live Glass Box (exact egress slice + gate count from the UI).
    if run_id is None and live:
        import receipt
        from tui.ui import _base
        mark = receipt.turn_mark()
        gated = _base._status.get("gates", 0) if isinstance(_base._status, dict) else 0
        # The exact slice is trustworthy only when a turn mark was recorded AND no
        # `/privacy egress clear` wiped events past it. Otherwise pass None (unknown): an
        # empty-because-cleared slice must never render as 'local-only this turn'.
        ev = None
        if mark > 0 and not egress.cleared_since(mark):
            ev = egress.events_since(mark)
        gb = glassbox.build_from_state(state, egress_events=ev, gated=gated)
        ui.show_glassbox(gb)
        return

    # Otherwise reconstruct from the recorded run (last, or the requested #id).
    with _connect(ctx.db_path) as conn:
        if run_id is None:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
            run_id = row[0] if row else None
            if run_id is None:
                _print("  (no runs recorded yet — ask something first)")
                return
        run = conn.execute(
            "SELECT run_id, query, response FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run:
            _print(f"  no run #{run_id} — try /trace -l to list recorded runs.")
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

  /trace why [#id]     decision provenance: not WHAT happened but WHY — the plan it drafted, the
                       model's recorded reasoning + tool choice at each step, the evidence the
                       answer was built from, the groundedness judge's verdict, and the cited
                       sources. Defaults to the last run.
  /trace answer [#id]  the Glass Box (also: /glass): answer-level provenance — each cited source's
                       origin (local vs network) and trust, what left the machine, and whether a
                       span of an untrusted source bled verbatim into the answer (the data→answer
                       channel, colored red inline). Bare = the live last turn; #id reconstructs a
                       recorded run.
  /trace invoke [#id]  the LLM calls of a run: each model call's INPUT messages + OUTPUT, with
                       timing + token counts. Defaults to the most recent run with LLM calls; add
                       --full to show whole messages, -l to list runs that have them.
  /trace calls [n]     recent tool calls + their outputs
  /trace cost [--all]  session totals: turns, time, tokens
  /trace state         dump the live AgentState (message count, plan steps, tools called, etc.)
                       pass --full to also dump the raw state dict
  /trace export [#id]  write a run's complete record (events + tool I/O + LLM calls) to a
                       self-contained file under logging/exports/ — JSON with a sha256 integrity
                       digest AND (when `cryptography` is installed) an ed25519 signature proving
                       it came from this machine's key; --md for a readable markdown report;
                       -o <path> to choose the destination. The audit record you can hand to
                       someone else.
  /trace verify <file> recompute an exported record's digest (content unchanged?) and check its
                       signature (signed by which key?) — a modified or forged record fails loudly.
  /trace key           show this machine's audit-signing public key + fingerprint — the value you
                       publish so others can confirm a signed export is yours. The private key
                       never leaves this machine.
  /trace replay <file> replay an exported record OFFLINE through the same drill-down view —
                       integrity-checked first, no database needed. What makes an export
                       shareable: anyone can replay a run you hand them (also: saturn --replay
                       <file> straight from the shell).

Live trace verbosity (controls what scrolls during a turn; recording is always on):

  /trace off    only the final response prints — runs quietly
  /trace on     normal: plan · agent · tools · synthesize (plumbing nodes folded)  [default]
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
    if args and args[0].lower() in ("verify", "--verify"):
        return _verify(ctx, args[1:])
    if args and args[0].lower() in ("key", "--key", "pubkey", "publickey"):
        return _key(ctx, args[1:])
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

    with _connect(ctx.db_path) as conn:
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
                _print(f"    #{rid:<4} {when}  {str(status):<7} {n_events:>2}ev  {_clip(query, 56)}")
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
                (max(1, run_id or 10),),
            ).fetchall()
            if not rows:
                _print("  (no LLM calls recorded yet)")
                return
            _print("  runs with LLM calls — newest first  (/trace invoke #<id> to expand one):")
            for rid, n, dur, query in rows:
                _print(f"    #{rid:<4} {n:>2} call(s)  {float(dur or 0):>6.1f}s  {_clip(query, 50)}")
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

    ui.show_llm_calls(run, calls, full=full)
