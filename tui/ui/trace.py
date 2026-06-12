"""
The execution trace: the live per-node rail (`show_node` + the tool-I/O sub-tree) and the recorded
drill-downs the `/trace` command replays (`show_run` for the node-level log, `show_llm_calls` for
the model-level `/trace invoke` view). Shares the rail/glyph/tree vocabulary across live and replay
so a turn reads the same whether it's happening now or being inspected later.
"""

import time

from textutil import human_bytes

from . import _base
from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _FAINT, _NODE_W, _RAIL, _RAIL_GLYPH,
    _TREE_END, _TREE_LEAF, _TREE_MID, _TREE_PIPE,
    _emit, _fmt_args, _fmt_dur, _human_tokens, _rail, _term_width, _truncate,
)
from .statusbar import _live_refresh
from .plan import show_plan


# ── execution trace ─────────────────────────────────────────────────────────────
def _metric_parts(delta: dict) -> list[str]:
    """Per-node metric annotations (iteration · context tokens · tok/s) pulled from a node delta —
    shared by the live trace (show_node) and the recorded replay (show_run)."""
    parts = []
    if "iteration" in delta:
        parts.append(f"iter {delta['iteration']}")
    used = delta.get("context_tokens") or 0
    if used > 0:
        parts.append(f"{_human_tokens(used)} ctx")
    tps = delta.get("tok_per_sec") or 0.0
    if tps > 0:
        parts.append(f"{tps:.0f} tok/s")
    return parts


def _node_line(node: str, dur: float, delta: dict) -> "Text | str":
    """Build one `│ ✓ node  elapsed  metrics` trace row (metrics dim) — the shared format for the
    live trace and the /trace replay. Returns a rich Text, or a plain string without rich."""
    extra = " · ".join(_metric_parts(delta))
    if _RICH:
        line = _rail()
        line.append("✓ ", style="green")  # the node has finished by the time its line prints
        line.append(f"{node:<{_NODE_W}}", style="default")
        line.append(f"{_fmt_dur(dur):>7}", style=_DIM)
        if extra:
            line.append(f"   {extra}", style=_DIM)  # metrics are tertiary — dim, never the accent
        return line
    tail = f"   {extra}" if extra else ""
    return f"  {_RAIL_GLYPH} ✓ {node:<{_NODE_W}}{_fmt_dur(dur):>7}{tail}"


def show_node(node: str, delta: dict | None = None) -> None:
    """One trace line per node execution — `│ <node>  <elapsed>  <annotation>` — with the elapsed
    measured since the previous node emitted (htop-style). LLM nodes annotate with iter / context
    tokens / tok-per-sec; the `tools` node renders a sub-tree of its calls (args · timing · result
    preview) beneath the header, so the agent's actual actions are fully visible, not hidden."""
    now = time.perf_counter()
    dur = now - _base._t_last if _base._t_last is not None else 0.0
    _base._t_last = now

    # plan_gate is a control checkpoint, not an informative node — skip its rail line so the
    # per-step pass-throughs don't double the trace. Its effects still surface elsewhere: a plan
    # edit via show_plan (the on_update subscriber calls it on a `plan` delta), a pause via the
    # plan-review prompt. _t_last is already advanced, so the next node's timing excludes the gate.
    # The plumbing nodes (ground, update_plan) fold the same way at normal verbosity: their timing
    # rolls into the next visible node, and update_plan's plan diff still prints (show_plan is
    # driven separately by on_update). Everything stays in the trace DB for /trace and /trace calls.
    if node == "plan_gate":
        return
    if node in _base._FOLD_NODES and _base._VERBOSITY != "verbose":
        return

    delta = delta or {}
    # Feed the pinned status bar from whatever this delta carried.
    called = delta.get("tools_called") or []
    _base._status["tools"] += len(called)
    if "iteration" in delta:
        _base._status["iteration"] = delta["iteration"]
    tps = delta.get("tok_per_sec") or 0.0
    if tps > 0:
        _base._status["tok_per_sec"] = tps
    used = delta.get("context_tokens") or 0
    if used > 0:
        _base._status["ctx_used"] = used
    _base._status["node"] = node

    # synthesize falls through to the normal rail line — its metrics (tok/s, context) are useful
    # for transparency; the streamed answer prints directly after it so the output is not repeated.

    # Per-node trace row: `│ ✓ node  elapsed  metrics` (metrics dim). The metric annotations are
    # built from the delta by the shared _node_line helper (the live trace + the /trace replay
    # render identical rows).
    if not _base._trace_started:
        _emit("")  # one blank line parting the turn's trace from the prompt above it
        _base._trace_started = True
    _emit(_node_line(node, dur, delta))

    # Live reasoning: when the agent thinks out loud BEFORE acting (text content alongside its
    # tool calls), surface it as a dim leaf under the agent's rail line — the "why" behind the
    # tool sub-tree that follows. Text-only agent messages are skipped: a draft answer flows into
    # synthesize (which the user sees), and showing it here would print it twice.
    if node == "agent":
        _render_agent_reasoning(delta.get("messages") or [])

    if delta.get("tool_events"):
        _render_tool_events(delta["tool_events"])

    _render_trust_annotations(node, delta)

    _live_refresh()  # repaint the bar with the new node/iter/tools immediately


# Cap the live reasoning preview: enough to read the thought, not enough to drown the trace.
_REASONING_CAP = 280


def _node_leaf(text: str, style: str) -> None:
    """One wrapped `└ …` annotation leaf directly under a node's rail line — the shared shape for
    the agent's reasoning preview, the judge's verdict, and the gate-decision echo."""
    import textwrap

    avail = max(20, _term_width() - 10)
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        prefix = f"{_TREE_LEAF} " if i == 0 else "  "
        if _RICH:
            row = _rail()
            row.append(f"  {prefix}", style=_RAIL)
            row.append(ln, style=style)
            _emit(row)
        else:
            _emit(f"  {_RAIL_GLYPH}   {prefix}{ln}")


def _render_agent_reasoning(messages: list) -> None:
    """Render the agent's pre-action reasoning (the text content of a tool-calling AIMessage) as
    dim, wrapped leaf lines under the agent's trace row. Quietly does nothing when the message
    has no text or no tool calls."""
    msg = messages[-1] if messages else None
    if msg is None or not getattr(msg, "tool_calls", None):
        return
    text = msg.content if isinstance(getattr(msg, "content", ""), str) else str(getattr(msg, "content", ""))
    text = " ".join(text.split())
    if not text:
        return
    _node_leaf(_truncate(text, _REASONING_CAP), _DIM)


def _render_trust_annotations(node: str, delta: dict) -> None:
    """The trust-stack annotations a node's delta carries, rendered identically in the live rail
    and the /trace replay — the moments that used to be invisible without a command:

      - under `replan`, the judge's verdict: the delta carries `replans` only when the judge
        found the draft UNGROUNDED and inserted a search (the grounded/skip path updates
        nothing, so all the rail can honestly say there is "no escalation");
      - under `approval`, the echo of each HUMAN gate decision (state["gate_events"]): the
        interactive prompt scrolls away with the turn, so this leaf is the transcript's
        permanent record of who allowed what — green for approved, red for rejected, with the
        quarantine/taint escalation named when one forced the prompt."""
    if node == "replan":
        if delta.get("replans"):
            _node_leaf("judge: draft answer ungrounded — inserted a web search step", "yellow")
        else:
            _node_leaf("judge: draft answer accepted (no escalation)", _DIM)
    for ev in delta.get("gate_events") or []:
        if not isinstance(ev, dict):
            continue
        calls = [c for c in ev.get("calls") or [] if isinstance(c, dict)]
        approved = [str(c.get("name") or "?") for c in calls if c.get("approved")]
        rejected = [str(c.get("name") or "?") for c in calls if not c.get("approved")]
        why = []
        if ev.get("quarantine"):
            why.append("quarantine escalation")
        if ev.get("taint"):
            why.append("tainted args")
        suffix = f" ({', '.join(why)})" if why else ""
        if approved:
            _node_leaf("✓ you approved " + ", ".join(approved) + suffix, "green")
        if rejected:
            _node_leaf("✗ you rejected " + ", ".join(rejected) + suffix, "red")


def _emit_result_leaf(cont: str, text: str, style: str) -> None:
    """Emit a tool result/error leaf under its call branch, word-wrapped to the terminal with a
    HANGING INDENT: the first line carries the `└` leaf glyph, continuation lines indent to sit
    under the text (keeping the `cont` rail gutter), so a long output stays inside the trace rail
    instead of spilling to column 0."""
    import textwrap

    first = f"{cont}  {_TREE_LEAF} "   # "│  └ " / "   └ " — leaf glyph, under the call text
    rest = f"{cont}    "               # "│    " / "     " — aligns continuation under the text
    avail = max(20, _term_width() - (4 + 2 + len(first)))  # minus rail(4) + nest(2) + leaf prefix
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        prefix = first if i == 0 else rest
        if _RICH:
            row = _rail()
            row.append("  ", style=_RAIL)
            row.append(prefix, style=_RAIL)
            row.append(ln, style=style)
            _console.print(row)
        else:
            print(f"  {_RAIL_GLYPH}   {prefix}{ln}")


def _render_tool_events(events: list[dict], *, always_show_results: bool = False) -> None:
    """Draw the tool-I/O sub-tree under the `tools` node header: one `├─ name(args)  dur` branch
    per call, the call repr sized to the terminal and durations column-aligned within the round so
    they read as a column. The raw result preview is **hidden** by default — it's noisy JSON, and
    `/trace calls` (or `/trace full`) surfaces full outputs on demand — but a FAILED call still shows
    its error leaf inline (signal, not noise). What the agent *did* (name · args · cost · ok/fail)
    always stays visible. `always_show_results=True` (the /trace replay) shows every output, word-
    wrapped under the rail with a hanging indent."""
    n = len(events)
    # Width-responsive: size the call repr to the room left after the tree prefix (~9) and the right
    # `   dur` column (~9), then align durations to the widest call in this round.
    call_cap = max(24, _term_width() - 18)
    calls = [_truncate(f"{ev.get('name', '?')}({_fmt_args(ev.get('args', {}))})", call_cap)
             for ev in events]
    col_w = max((len(c) for c in calls), default=0)
    for i, ev in enumerate(events):
        last = i == n - 1
        branch = _TREE_END if last else _TREE_MID
        cont = " " if last else _TREE_PIPE  # gutter under the branch for the result/error leaf
        call = calls[i]
        dur = _fmt_dur(ev.get("dur", 0.0))
        ok = ev.get("ok", True)
        result = ev.get("result", "")
        # Outputs are hidden by default; show only errors, or everything under /trace full or in
        # the /trace replay (always_show_results).
        show_result = bool(result) and (always_show_results or not ok or _base._VERBOSITY == "verbose")

        if _RICH:
            line = _rail()
            line.append("  ", style=_RAIL)            # nest under the node column
            line.append(f"{branch} ", style=_RAIL)
            line.append(f"{call:<{col_w}}", style="default" if ok else "red")
            line.append(f"   {dur}", style=_DIM)
            _console.print(line)
        else:
            print(f"  {_RAIL_GLYPH}   {branch} {call:<{col_w}}   {dur}")
        # Boundary events this call produced (tool_events[].egress, attached by tool_node): the
        # moment something leaves the machine the rail says so — a send in yellow, an air-gap
        # block in red. Signal, like an error leaf, never folded by verbosity.
        for eg in ev.get("egress") or []:
            if isinstance(eg, dict):
                text, style = _egress_leaf(eg)
                _emit_result_leaf(cont, text, style)
        # Injection quarantine: an untrusted result that carried instruction-shaped content was
        # flagged + fenced (quarantine.py) — surface that in the rail, always (it's signal, like
        # an error leaf, never folded by verbosity).
        q = ev.get("quarantine")
        if q:
            _emit_result_leaf(
                cont,
                f"⚠ embedded instructions detected ({', '.join(q)}) — content quarantined, "
                "treated as data",
                "yellow",
            )
        if show_result:
            _emit_result_leaf(cont, result, _DIM if ok else "red")


def _egress_leaf(eg: dict) -> tuple[str, str]:
    """(text, style) for one per-call egress annotation (the dicts nodes/tools._egress_slice
    attaches). A send names the host, size, channel and any redactions; a block names what the
    air-gap refused. The `more` marker is the slice's own overflow cap."""
    if "more" in eg:
        n = eg.get("more")
        return (f"⇅ +{n} more egress event{'s' if n != 1 else ''} — /privacy egress", "yellow")
    host = str(eg.get("host") or "?")
    channel = str(eg.get("channel") or "")
    if eg.get("status") == "blocked":
        return (f"⛔ air-gap blocked {channel or 'egress'} → {host} — nothing sent", "bold red")
    parts = [f"⇅ sent → {host}"]
    n = eg.get("n_bytes") or 0
    if n:
        parts.append(human_bytes(n))
    if channel:
        parts.append(channel)
    r = eg.get("redactions") or 0
    if r:
        parts.append(f"{r} redaction{'s' if r != 1 else ''}")
    return (" · ".join(parts), "yellow")


_MSG_ROLE = {"AIMessage": "ai", "HumanMessage": "in", "SystemMessage": "sys"}


def _msg_kind_content(m) -> tuple[str, str]:
    """Normalize one delta message to `(kind, content)` — handling BOTH forms it can take:
      - a live LangChain message OBJECT (the live trace: `delta["messages"]` straight off the
        graph), or
      - the trace DB's pre-serialized `"AIMessage: <text> [tool_calls: ...]"` STRING (the /trace
        replay: deltas are JSON, and `stores.trace._json_default` flattened each message to a string).
    Returning the same `(kind, content)` for both keeps the live and replay rendering identical.
    The object branch mirrors `_json_default`'s format (content + a `[tool_calls: …]` suffix) so a
    content-less tool-calling turn still records WHAT the agent decided."""
    if not isinstance(m, str):
        kind = type(m).__name__
        content = str(getattr(m, "content", "") or "")
        calls = getattr(m, "tool_calls", None)
        if calls:
            names = ", ".join(c.get("name", "?") for c in calls)
            content = (content + " " if content else "") + f"[tool_calls: {names}]"
        return kind, content
    kind, _, content = m.partition(": ")
    return kind.strip(), content


def _emit_message_leaf(label: str, text: str) -> None:
    """One message/reasoning leaf under a node row: `└ <role>  <wrapped text>`, hanging-indented
    under the rail so a long thought stays inside the trace gutter. Dim — it's narrative, not the
    accent. Used by the /trace replay to surface the agent's actual thinking between tool calls."""
    import textwrap

    head = f"  {_TREE_END} {label:<3} "   # nest(2) + leaf glyph + fixed-width role tag
    rest = " " * len(head)               # continuation lines align under the text
    avail = max(20, _term_width() - (4 + len(head)))
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        if _RICH:
            row = _rail()
            row.append(head if i == 0 else rest, style=_RAIL)
            row.append(ln, style=_DIM)
            _console.print(row)
        else:
            print(f"  {_RAIL_GLYPH} {head if i == 0 else rest}{ln}")


def _render_trace_messages(node: str, delta: dict, max_chars: int | None = None) -> None:
    """Render the messages a node ADDED — chiefly the agent's reasoning text and its tool-call
    decisions — as dim leaves under its trace row. This is the piece the default tool tree never
    surfaces, and what turns the /trace replay from a reprint of the answer into a real execution
    log. ToolMessages are skipped (the tool sub-tree already carries their output) and the
    synthesize node's message is skipped (it's the final answer, shown once in the response section
    below). Used by the /trace replay (`max_chars=None`); `_msg_kind_content` normalizes message
    forms. `max_chars` clips each leaf to a preview (the full text lives in the /trace replay)."""
    if node == "synthesize":
        return
    for m in (delta.get("messages") or []):
        kind, content = _msg_kind_content(m)
        if "ToolMessage" in kind:
            continue
        content = " ".join(content.split())  # collapse to a compact one-block preview
        if not content:
            continue
        if max_chars:
            content = _truncate(content, max_chars)
        _emit_message_leaf(_MSG_ROLE.get(kind, kind.lower() or "msg"), content)


# ── run drill-down (the /trace expanded view) ─────────────────────────────────────
def _enrich_results(events: list[dict], results: list, cap: int = 1200) -> list[dict]:
    """Pair each recorded tool event with the fuller `call -> observation` from tool_results
    (collapsed to one line, capped), so the /trace replay shows real output where the live tree
    deliberately showed nothing. Falls back to the event's own preview when no pair exists."""
    out = []
    for i, ev in enumerate(events):
        ev = dict(ev)
        if i < len(results):
            _, _, obs = str(results[i]).partition(" -> ")
            obs = " ".join(obs.split())
            if obs:
                ev["result"] = _truncate(obs, cap)
        out.append(ev)
    return out


def show_run(run, events) -> None:
    """Replay one recorded run from the trace DB as an expanded drill-down (the default /trace view):
    the query, every node with its wall-clock step time + metrics, the plan as it advanced, the
    agent's reasoning + tool-call decisions per step (the `ai`/`in` leaves — the execution-log detail
    the live trace omits), each tool call WITH its output (the live trace hides these too), and last,
    de-emphasized, the recorded final answer. The full-fidelity counterpart to the live trace; same
    rail/glyph/tree vocabulary, but here the EXECUTION LOG is the subject, not the response.

    `run` is the row `(run_id, query, started_at, ended_at, status, response)`; `events` are its
    `(seq, ts, node, summary, data)` rows in order. Step times are wall-clock deltas between event
    timestamps, so a tool step that waited on the approval gate honestly includes that pause."""
    from stores.trace import decode_json, parse_ts

    run_id, query, started_at, ended_at, status, response_text = run

    # header: run id · query · when / status / total wall time
    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append(f"run #{run_id}", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
    else:
        print(f"  ╶── run #{run_id} " + "─" * 40)

    q = " ".join(str(query or "").split()) or "(empty)"
    start_dt, end_dt = parse_ts(started_at), parse_ts(ended_at)
    when = (started_at or "")[:19].replace("T", " ")
    total = _fmt_dur((end_dt - start_dt).total_seconds()).strip() if (start_dt and end_dt) else ""
    status_style = {"ok": "green", "error": "bold red", "running": "yellow"}.get(str(status), _DIM)
    if _RICH:
        qline = Text("  ")
        qline.append("query  ", style=_DIM)
        qline.append(q, style="default")
        _console.print(qline)
        meta = Text("  ")
        meta.append(when or "—", style=_DIM)
        meta.append("  ·  ", style=_DIM)
        meta.append(str(status), style=status_style)
        if total:
            meta.append("  ·  ", style=_DIM)
            meta.append(total, style=_DIM)
        _console.print(meta)
    else:
        print(f"  query  {q}")
        print(f"  {when}  ·  {status}" + (f"  ·  {total}" if total else ""))
    _emit("")

    # node-by-node replay. plan_gate is a control checkpoint with no info (folded in the live trace
    # too); everything else shows — this IS the full drill-down, plumbing and tool outputs included.
    saved_seen = _base._plan_seen
    _base._plan_seen = {}  # let show_plan diff afresh over this run's plan events
    prev = start_dt
    try:
        for _seq, ts, node, _summary, data in events:
            if node == "plan_gate":
                continue
            delta = decode_json(data, {})
            cur = parse_ts(ts)
            dur = (cur - prev).total_seconds() if (cur and prev) else 0.0
            if cur:
                prev = cur
            _emit(_node_line(node, dur, delta))
            if delta.get("plan"):
                show_plan(delta["plan"])
            # the agent's reasoning / tool-call decisions for this step — the execution-log detail
            # the live trace omits; this is the point of the drill-down
            _render_trace_messages(node, delta)
            tev = delta.get("tool_events") or []
            if tev:
                _render_tool_events(_enrich_results(tev, delta.get("tool_results") or []),
                                    always_show_results=True)
            # judge verdicts + human gate decisions replay exactly as the live rail showed them
            _render_trust_annotations(node, delta)
    finally:
        _base._plan_seen = saved_seen

    # the run's final answer — subordinate in the replay. The execution log above is the subject of
    # /trace; the answer is just the recorded outcome, so it's rendered quietly (dim plaintext under
    # a faint label) rather than as the bold-accent markdown the LIVE turn already showed.
    if response_text:
        _emit("")
        if _RICH:
            rule = Text()
            rule.append("  ╶ ", style=_FAINT)
            rule.append("final answer", style=_DIM)
            rule.append(" (recorded)", style=_FAINT)
            _console.print(rule)
            for ln in response_text.splitlines() or [""]:
                row = Text("  ")
                row.append(ln, style=_DIM)
                _console.print(row)
        else:
            print("  ╶ final answer (recorded)")
            for ln in response_text.splitlines() or [""]:
                print(f"  {ln}")


# ── LLM-call replay (/trace invoke) ──────────────────────────────────────────────
_LLM_PREVIEW_CHARS = 240  # per-message clip in the default (non --full) view
_LLM_ROLE = {"system": "sys", "human": "usr", "ai": "ai", "tool": "tool", "function": "fn"}


def _llm_leaf(tag: str, text: str, style: str, clip: int | None) -> None:
    """One input/output message under an LLM-call header: `tag  <wrapped text>`, hanging-indented to
    align continuation lines, in the trace palette. `clip` bounds the preview (None = full)."""
    import textwrap

    text = " ".join(str(text).split())
    if clip:
        text = _truncate(text, clip)
    head = f"    {tag:<4} "
    rest = " " * len(head)
    avail = max(20, _term_width() - (4 + len(head)))
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [""]):
        if _RICH:
            row = Text()
            row.append(head if i == 0 else rest, style=_RAIL)
            row.append(ln, style=style)
            _console.print(row)
        else:
            print(f"{head if i == 0 else rest}{ln}")


def show_llm_calls(run, calls, full: bool = False) -> None:
    """Replay every LLM call recorded for one run: per call its node + model + timing + token counts,
    the input messages sent, and the output produced. The `/trace invoke` view — the model-level
    companion to show_run's node-level replay, and the answer to "what did each model call actually
    see and say". `run` is (run_id, query, started_at, ended_at, status, response); `calls` are the
    (seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status) rows in order.
    `full` lifts the per-message preview clip so the entire stored message text shows."""
    from stores.trace import decode_json

    run_id, query, *_rest = run
    clip = None if full else _LLM_PREVIEW_CHARS

    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append(f"run #{run_id}", style=f"bold {_ACCENT}")
        rule.append("  llm calls", style=_DIM)
        rule.append(" " + "─" * 32, style=_DIM)
        _console.print(rule)
    else:
        print(f"  ╶── run #{run_id}  llm calls " + "─" * 32)
    q = " ".join(str(query or "").split()) or "(empty)"
    _emit(("  query  " + q) if not _RICH else Text("  query  ", style=_DIM) + Text(q))

    if not calls:
        _emit("  (no LLM calls recorded for this run)")
        return

    # one-line roll-up: count · total time · total tokens in/out. Column order matches the query in
    # commands.trace._show_llm_calls: (seq, ts, node, model, dur, prompt_tokens, output_tokens, …).
    total_dur = sum((c[4] or 0) for c in calls)
    total_in = sum((c[5] or 0) for c in calls)
    total_out = sum((c[6] or 0) for c in calls)
    roll = (f"  {len(calls)} call(s)  ·  {_fmt_dur(total_dur).strip()}"
            f"  ·  {_human_tokens(total_in)}→{_human_tokens(total_out)} tok")
    _emit(roll if not _RICH else Text(roll, style=_DIM))
    _emit("")

    for idx, (_seq, _ts, node, model, dur, ptok, otok, inp, outp, status) in enumerate(calls, 1):
        toks = f"  ·  {_human_tokens(ptok or 0)}→{_human_tokens(otok or 0)} tok" if (ptok or otok) else ""
        if _RICH:
            h = Text("  ")
            h.append(f"{idx}. ", style=f"bold {_ACCENT}")
            h.append(str(node), style="default")
            h.append(f"  ·  {model}", style=_DIM)
            h.append(f"  ·  {_fmt_dur(dur or 0).strip()}{toks}", style=_DIM)
            if status != "ok":
                h.append(f"  ·  {status}", style="bold red")
            _console.print(h)
        else:
            err = f"  ·  {status}" if status != "ok" else ""
            print(f"  {idx}. {node}  ·  {model}  ·  {_fmt_dur(dur or 0).strip()}{toks}{err}")

        for m in decode_json(inp, []):
            tag = _LLM_ROLE.get(m.get("role", ""), (m.get("role") or "msg")[:4])
            body = m.get("content", "")
            tc = m.get("tool_calls")
            if tc:
                names = ", ".join(str(c.get("name")) for c in tc)
                body = (body + " " if body else "") + f"[tool_calls: {names}]"
            if m.get("truncated"):
                body += f"  (+{m['truncated'] - _LLM_PREVIEW_CHARS} chars)" if not full else ""
            _llm_leaf(tag, body or "(empty)", _DIM, clip)

        out = decode_json(outp, {})
        out_body = out.get("content", "")
        if out.get("tool_calls"):
            names = ", ".join(str(c.get("name")) for c in out["tool_calls"])
            out_body = (out_body + " " if out_body else "") + f"[tool_calls: {names}]"
        if out.get("error"):
            out_body = f"ERROR: {out['error']}"
        _llm_leaf("out", out_body or "(no output)", "default" if status == "ok" else "red", clip)
        _emit("")
