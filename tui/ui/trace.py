"""
The execution trace: the live per-node rail (`show_node` + the tool-I/O sub-tree) and the recorded
drill-downs the `/trace` command replays (`show_run` for the node-level log, `show_llm_calls` for
the model-level `/trace invoke` view). Shares the rail/glyph/tree vocabulary across live and replay
so a turn reads the same whether it's happening now or being inspected later.
"""

import textwrap
import time

from textutil import clip, fmt_args, fmt_call, human_bytes, split_call_result

from . import _base
from ._base import (
    Padding, Text, _console,
    _ACCENT, _BLOCKED_GLYPH, _DIM, _FAINT, _NODE_W, _RAIL,
    _TREE_END, _TREE_LEAF, _TREE_MID, _TREE_PIPE,
    _emit, _fmt_dur, _human_tokens, _rail, _term_width, _truncate,
)
from .statusbar import _live_refresh
from .plan import show_plan
from .listing import section


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


def _node_line(node: str, dur: float, delta: dict) -> "Text":
    """Build one `│ ✓ node  elapsed  metrics` trace row (metrics dim) — the shared format for the
    live trace and the /trace replay."""
    extra = " · ".join(_metric_parts(delta))
    line = _rail()
    line.append("✓ ", style="green")  # the node has finished by the time its line prints
    line.append(f"{node:<{_NODE_W}}", style="default")
    line.append(f"{_fmt_dur(dur):>7}", style=_DIM)
    if extra:
        line.append(f"   {extra}", style=_DIM)  # metrics are tertiary — dim, never the accent
    return line


def _is_answer(delta: dict) -> bool:
    """Whether an agent delta is the ANSWER pass: its last message is an AIMessage without tool
    calls (the text that streamed into the response region). A tool-calling pass is not."""
    msgs = (delta or {}).get("messages") or []
    last = msgs[-1] if msgs else None
    if last is None or isinstance(last, str):
        return False
    return getattr(last, "type", "") == "ai" and not getattr(last, "tool_calls", None)


def _agent_row_is_quiet(node: str, delta: dict) -> bool:
    """Whether the agent's rail row can be skipped this pass: the ANSWER pass at normal
    verbosity, and only when `_render_trust_annotations` will draw NOTHING under it (a bounded
    record keeps the row so its leaf has a parent). The answer pass's update fires after the
    answer began streaming into the open response region, and rich inserts a console print
    ABOVE a live display — the row would shove the streaming answer down mid-stream. The
    question is answered by running the real annotator with its output discarded, never by a
    second list of delta keys."""
    if node != "agent" or _base._VERBOSITY == "verbose" or not _is_answer(delta):
        return False
    return _render_trust_annotations(node, delta, emit=lambda _text, _style: None) == 0


def show_node(node: str, delta: dict | None = None) -> None:
    """One trace line per node execution — `│ <node>  <elapsed>  <annotation>` — with the elapsed
    measured since the previous node emitted (htop-style). LLM nodes annotate with iter / context
    tokens / tok-per-sec; the `tools` node renders a sub-tree of its calls (args · timing · result
    preview) beneath the header, so the agent's actual actions are fully visible, not hidden."""
    now = time.perf_counter()
    dur = now - _base._t_last if _base._t_last is not None else 0.0
    _base._t_last = now

    # The plumbing node (ground) folds at normal verbosity: its timing rolls into the next
    # visible node. Everything stays in the trace DB for /trace and /trace invoke.
    if node in _base._FOLD_NODES and _base._VERBOSITY != "verbose":
        return
    # approval passes through before EVERY tool round; an auto-approved pass is plumbing. When a
    # HUMAN decided (the delta carries gate_events) it is signal and renders with its leaf below.
    if node == "approval" and not (delta or {}).get("gate_events") and _base._VERBOSITY != "verbose":
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

    # Per-node trace row: `│ ✓ node  elapsed  metrics` (metrics dim). The metric annotations are
    # built from the delta by the shared _node_line helper (the live trace + the /trace replay
    # render identical rows).
    #
    # The agent's ANSWER pass is the exception (`_agent_row_is_quiet`): its update fires after
    # the answer began streaming into the open response region, so at normal verbosity the ROW
    # is skipped while everything else about the pass still lands — its metrics were fed to the
    # status bar above and are echoed in the receipt, and `/trace full` restores the row.
    if not _agent_row_is_quiet(node, delta):
        if not _base._trace_started:
            _emit("")  # one blank line parting the turn's trace from the prompt above it
            _base._trace_started = True
        _emit(_node_line(node, dur, delta))

    # The agent's thought: the text alongside its tool calls (the same words the gate's
    # `e(xplain)` shows) renders as a dim leaf under the agent rail line — the "why" of the
    # calls. The answer pass streams under `── response` instead, never here.
    if node == "agent" and not _is_answer(delta):
        _render_agent_thought(delta.get("messages") or [])

    if delta.get("tool_events"):
        _render_tool_events(delta["tool_events"])

    _render_trust_annotations(node, delta)

    _live_refresh()  # repaint the bar with the new node/iter/tools immediately


# Cap the live reasoning preview: enough to read the thought, not enough to drown the trace.
_REASONING_CAP = 280


def _leaf(first: str, rest: str, text: str, style: str, *, reserve: int = 0) -> None:
    """One word-wrapped leaf under the rail: `│ <first><text>`, continuation lines hanging at
    `rest` (the same width as `first`) so a long text stays inside the trace gutter instead of
    spilling to column 0. THE one wrapped-leaf renderer — every leaf shape below is a prefix pair."""
    avail = max(20, _term_width() - (4 + len(first) + reserve))  # minus the rail and the prefix
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        row = _rail()
        row.append(first if i == 0 else rest, style=_RAIL)
        row.append(ln, style=style)
        _emit(row)


def _node_leaf(text: str, style: str) -> None:
    """One wrapped `└ …` annotation leaf directly under a node's rail line — the shared shape for
    the agent's reasoning preview and the gate-decision echo."""
    _leaf(f"  {_TREE_LEAF} ", "    ", text, style, reserve=2)


def _render_agent_thought(messages: list) -> None:
    """Render the agent's message text as dim, wrapped leaf lines under its trace row: the
    pre-call thought alongside its tool calls (the same words the gate's `e(xplain)` shows).
    Quietly does nothing when the message has no text."""
    msg = messages[-1] if messages else None
    if msg is None:
        return
    text = msg.content if isinstance(getattr(msg, "content", ""), str) else str(getattr(msg, "content", ""))
    text = clip(text, _REASONING_CAP)
    if not text:
        return
    _node_leaf(text, _DIM)


def _render_trust_annotations(node: str, delta: dict, *, emit=None) -> int:
    """The trust-stack annotations a node's delta carries, rendered identically in the live rail
    and the /trace replay:

      - under `approval`, the echo of each HUMAN gate decision (state["gate_events"]): the
        interactive prompt scrolls away with the turn, so this leaf is the transcript's
        permanent record of who allowed what — green for approved, red for rejected, with the
        quarantine escalation named when one forced the prompt.

    Returns the number of leaves drawn, and takes an `emit` override so a caller can ask what
    WOULD be drawn without drawing it (`_agent_row_is_quiet`) — the conditions below stay the
    single producer of that answer."""
    drawn = 0
    _emit_leaf = emit or _node_leaf

    def leaf(text: str, style: str) -> None:
        nonlocal drawn
        drawn += 1
        _emit_leaf(text, style)

    # A delta the tracer had to bound at write time says so (stores/trace._bound_delta): what
    # was dropped is NAMED, never silently absent from the record the user is reading.
    tr = delta.get("truncated")
    if isinstance(tr, dict):
        dropped = tr.get("dropped") or []
        what = ("everything" if dropped == ["*"] else
                ", ".join(str(k) for k in dropped) if dropped else "no keys (leaves clipped)")
        size = tr.get("original_chars")
        leaf("record bounded at write time"
             + (f" ({size} chars)" if size is not None else "")
             + f" — dropped: {what}", "yellow")
    for ev in delta.get("gate_events") or []:
        if not isinstance(ev, dict):
            continue
        calls = [c for c in ev.get("calls") or [] if isinstance(c, dict)]
        approved = [str(c.get("name") or "?") for c in calls if c.get("approved")]
        rejected = [str(c.get("name") or "?") for c in calls if not c.get("approved")]
        why = []
        if ev.get("quarantine"):
            why.append("quarantine escalation")
        suffix = f" ({', '.join(why)})" if why else ""
        if approved:
            leaf("✓ you approved " + ", ".join(approved) + suffix, "green")
        if rejected:
            leaf("✗ you rejected " + ", ".join(rejected) + suffix, "red")
    return drawn


def _emit_result_leaf(cont: str, text: str, style: str) -> None:
    """A tool result/error leaf under its call branch: the `└` leaf glyph under the call text,
    the `cont` gutter kept on continuation lines."""
    _leaf(f"  {cont}  {_TREE_LEAF} ", f"  {cont}    ", text, style)


def _render_tool_events(events: list[dict], *, always_show_results: bool = False) -> None:
    """Draw the tool-I/O sub-tree under the `tools` node header: one `├─ name(args)  dur` branch
    per call, the call repr sized to the terminal and durations column-aligned within the round so
    they read as a column. Each call's result renders as ONE clipped line beneath it (watching it
    work means seeing what came back); `/trace #id`, `/trace full` and the /trace replay (`always_show_results=True`) show the full output, word-wrapped under the
    rail with a hanging indent. A FAILED call's error always shows whole."""
    n = len(events)
    # Width-responsive: size the call repr to the room left after the tree prefix (~9) and the right
    # `   dur` column (~9), then align durations to the widest call in this round.
    call_cap = max(24, _term_width() - 18)
    calls = [_truncate(f"{ev.get('name', '?')}({fmt_args(ev.get('args', {}), 48)})", call_cap)
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
        # One clipped line by default; the whole output under /trace full, in the /trace replay,
        # or for an error (signal, never clipped).
        show_result = bool(result)
        full = always_show_results or not ok or _base._VERBOSITY == "verbose"

        line = _rail()
        line.append("  ", style=_RAIL)            # nest under the node column
        line.append(f"{branch} ", style=_RAIL)
        line.append(f"{call:<{col_w}}", style="default" if ok else "red")
        line.append(f"   {dur}", style=_DIM)
        _console.print(line)
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
            text = result if full else _truncate(" ".join(str(result).split()), call_cap)
            _emit_result_leaf(cont, text, _DIM if ok else "red")


def _egress_leaf(eg: dict) -> tuple[str, str]:
    """(text, style) for one per-call egress annotation (the dicts nodes/tools._egress_slice
    attaches). A send names the host, size and channel; a block names what the
    air-gap refused. The `more` marker is the slice's own overflow cap."""
    if "more" in eg:
        n = eg.get("more")
        return (f"⇅ +{n} more egress event{'s' if n != 1 else ''} — /policy egress", "yellow")
    host = str(eg.get("host") or "?")
    channel = str(eg.get("channel") or "")
    if eg.get("status") == "blocked":
        # `⊘`, not `⛔`: the latter is East-Asian Wide AND emoji-presentation, so terminals render
        # it as a color emoji that ignores the `bold red` style and overflows the rail column.
        # `⊘` (_base._BLOCKED_GLYPH, the receipt's glyph too) is one cell and takes the style.
        return (f"{_BLOCKED_GLYPH} air-gap blocked {channel or 'egress'} → {host} — nothing sent", "bold red")
    parts = [f"⇅ sent → {host}"]
    n = eg.get("n_bytes") or 0
    if n:
        parts.append(human_bytes(n))
    if channel:
        parts.append(channel)
    return (" · ".join(parts), "yellow")


_MSG_ROLE = {"AIMessage": "ai", "HumanMessage": "in", "SystemMessage": "sys"}


def _render_trace_messages(delta: dict) -> None:
    """Render the messages a recorded node ADDED — chiefly the agent's reasoning text and its
    tool-call decisions — as dim `└ <role>  <text>` leaves under its /trace replay row: what
    turns the replay from a reprint of the answer into a real execution log. Messages arrive as
    the trace DB's `"AIMessage: <text> [tool_calls: ...]"` strings (stores.trace._json_default).
    ToolMessages are skipped (the tool sub-tree already carries their output)."""
    for m in (delta.get("messages") or []):
        kind, _, content = str(m).partition(": ")
        kind = kind.strip()
        if "ToolMessage" in kind:
            continue
        content = " ".join(content.split())  # collapse to a compact one-block preview
        if not content:
            continue
        head = f"  {_TREE_END} {_MSG_ROLE.get(kind, kind.lower() or 'msg'):<3} "
        _leaf(head, " " * len(head), content, _DIM)


# ── run drill-down (the /trace expanded view) ─────────────────────────────────────
def _enrich_results(events: list[dict], results: list, cap: int = 1200) -> list[dict]:
    """Pair each recorded tool event with the fuller `call -> observation` from tool_results
    (collapsed to one line, capped), so the /trace replay shows real output where the live tree
    deliberately showed nothing. Paired by the CALL, not by position: tool_results holds only the
    calls that completed and gathered something (nodes/tools.py), so a failed call, a plan
    update or a knowledge-base search has no entry and keeps its own preview."""
    # THE one parser of the `name(args) -> observation` serialization nodes/tools.py builds.
    pending = [split_call_result(r) for r in results]
    out = []
    for ev in events:
        ev = dict(ev)
        label = fmt_call(str(ev.get("name")), ev.get("args") if isinstance(ev.get("args"), dict) else {})
        hit = next((p for p in pending if p[0] == label), None) if ev.get("ok", True) else None
        if hit is not None:
            pending.remove(hit)
            obs = clip(hit[1], cap)
            if obs:
                ev["result"] = obs
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
    from stores.trace import decode_json, parse_ts, response_truncated

    run_id, query, started_at, ended_at, status, response_text = run

    # header: run id, then the query echoed at a `»`, then a dim when · status · total-time line.
    _run_header(f"run #{run_id}", query)
    start_dt, end_dt = parse_ts(started_at), parse_ts(ended_at)
    when = (started_at or "")[:19].replace("T", " ")
    total = _fmt_dur((end_dt - start_dt).total_seconds()).strip() if (start_dt and end_dt) else ""
    status_style = {"ok": "green", "error": "bold red", "running": "yellow"}.get(str(status), _DIM)
    meta = Text("  ")
    meta.append(when or "—", style=_DIM)
    meta.append(" · ", style=_DIM)
    meta.append(str(status), style=status_style)
    if total:
        meta.append(" · ", style=_DIM)
        meta.append(total, style=_DIM)
    _console.print(meta)
    _emit("")

    # node-by-node replay: every node shows — this IS the full drill-down, plumbing (ground, an
    # auto-approved approval pass) and tool outputs included.
    saved_seen = _base._plan_seen
    _base._plan_seen = {}  # let show_plan diff afresh over this run's plan events
    prev = start_dt
    try:
        for _seq, ts, node, _summary, data in events:
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
            _render_trace_messages(delta)
            tev = delta.get("tool_events") or []
            if tev:
                _render_tool_events(_enrich_results(tev, delta.get("tool_results") or []),
                                    always_show_results=True)
            # human gate decisions replay exactly as the live rail showed them
            _render_trust_annotations(node, delta)
    finally:
        _base._plan_seen = saved_seen

    # the run's final answer — subordinate in the replay. The execution log above is the subject of
    # /trace; the answer is just the recorded outcome, so it's rendered quietly (dim plaintext under
    # a faint label) rather than as the bold-accent markdown the LIVE turn already showed.
    if response_text:
        _emit("")
        # end_run's write-time truncation marker (stores.trace.response_truncated): the stored row
        # holds a capped answer, so the label says "truncated" up front rather than letting the
        # reader discover the cut at the tail marker. Rows without the marker read False —
        # absent-as-unknown, never an inferred flag.
        rule = Text()
        rule.append("  ╶ ", style=_FAINT)
        rule.append("final answer", style=_DIM)
        rule.append(" (recorded)", style=_FAINT)
        if response_truncated(response_text):
            rule.append(" (truncated)", style=_DIM)
        _console.print(rule)
        # Recorded answers are typically long single-line paragraphs: render through the same
        # Padding idiom as the live answer body (response._print_markdown_body's fallback) so
        # every soft-wrapped continuation keeps the 2-space indent instead of spilling to
        # column 0. The measure IS the live answer's _BODY_WIDTH — imported, not copied, so
        # tuning it can never leave the replay wrapping at a stale width.
        from .response import _BODY_WIDTH

        body = Text(response_text, style=_DIM)
        _console.print(Padding(body, (0, 0, 0, 2)), width=min(_term_width(), _BODY_WIDTH))


def _run_header(title: str, query) -> None:
    """The replay header shared by show_run and show_llm_calls: the section rule, then the query
    echoed at a `»` (the same glyph it was typed at)."""
    section(title)
    qline = Text("  ")
    qline.append("» ", style=_DIM)
    qline.append(" ".join(str(query or "").split()) or "(empty)", style="default")
    _console.print(qline)


# ── LLM-call replay (/trace invoke) ──────────────────────────────────────────────
_LLM_PREVIEW_CHARS = 240  # per-message clip in the default (non --full) view
_LLM_ROLE = {"system": "sys", "human": "usr", "ai": "ai", "tool": "tool", "function": "fn"}


def _llm_leaf(tag: str, text: str, style: str, cap: int | None) -> None:
    """One input/output message under an LLM-call header: `│ tag  <wrapped text>`. `cap` bounds
    the preview (None = full)."""
    text = " ".join(str(text).split())
    if cap:
        text = _truncate(text, cap)
    head = f"  {tag:<4} "
    _leaf(head, " " * len(head), text, style)


def _recording_cut(m: dict) -> "str | None":
    """The disclosure for a message the TRACE capped at write time (stores.trace._LLM_MSG_CAP),
    or None when nothing was dropped. Measured against the recorded content (not the display
    preview), and emitted by callers as its own un-clipped leaf — appended to the body, a preview
    clip would cut it off for exactly the long messages it describes."""
    try:
        original = int(m.get("truncated") or 0)
    except (TypeError, ValueError):
        return None
    if not original:
        return None
    extra = original - len(str(m.get("content", "")))
    return f"… (+{extra} chars not recorded)" if extra > 0 else None


def show_llm_calls(run, calls, full: bool = False) -> None:
    """Replay every LLM call recorded for one run: per call its node + model + timing + token counts,
    the input messages sent, and the output produced. The `/trace invoke` view — the model-level
    companion to show_run's node-level replay, and the answer to "what did each model call actually
    see and say". `run` is (run_id, query, started_at, ended_at, status, response); `calls` are the
    (seq, ts, node, model, dur, prompt_tokens, output_tokens, input, output, status) rows in order.
    `full` lifts the per-message preview clip so the entire stored message text shows."""
    from stores.trace import decode_json

    run_id, query, *_rest = run
    cap = None if full else _LLM_PREVIEW_CHARS  # `cap`, not `clip` — textutil.clip is imported here

    _run_header(f"run #{run_id} · llm calls", query)

    if not calls:
        _emit("  (no LLM calls recorded for this run)")
        return

    # one-line roll-up: count · total time · total tokens in/out. Column order matches the query in
    # commands.trace._show_llm_calls: (seq, ts, node, model, dur, prompt_tokens, output_tokens, …).
    total_dur = sum((c[4] or 0) for c in calls)
    total_in = sum((c[5] or 0) for c in calls)
    total_out = sum((c[6] or 0) for c in calls)
    roll = (f"  {len(calls)} call(s) · {_fmt_dur(total_dur).strip()}"
            f" · {_human_tokens(total_in)}→{_human_tokens(total_out)} tok")
    _emit(Text(roll, style=_DIM))
    _emit("")

    for idx, (_seq, _ts, node, model, dur, ptok, otok, inp, outp, status) in enumerate(calls, 1):
        toks = f" · {_human_tokens(ptok or 0)}→{_human_tokens(otok or 0)} tok" if (ptok or otok) else ""
        h = Text("  ")
        h.append(f"{idx}. ", style=f"bold {_ACCENT}")
        h.append(str(node), style="default")
        h.append(f" · {model}", style=_DIM)
        h.append(f" · {_fmt_dur(dur or 0).strip()}{toks}", style=_DIM)
        if status != "ok":
            h.append(f" · {status}", style="bold red")
        _console.print(h)

        for m in decode_json(inp, []):
            tag = _LLM_ROLE.get(m.get("role", ""), (m.get("role") or "msg")[:4])
            body = m.get("content", "")
            tc = m.get("tool_calls")
            if tc:
                names = ", ".join(str(c.get("name")) for c in tc)
                body = (body + " " if body else "") + f"[tool_calls: {names}]"
            _llm_leaf(tag, body or "(empty)", _DIM, cap)
            cut = _recording_cut(m)
            if cut:  # its own leaf: never clipped away with the body it describes
                _llm_leaf("", cut, _DIM, None)

        out = decode_json(outp, {})
        out_body = out.get("content", "")
        if out.get("tool_calls"):
            names = ", ".join(str(c.get("name")) for c in out["tool_calls"])
            out_body = (out_body + " " if out_body else "") + f"[tool_calls: {names}]"
        if out.get("error"):
            out_body = f"ERROR: {out['error']}"
        _llm_leaf("out", out_body or "(no output)", "default" if status == "ok" else "red", cap)
        # The output side is capped at write time too (stores.trace._msg_out), so it carries the
        # same disclosure as the inputs above — without it, `--full` presents a capped reply as
        # the model's complete output.
        cut = _recording_cut(out)
        if cut:
            _llm_leaf("", cut, _DIM, None)
        _emit("")

