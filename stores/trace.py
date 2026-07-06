"""
Structured run trace -> SQLite (database/db.sqlite).

Every turn becomes a row in `runs`; every node update streamed during that turn becomes a row
in `events`. This is the transparency/observability layer: it makes
every run inspectable after the fact and is the data source the frontend will render later.

It supersedes the scattered `print(perf_counter)` lines — those now go to `diag.log()` (the file
diagnostic log), while the durable, queryable per-turn record lives here.
"""

import json
import sqlite3
from datetime import datetime
from time import perf_counter

from textutil import head_tail, map_strings


# ── read-side helpers (shared by /trace in commands and the tui replay views) ──
def decode_json(data, default):
    """Decode a JSON blob stored by the tracer (an `events.data` delta or an `llm_calls`
    input/output column), falling back to `default` on NULL or undecodable rows."""
    try:
        return json.loads(data) if data else default
    except (json.JSONDecodeError, TypeError):
        return default


def parse_ts(ts):
    """Parse a stored ISO timestamp back to a datetime; None for NULL/garbage rows."""
    try:
        return datetime.fromisoformat(ts) if ts else None
    except (TypeError, ValueError):
        return None


# Write-time truncation marker for the recorded final answer (end_run). The stable PREFIX is the
# detection key — the cap value is appended after it so the stored row is self-describing even if
# the cap changes between recording and reading. One constant + one detector, shared by every
# reader (show_run's label, /glass #id, the export's answer attestation), so they can't drift.
_RESPONSE_TRUNCATION_MARKER = "… [recorded answer truncated at "


def response_truncated(text) -> bool:
    """True when a recorded `runs.response` carries end_run's write-time truncation marker.
    Readers treat a marked row as INCOMPLETE (show_run says "truncated", the Glass Box /
    attestation pass complete=False). Historical rows cut at the old 2000-char cap carry no
    marker and read False here — absent-as-unknown (the gotcha #7 convention): never try to
    infer truncation for legacy rows."""
    return _RESPONSE_TRUNCATION_MARKER in str(text or "")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT,
    query      TEXT,
    started_at TEXT,
    ended_at   TEXT,
    status     TEXT,
    response   TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER,
    seq     INTEGER,
    ts      TEXT,
    node    TEXT,
    summary TEXT,
    data    TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER,
    seq           INTEGER,
    ts            TEXT,
    node          TEXT,    -- the langgraph node the call was made from (plan/agent/replan/synthesize)
    model         TEXT,
    dur           REAL,    -- wall-clock seconds for the single model call
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    input         TEXT,    -- JSON: the messages sent to the model
    output        TEXT,    -- JSON: {content, tool_calls} the model returned
    status        TEXT     -- ok | error
);
"""


# How much of each message / delta the trace retains. These bound the durable execution log the
# /trace replay reads, so they're generous: the replay is the full-fidelity record (reasoning +
# tool decisions), not the abbreviated live rail. Bumped from 300/4000 — at the old caps a turn's
# reasoning was clipped to a sentence and a busy delta lost its tail.
_CONTENT_CAP = 1500
_DATA_CAP = 16000


def _json_default(o):
    # Messages -> "AIMessage: <content> [tool_calls: ...]"; pydantic objects -> dict; else -> str.
    # We fold the tool-call decision into the string so a content-less tool-calling turn still
    # records WHAT the agent decided to do (the live tool tree shows it; the replay needs it too).
    if hasattr(o, "content"):
        text = str(o.content)[:_CONTENT_CAP]
        calls = getattr(o, "tool_calls", None)
        if calls:
            names = ", ".join(c.get("name", "?") for c in calls)
            text = (text + " " if text else "") + f"[tool_calls: {names}]"
        return f"{type(o).__name__}: {text}"
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


# Per-string-leaf cap when a delta overruns _DATA_CAP (see _summarize).
_LEAF_CAP = 2000


def _summarize(delta: dict) -> tuple[str, str]:
    parts = []
    if delta.get("plan"):
        parts.append("plan=[" + "; ".join(
            f"{s.get('status', '?')}:{s.get('label', '?')}" for s in delta["plan"]) + "]")
    if delta.get("tools_called"):
        parts.append("tools=" + ", ".join(delta["tools_called"]))
    if "iteration" in delta:
        parts.append(f"iter={delta['iteration']}")
    if "messages" in delta:
        parts.append(f"+{len(delta['messages'])}msg")
    summary = " | ".join(parts) or "(update)"
    data = json.dumps(delta, default=_json_default)
    if len(data) > _DATA_CAP:
        # Clip long string LEAVES and re-encode instead of slicing the JSON text: a mid-token
        # cut stores an undecodable blob (decode_json -> default), degrading /trace replays and
        # Glass Box reconstruction to INCOMPLETE. Leaf-clipping keeps the record parseable.
        try:
            clipped = map_strings(json.loads(data), lambda s: head_tail(s, _LEAF_CAP))
            data = json.dumps(clipped)
        except Exception:
            pass
        data = data[:_DATA_CAP]  # last-resort bound (e.g. a plan with hundreds of steps)
    return summary, data


class Tracer:
    """Every write is BEST-EFFORT: the watcher must never take down the watched. log_event runs
    on every node delta of a live turn, so a sqlite failure here (db.sqlite locked by a second
    instance / an open DB browser — it's shared with SqliteSaver — or disk full) would otherwise
    raise out of run_turn's stream loop and report a healthy turn as failed."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._seq = 0
        self._llm_seq = 0
        self._broken = False  # one-shot circuit breaker — see _trip

    @property
    def broken(self) -> bool:
        """True while the circuit breaker is tripped (recording disabled until the next
        start_run). Exposed so the LOOP can surface a "trace recording degraded" notice —
        stores/ must not import tui, so this module never warns the user itself."""
        return self._broken

    def _trip(self, where: str, exc: Exception) -> None:
        """Swallow a write failure and trip the one-shot circuit breaker: after the FIRST failed
        write every later PER-DELTA write (log_event / log_llm_call) no-ops, because each failed
        execute/commit can block up to sqlite's busy timeout PER node delta — a dead
        observability layer must degrade to silence, not a multi-second stall on every update.
        end_run is exempt (one terminal write; see there). diag-logged once at trip time;
        start_run re-arms, so the next turn retries exactly once; the loop reads `broken` and
        warns the user the record degraded."""
        if not self._broken:
            import diag
            diag.log(f"trace {where} failed — recording disabled until the next run: {exc}")
        self._broken = True

    def start_run(self, thread_id: str, query: str) -> int:
        self._seq = 0
        self._llm_seq = 0
        self._broken = False  # re-arm the breaker: one retry per turn, never a permanently dead trace
        try:
            cur = self.conn.execute(
                "INSERT INTO runs (thread_id, query, started_at, status) VALUES (?, ?, ?, ?)",
                (thread_id, query, datetime.now().isoformat(), "running"),
            )
            self.conn.commit()
            return cur.lastrowid
        except Exception as exc:
            self._trip("start_run", exc)
            # Sentinel: log_event/end_run against -1 are harmless orphan writes / no-op updates
            # (and the breaker is tripped anyway). Headless --export fails loudly on its own path.
            return -1

    def log_event(self, run_id: int, node: str, delta: dict) -> None:
        # seq increments even when broken/failing, so any later successful rows stay ordered.
        self._seq += 1
        if self._broken:
            return
        try:
            # Inside the guard on purpose: _summarize serializes arbitrary node deltas (a
            # circular structure, an exotic object) and the watcher must never take down the
            # watched — an encode failure trips the breaker like any write failure.
            summary, data = _summarize(delta or {})
            self.conn.execute(
                "INSERT INTO events (run_id, seq, ts, node, summary, data) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, self._seq, datetime.now().isoformat(), node, summary, data),
            )
            self.conn.commit()
        except Exception as exc:
            self._trip("log_event", exc)

    def log_llm_call(self, run_id, node, model, dur, prompt_tokens, output_tokens,
                     input_json, output_json, status="ok") -> None:
        """Record one model call's input + output (from the LLMTraceHandler). Best-effort: a logging
        failure must never propagate into the running model call."""
        self._llm_seq += 1
        if self._broken:
            return
        try:
            self.conn.execute(
                "INSERT INTO llm_calls (run_id, seq, ts, node, model, dur, prompt_tokens, "
                "output_tokens, input, output, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, self._llm_seq, datetime.now().isoformat(), node, model, dur,
                 prompt_tokens, output_tokens, input_json, output_json, status),
            )
            self.conn.commit()
        except Exception as exc:
            self._trip("log_llm_call", exc)

    def llm_handler(self, run_id: int) -> "LLMTraceHandler":
        """A run-scoped LangChain callback that captures every model call's input/output into the
        trace DB. Pass it in the graph stream config's `callbacks` so it propagates to all nodes."""
        return LLMTraceHandler(self, run_id)

    def end_run(self, run_id: int, status: str, response: str = "") -> None:
        text = response or ""
        # The recorded answer is capped like a delta (_DATA_CAP — it IS the headline record every
        # after-the-fact surface reads: show_run, the export, /glass #id's reconstruction).
        # When it still overflows, the cut gets an explicit write-time marker so the stored row
        # is self-describing: readers render "truncated" / complete=False instead of presenting
        # a mid-sentence cut as the whole answer, and the export's digest commits the marker
        # honestly. (The old silent [:2000] cut even lost the Sources: footer.)
        if len(text) > _DATA_CAP:
            text = text[:_DATA_CAP] + f"\n{_RESPONSE_TRUNCATION_MARKER}{_DATA_CAP} chars]"
        # Deliberately EXEMPT from the circuit breaker: end_run is ONE write at turn end (not
        # the per-delta hot path the breaker protects from repeated busy-timeout stalls) and it
        # carries the run's terminal status + answer — a transient lock that tripped the breaker
        # early in the turn and cleared since must not leave this run 'running' forever with no
        # recorded response (/trace, /glass #id, and exports all read that row). Worst case one
        # more busy-timeout wait per turn; a failure still just trips/diag-logs.
        try:
            self.conn.execute(
                "UPDATE runs SET ended_at = ?, status = ?, response = ? WHERE run_id = ?",
                (datetime.now().isoformat(), status, text, run_id),
            )
            self.conn.commit()
        except Exception as exc:
            self._trip("end_run", exc)


# ── LLM-call capture ───────────────────────────────────────────────────────────
# A LangChain callback handler that records the raw input messages + output of every model call in
# a turn. Attached run-scoped in the graph stream config (agent.run_turn); it rides LangChain's
# contextvar callback propagation down into each node's model.invoke()/stream(), so it sees the
# planner, agent, judge, and synthesizer calls without any node having to thread it through. Read
# back by `/trace invoke`.

_LLM_MSG_CAP = 8000  # per-message content cap stored to the DB (the display truncates further)

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402  (core dep; safe to import)


def _msg_to_dict(m) -> dict:
    """Serialize one input message to a compact role/content dict, folding in a tool-call decision
    or a tool result's name so the recorded input is faithful to what the model actually saw."""
    role = type(m).__name__.replace("Message", "").lower() or "msg"
    content = m.content
    if isinstance(content, list):
        content = " ".join(str(p) for p in content)
    content = str(content)
    d: dict = {"role": role, "content": content[:_LLM_MSG_CAP]}
    if len(content) > _LLM_MSG_CAP:
        d["truncated"] = len(content)
    calls = getattr(m, "tool_calls", None)
    if calls:
        d["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    name = getattr(m, "name", None)
    if name:
        d["name"] = name
    return d


def _llm_output(response) -> tuple[dict, int, int]:
    """Pull (output dict, prompt_tokens, output_tokens) out of an LLMResult. The output dict is the
    model's text + any tool calls; tokens come from the message's usage_metadata, falling back to
    Ollama's response_metadata eval counts."""
    gens = getattr(response, "generations", None) or []
    msg = None
    text = ""
    if gens and gens[0]:
        g0 = gens[0][0]
        msg = getattr(g0, "message", None)
        text = getattr(g0, "text", "") or (str(getattr(msg, "content", "")) if msg is not None else "")
    tool_calls = []
    ptok = otok = 0
    if msg is not None:
        for c in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append({"name": c.get("name"), "args": c.get("args")})
        usage = getattr(msg, "usage_metadata", None) or {}
        ptok = usage.get("input_tokens") or 0
        otok = usage.get("output_tokens") or 0
        if not (ptok or otok):
            meta = getattr(msg, "response_metadata", None) or {}
            ptok = meta.get("prompt_eval_count") or 0
            otok = meta.get("eval_count") or 0
    text = str(text)
    out = {"content": text[:_LLM_MSG_CAP], "tool_calls": tool_calls}
    if len(text) > _LLM_MSG_CAP:
        # Same convention as _msg_to_dict's input flag: record the ORIGINAL length so the
        # /trace invoke renderer can disclose the recording cut — without it, --full presents
        # a capped output as the model's complete reply.
        out["truncated"] = len(text)
    return out, int(ptok or 0), int(otok or 0)


def _extract_model(serialized, metadata, kwargs) -> str:
    """Best-effort model id for a call, across the metadata / invocation_params / serialized shapes."""
    md = metadata or {}
    if md.get("ls_model_name"):
        return str(md["ls_model_name"])
    inv = kwargs.get("invocation_params") or {}
    for k in ("model", "model_name", "model_id"):
        if inv.get(k):
            return str(inv[k])
    kw = (serialized or {}).get("kwargs") or {}
    for k in ("model", "model_name", "model_id"):
        if kw.get(k):
            return str(kw[k])
    return "?"


class LLMTraceHandler(BaseCallbackHandler):
    """Captures each model call's input messages + output into the trace DB, keyed by the turn's
    run_id. Correlates start↔end by the per-call run UUID LangChain passes to both. Every callback
    is wrapped so a capture failure can never disturb the model call it's observing."""

    def __init__(self, tracer: "Tracer", run_id: int):
        self._tracer = tracer
        self._run_id = run_id
        self._pending: dict = {}  # call run_uuid -> {start, node, model, input, ts}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs):
        try:
            node = (metadata or {}).get("langgraph_node") or "?"
            model = _extract_model(serialized, metadata, kwargs)
            flat = messages[0] if (messages and isinstance(messages[0], list)) else (messages or [])
            self._pending[run_id] = {
                "start": perf_counter(),
                "node": node,
                "model": model,
                "input": [_msg_to_dict(m) for m in flat],
            }
        except Exception as exc:
            import diag
            diag.log(f"LLMTraceHandler.on_chat_model_start failed: {exc}")

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        rec = self._pending.pop(run_id, None)
        if rec is None:
            return
        try:
            out, ptok, otok = _llm_output(response)
            self._tracer.log_llm_call(
                self._run_id, rec["node"], rec["model"], perf_counter() - rec["start"],
                ptok, otok, json.dumps(rec["input"], default=str), json.dumps(out, default=str), "ok",
            )
        except Exception as exc:
            import diag
            diag.log(f"LLMTraceHandler.on_llm_end failed: {exc}")

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        rec = self._pending.pop(run_id, None)
        if rec is None:
            return
        try:
            self._tracer.log_llm_call(
                self._run_id, rec["node"], rec["model"], perf_counter() - rec["start"],
                0, 0, json.dumps(rec["input"], default=str),
                json.dumps({"content": "", "tool_calls": [], "error": str(error)}), "error",
            )
        except Exception as exc:
            import diag
            diag.log(f"LLMTraceHandler.on_llm_error failed: {exc}")
