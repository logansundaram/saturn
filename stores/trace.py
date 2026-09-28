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
# reader (show_run's label, the export), so they can't drift.
_RESPONSE_TRUNCATION_MARKER = "… [recorded answer truncated at "


def response_truncated(text) -> bool:
    """True when a recorded `runs.response` carries end_run's write-time truncation marker.
    Readers treat a marked row as INCOMPLETE (show_run says "truncated", the export
    reconstruction passes complete=False). Historical rows cut at the old 2000-char cap carry no
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
-- Every /trace read selects `WHERE run_id = ?`, and the run listing counts events per run with a
-- correlated subquery — both are full table scans without these. The tables only grow (runs are
-- retained on purpose; the checkpointer's threads are what get pruned), so the scans get slower
-- for the life of the install.
CREATE INDEX IF NOT EXISTS ix_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_run ON llm_calls(run_id);
"""

# Full-text index over the runs the agent may search (`recall_runs`, `/trace search`). SQLite's
# FTS5 ships in every CPython wheel we target, but a distro build can omit it, so this is applied
# separately from _SCHEMA and its absence degrades to a LIKE scan (search_runs) — never a failed
# tracer. External-content table: the runs row stays the record; the index is rebuilt from it
# when first created, and the triggers keep it current from then on.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(
    query, response, content='runs', content_rowid='run_id'
);
CREATE TRIGGER IF NOT EXISTS runs_fts_ai AFTER INSERT ON runs BEGIN
  INSERT INTO runs_fts(rowid, query, response) VALUES (new.run_id, new.query, new.response);
END;
CREATE TRIGGER IF NOT EXISTS runs_fts_ad AFTER DELETE ON runs BEGIN
  INSERT INTO runs_fts(runs_fts, rowid, query, response)
    VALUES ('delete', old.run_id, old.query, old.response);
END;
CREATE TRIGGER IF NOT EXISTS runs_fts_au AFTER UPDATE ON runs BEGIN
  INSERT INTO runs_fts(runs_fts, rowid, query, response)
    VALUES ('delete', old.run_id, old.query, old.response);
  INSERT INTO runs_fts(rowid, query, response) VALUES (new.run_id, new.query, new.response);
END;
"""


_FTS_TRIGGERS = ("runs_fts_ai", "runs_fts_ad", "runs_fts_au")


def ensure_fts(conn) -> bool:
    """Create the runs full-text index (and backfill it from the existing rows when the table or
    its triggers were missing). Returns True when FTS5 is usable on this connection. Best-effort:
    a build without FTS5 returns False and search_runs falls back to LIKE.

    The triggers are the hazard: they live in the DB file, and an INSERT on `runs` compiles them
    — on a build WITHOUT fts5 a trigger left behind by a build WITH it fails every start_run with
    "no such module: fts5". So when the probe fails, the triggers are dropped (plain DDL, no
    module needed) and the tracer keeps recording; the next fts5-capable open recreates them and
    rebuilds the index from the rows written meanwhile."""
    try:
        have = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?, ?)",
            ("runs_fts", *_FTS_TRIGGERS)).fetchall()}
        # Probe the module before touching the schema: a virtual table whose module is missing
        # still shows in sqlite_master, and the CREATE ... IF NOT EXISTS below would not notice.
        conn.execute("SELECT count(*) FROM runs_fts" if "runs_fts" in have
                     else "CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5("
                          "query, response, content='runs', content_rowid='run_id')")
        conn.executescript(_FTS_SCHEMA)
        if not have.issuperset({"runs_fts", *_FTS_TRIGGERS}):
            conn.execute("INSERT INTO runs_fts(runs_fts) VALUES ('rebuild')")
        conn.commit()
        return True
    except Exception:
        try:
            for trig in _FTS_TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            conn.commit()
        except Exception:
            pass
        return False


# Filler the natural-language queries the tool is advertised for carry ("what did we do last
# week"): dropped before matching so they never veto a hit. A query that is ALL filler has
# nothing to search for and matches nothing (never a false hit on "what").
_SEARCH_STOPWORDS = frozenset("""
the and for with that this from what when where which who how are was were will would can
could should did does done do into onto about after before over under then than them they
there here have has had not but all any some our your their its his her you we us me my it
is be been being on in at to of by or as an a if so up out no yes please make give find show
tell let get use using run take need want like just also last week month ago yesterday today
""".split())


def _search_terms(text: str) -> list[str]:
    import re

    words = [t for t in re.findall(r"[\w'\-]+", str(text or "")) if len(t) > 1]
    return [t for t in words if t.lower() not in _SEARCH_STOPWORDS][:12]


def search_runs(db_path, text: str, limit: int = 5) -> list[dict]:
    """Past runs whose query or recorded answer matches `text`, newest-relevant first:
    `[{run_id, started_at, status, query, response}]`. Stopwords are dropped, then every
    content term must match (AND); when nothing does, runs matching ANY term are returned
    instead, best match first — so "the report we made Monday" still finds the report. FTS5
    (bm25-ranked) when the index exists, else a LIKE scan. Terms are quoted individually so user
    text can never inject FTS syntax, and `%`/`_` are escaped in the LIKE path. Read-side helper
    (its own short-lived connection) — shared by the `recall_runs` tool and `/trace search`."""
    terms = _search_terms(text)
    if not terms:
        return []
    limit = max(1, min(int(limit or 5), 50))
    conn = sqlite3.connect(str(db_path))
    try:
        rows = None
        if ensure_fts(conn):
            quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
            try:
                for match in (" ".join(quoted), " OR ".join(quoted)):
                    rows = conn.execute(
                        "SELECT r.run_id, r.started_at, r.status, r.query, r.response "
                        "FROM runs_fts f JOIN runs r ON r.run_id = f.rowid "
                        "WHERE runs_fts MATCH ? ORDER BY bm25(runs_fts), r.run_id DESC LIMIT ?",
                        (match, limit),
                    ).fetchall()
                    if rows or len(quoted) == 1:
                        break
            except sqlite3.Error:
                rows = None
        if rows is None:
            clause = ("(LOWER(COALESCE(query,'')) LIKE ? ESCAPE '\\' "
                      "OR LOWER(COALESCE(response,'')) LIKE ? ESCAPE '\\')")
            params: list = []
            for t in terms:
                like = "%" + t.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                params += [like, like]
            for joiner in (" AND ", " OR "):
                rows = conn.execute(
                    f"SELECT run_id, started_at, status, query, response FROM runs "
                    f"WHERE {joiner.join(clause for _ in terms)} ORDER BY run_id DESC LIMIT ?",
                    (*params, limit),
                ).fetchall()
                if rows or len(terms) == 1:
                    break
    finally:
        conn.close()
    return [
        {"run_id": rid, "started_at": started, "status": status, "query": query or "",
         "response": response or ""}
        for rid, started, status, query, response in rows
    ]


# The run the live turn is recording into, for provenance stamps made from inside a tool
# (`remember` writes `run=<id>` on the fact it stores, so `/memory why` can point at the run).
# Set by start_run, cleared by end_run; None between turns and when the trace could not open a
# run (the -1 sentinel never leaks as a provenance id).
_CURRENT_RUN_ID = None


def current_run_id():
    return _CURRENT_RUN_ID


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

# Entries kept from an oversized list of NON-string leaves (a long list of small numeric dicts,
# which map_strings cannot shrink by a byte): without this rung the halving string ladder below
# spins to its floor without progress and per-key salvage drops the whole key. A shortened list
# with the loss NAMED beats losing the key entirely.
_LIST_CAP = 400


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
        data = _bound_delta(data, len(data))
    return summary, data


def _thin_lists(obj, cap: int, dropped: list, path: str = ""):
    """Truncate every list longer than `cap`, naming each loss in `dropped`. Entries are dropped
    from the TAIL and nothing is substituted in their place — a marker entry inside the list would
    reach consumers that read `.get("start")` off every element. The loss is recorded once, at the
    top level, in the same `truncated` record the rest of this ladder uses."""
    if isinstance(obj, dict):
        return {k: _thin_lists(v, cap, dropped, f"{path}.{k}" if path else str(k))
                for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_thin_lists(v, cap, dropped, f"{path}[]") for v in obj]
        if len(items) > cap:
            dropped.append(f"{path or 'root'}[{len(items) - cap} of {len(items)} entries]")
            return items[:cap]
        return items
    return obj


def _bound_delta(data: str, original: int) -> str:
    """Bring an oversized delta under _DATA_CAP while keeping it PARSEABLE (transplanted from
    the visibility isolate). Clip long string LEAVES (head+tail, marker inside the text) with a
    cap that halves until the JSON fits; a delta that still overflows (a plan with hundreds of
    steps, a thousand tiny messages) keeps every key that fits on its own and replaces the rest
    with an explicit `truncated` record naming what was dropped and the original size. Never a
    slice of the JSON text: a mid-token cut stores an undecodable blob — decode_json -> default,
    the whole delta (tool events, the plan update) silently gone from /trace replay, `data: null`
    in an export — INCOMPLETE for the wrong reason."""
    note = f"delta exceeded the {_DATA_CAP}-char record cap at write time"
    try:
        obj = json.loads(data)
    except Exception:
        # json.dumps produced `data`, so this is belt and braces.
        return json.dumps({"truncated": {"original_chars": original, "dropped": ["*"],
                                         "note": "delta could not be re-encoded"}})
    def _string_ladder(o):
        """Halve the per-string-leaf cap until the encoding fits; None if it never does."""
        cap = _LEAF_CAP
        while cap >= 50:
            try:
                clipped = json.dumps(map_strings(o, lambda s, c=cap: head_tail(s, c)))
            except Exception:
                return None
            if len(clipped) <= _DATA_CAP:
                return clipped
            cap //= 2
        return None

    fitted = _string_ladder(obj)
    if fitted is not None:
        return fitted
    # Clipping strings could not do it, so the bulk is in NON-string leaves. Thin the long
    # lists before falling back to dropping whole keys, and re-run the string ladder at each
    # rung, so text is only clipped as hard as that rung actually needs.
    if isinstance(obj, dict):
        list_cap = _LIST_CAP
        while list_cap >= 25:
            thinned_drops: list = []
            try:
                thinned = _thin_lists(obj, list_cap, thinned_drops)
                if thinned_drops:
                    thinned = {**thinned, "truncated": {"original_chars": original,
                                                        "dropped": thinned_drops, "note": note}}
            except Exception:
                break
            fitted = _string_ladder(thinned)
            if fitted is not None:
                return fitted
            list_cap //= 2
    # Per-key salvage: keep the keys that fit, drop the rest with a marker.
    if isinstance(obj, dict):
        kept: dict = {}
        dropped = []
        budget = _DATA_CAP - 200  # room for the marker itself
        for k, v in obj.items():
            try:
                small = map_strings(v, lambda s: head_tail(s, 100))
                piece = json.dumps({k: small})
            except Exception:
                dropped.append(str(k))
                continue
            if len(piece) <= budget:
                kept[k] = small
                budget -= len(piece)
            else:
                dropped.append(str(k))
        kept["truncated"] = {"original_chars": original, "dropped": dropped, "note": note}
        out = json.dumps(kept)
        if len(out) <= _DATA_CAP + 400:
            return out
    return json.dumps({"truncated": {"original_chars": original, "dropped": ["*"], "note": note}})


class Tracer:
    """Every write is BEST-EFFORT: the watcher must never take down the watched. log_event runs
    on every node delta of a live turn, so a sqlite failure here (db.sqlite locked by a second
    instance / an open DB browser — it's shared with SqliteSaver — or disk full) would otherwise
    raise out of run_turn's stream loop and report a healthy turn as failed."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # The trace is a best-effort observability log, not the durable record the answer depends
        # on — a torn tail after an OS-level crash costs one turn's events, while FULL's per-commit
        # fsync costs ~25ms on every node delta of every turn. (journal_mode is already WAL:
        # SqliteSaver sets it and it persists on the file.)
        try:
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:  # a pragma refusal must never cost the tracer its connection
            pass
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        ensure_fts(self.conn)  # best-effort; absent FTS5 degrades to LIKE
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
        global _CURRENT_RUN_ID
        self._seq = 0
        self._llm_seq = 0
        self._broken = False  # re-arm the breaker: one retry per turn, never a permanently dead trace
        try:
            cur = self.conn.execute(
                "INSERT INTO runs (thread_id, query, started_at, status) VALUES (?, ?, ?, ?)",
                (thread_id, query, datetime.now().isoformat(), "running"),
            )
            self.conn.commit()
            _CURRENT_RUN_ID = cur.lastrowid
            return cur.lastrowid
        except Exception as exc:
            self._trip("start_run", exc)
            _CURRENT_RUN_ID = None
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
        global _CURRENT_RUN_ID
        _CURRENT_RUN_ID = None
        text = response or ""
        # The recorded answer is capped like a delta (_DATA_CAP — it IS the headline record every
        # after-the-fact surface reads: show_run, the export).
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
        # recorded response (/trace and exports both read that row). Worst case one
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
# agent and utility calls without any node having to thread it through. Read
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
            # A GeneratorExit is not a model failure: the CONSUMER closed the stream on purpose
            # (the freeze latch breaking out of synthesize's loop, a cancelled turn) — langchain's
            # stream wrapper routes it here before re-raising. Record it as `cancelled`, not
            # `error` (whose message would be the blank str(GeneratorExit())), or /trace invoke
            # misreports every frozen interrupt-and-correct turn as a failed synthesize call.
            cancelled = isinstance(error, GeneratorExit)
            note = "stream closed before completion (freeze/cancel)" if cancelled else str(error)
            self._tracer.log_llm_call(
                self._run_id, rec["node"], rec["model"], perf_counter() - rec["start"],
                0, 0, json.dumps(rec["input"], default=str),
                json.dumps({"content": "", "tool_calls": [], "error": note}),
                "cancelled" if cancelled else "error",
            )
        except Exception as exc:
            import diag
            diag.log(f"LLMTraceHandler.on_llm_error failed: {exc}")
