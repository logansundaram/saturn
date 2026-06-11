"""
Egress ledger + air-gap enforcement — the network boundary made visible and verifiable.

The product's privacy proof point used to be *asserted* (`/privacy` lists what CAN leave this
machine) but never *proven* (what actually left). This module closes that gap. It is the single
chokepoint every outbound network operation reports through, so "nothing leaves your machine"
becomes an auditable fact rather than a slogan:

  - `record(...)`     every successful egress (a web search, an http_request, a remote MCP call,
                      a cloud-model invocation) appends one `EgressEvent` to a process-wide,
                      append-only ledger. `/privacy egress` renders it; the status bar shows a
                      live count.
  - `check(...)`      the air-gap gate. When `runtime.airgap` is on, an outbound op calls this
                      FIRST; it records a `blocked` event and returns a refusal string the caller
                      hands back instead of touching the network. Air-gap turns the privacy claim
                      from a promise into something the machine enforces.

Air-gap is read live from `runtime.airgap` (toggled by `/privacy airgap`), exactly like the budget
and auto-approve knobs — so flipping it applies to the very next op. Cloud LLM egress is enforced
separately in `llms.get_model` (it raises rather than returning a string, since a node can't run
without its model); the `/privacy airgap` command drops the model cache so a cached cloud model
can't sneak a call through.

The ledger is per-process (one Saturn session), like `budget.py` — a live boundary monitor, not a
durable audit log (that is `/trace export`). Imports only leaves (config, diag, signing, textutil),
so any module (web tools, mcp_client, llms) can import it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import diag
import signing
from config import get_config
from textutil import truncate

# Hard cap on retained events so a long session can't grow the ledger without bound (oldest drop).
_MAX_EVENTS = 5000

# A stable id for this process's session, stamped into every disk-logged event so the durable log
# can attribute each egress to the run it happened in. (Plain module-level datetime/pid — egress.py
# is an ordinary module, not a workflow script, so wall-clock is available here.)
_SESSION_ID = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

# Egress statuses, for display + filtering.
SENT = "sent"        # left the machine
BLOCKED = "blocked"  # air-gap refused it before anything was sent


@dataclass(frozen=True)
class EgressEvent:
    """One outbound network operation (or one air-gap refusal). `channel` is the kind of egress
    (web_search/web_extract/http_request/mcp/llm), `host` where it went, `detail` a short human
    label (the query, the URL, the model id), `provider` the backend when relevant, `n_bytes` the
    approximate size of what was SENT, `redactions` how many secrets were stripped first. `seq` is
    the session-wide ordinal (monotonic, never reused) — turn slices key on it, not list indexes,
    so the cap-trim and `clear()` can't shift a mark onto the wrong events."""

    ts: str
    channel: str
    host: str
    detail: str = ""
    provider: str = ""
    n_bytes: int = 0
    redactions: int = 0
    status: str = SENT
    seq: int = 0


_LEDGER: list[EgressEvent] = []
_SEQ = 0  # last seq handed out; survives clear() so turn-start marks stay valid


def airgap_on() -> bool:
    """Whether the air-gap is engaged (`runtime.airgap`). Read live so a toggle applies at once."""
    return bool(get_config().get("runtime.airgap", False))


def _host_label(host: str) -> str:
    return (host or "?").strip() or "?"


def _safe_int(v) -> int:
    try:
        n = int(v)
        return n if n > 0 else 0
    except (TypeError, ValueError):
        return 0


def record(channel: str, host: str, detail: str = "", *, provider: str = "",
           n_bytes: int = 0, redactions: int = 0, status: str = SENT) -> None:
    """Append one egress event to the ledger. Best-effort and crash-proof: a junk field is coerced
    to a safe default rather than dropping the event — losing the RECORD that something left the
    machine is the one failure a boundary ledger must never have. `host`/`detail` are display
    labels, so they are clipped here: an unbounded detail (a fat model-generated URL or query)
    would bloat every render AND could push a durable-log line past the tail window the chain
    appender reads."""
    global _SEQ
    try:
        ev = EgressEvent(
            ts=datetime.now().isoformat(),
            channel=str(channel),
            host=truncate(_host_label(str(host) if host is not None else ""), 200),
            detail=truncate(str(detail or ""), 500),
            provider=str(provider or ""),
            n_bytes=_safe_int(n_bytes),
            redactions=_safe_int(redactions),
            status=str(status or SENT),
            seq=_SEQ + 1,
        )
    except Exception:
        return
    _SEQ += 1
    _LEDGER.append(ev)
    if len(_LEDGER) > _MAX_EVENTS:
        del _LEDGER[: len(_LEDGER) - _MAX_EVENTS]
    _append_disk(ev)


# ── persistent, hash-chained egress log (runtime.egress_log) ──────────────────────────────────
# The in-memory ledger above is per-process — a live boundary monitor. This is its durable twin: an
# append-only JSONL file at paths.egress_log where "what left this machine" survives across sessions
# and is TAMPER-EVIDENT. Each line carries `prev` (the previous line's hash) and `h` = sha256(prev +
# canonical(payload)); a walk of the chain (verify_log) detects any edit, reorder, or deletion of
# the middle. Truncating the tail is the one thing a self-contained log can't prove against — that
# needs an external anchor (the signed /trace export plays that role for the run record). Writing is
# strictly best-effort: losing the live RECORD must never break an op, and neither must the disk log.


def _log_path() -> "Path | None":
    try:
        return get_config().path("egress_log")
    except Exception:
        return None


def _disk_enabled() -> bool:
    return bool(get_config().get("runtime.egress_log", True))


def _entry_hash(prev: str, payload: dict) -> str:
    # signing.canonical_json is THE canonical byte stream every Saturn digest commits to — the
    # chain must use the same one as trace exports / trust reports so the schemes can never drift.
    return hashlib.sha256((prev + signing.canonical_json(payload)).encode("utf-8")).hexdigest()


def _lock_handle(fh) -> bool:
    """Best-effort exclusive lock on the open log handle across the read-tip→append window, so two
    processes (an interactive session + a headless cron run) can't both chain from the same tip and
    fork the log. Bounded non-blocking retries; if the lock can't be had, the caller SKIPS the
    durable append (diag-logged) — appending unlocked would chain from an unverified tip, and a
    forked chain makes verify_log report tampering forever (a permanent false alarm is worse than
    one missing line; the in-memory ledger still carries the event)."""
    for _ in range(5):
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            time.sleep(0.02)
        except Exception:
            return False
    return False


def _unlock_handle(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


def _tail_tip(fh, size: int) -> str:
    """The `h` of the log's last line (the chain tip to extend), read from the open handle.
    Reads a GROWING tail window: a single record longer than the initial window (e.g. lines
    written before detail clipping existed) must not be parsed as a mid-line fragment — that
    would silently restart the chain with prev="" and break verification forever. The window is
    trustworthy only when it spans the whole file or contains a newline before the tip line."""
    window = 8192
    while True:
        fh.seek(max(0, size - window))
        tail = fh.read(min(size, window)).decode("utf-8", "replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if window >= size or len(lines) > 1:
            break
        window *= 8
    if not lines:
        return ""
    try:
        return str(json.loads(lines[-1]).get("h", "")) or ""
    except Exception:
        return ""


# (path, file size, tip hash) after our last append — steady-state appends skip the tail re-read;
# a size mismatch (another session appended) falls back to reading the real tip from disk.
_TIP_CACHE: "tuple[str, int, str] | None" = None


def _append_disk(ev: EgressEvent) -> None:
    """Append one event to the durable hash-chained log. Best-effort: any failure is swallowed
    (logged to diag) so the egress op it accompanies is never disturbed. The handle is locked
    across read-tip→append so concurrent sessions extend one linear chain; if the lock can't be
    had the append is SKIPPED (see _lock_handle) — never written unlocked from a stale tip."""
    global _TIP_CACHE
    if not _disk_enabled():
        return
    path = _log_path()
    if path is None:
        return
    try:
        payload = asdict(ev)
        payload.pop("seq", None)  # session-local ordinal, not part of the durable record
        payload["session"] = _SESSION_ID
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+b") as fh:
            if not _lock_handle(fh):
                # Another process holds the lock past the retry budget. Never append unlocked:
                # chaining from a tip that writer is about to move would FORK the hash chain and
                # turn every future verify into a permanent false tamper alarm. Drop this line
                # (the in-memory ledger keeps the event) and let the next append re-anchor.
                diag.log("egress: disk append skipped — log locked by another session")
                return
            try:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    prev = ""
                elif _TIP_CACHE and _TIP_CACHE[0] == str(path) and _TIP_CACHE[1] == size:
                    prev = _TIP_CACHE[2]
                else:
                    prev = _tail_tip(fh, size)
                payload["prev"] = prev
                payload["h"] = _entry_hash(
                    prev, {k: v for k, v in payload.items() if k not in ("prev", "h")}
                )
                fh.seek(0, os.SEEK_END)
                fh.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                fh.flush()
                _TIP_CACHE = (str(path), fh.tell(), payload["h"])
            finally:
                _unlock_handle(fh)
    except Exception as exc:
        diag.log(f"egress: disk append failed: {exc}")


def read_log(limit: "int | None" = None) -> list[dict]:
    """Parsed lines of the durable log, oldest first (a copy). Empty when logging is off or the file
    doesn't exist yet. `limit` keeps only the most recent N — and reads only a growing TAIL window
    of the file for them (the log is append-only and never trimmed, so a full read for the last 30
    rows would grow without bound across months of sessions)."""
    path = _log_path()
    if path is None or not path.exists():
        return []
    try:
        if limit:
            text = _read_tail_lines(path, limit)
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as exc:
        diag.log(f"egress: read_log failed: {exc}")
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:] if limit else rows


def _read_tail_lines(path, n_lines: int) -> str:
    """The last `n_lines`+ lines of `path` as text, via a growing tail window (same technique as
    _tail_tip). The window is trustworthy once it spans the whole file or starts past a newline —
    a partial first line is dropped rather than parsed as a mid-line fragment."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        window = max(8192, n_lines * 512)
        while True:
            fh.seek(max(0, size - window))
            tail = fh.read(min(size, window)).decode("utf-8", "replace")
            whole = window >= size
            if whole or tail.count("\n") > n_lines:
                break
            window *= 8
    if whole:
        return tail
    # Drop the (possibly partial) first line of a mid-file window.
    return tail.split("\n", 1)[1] if "\n" in tail else ""


def verify_log() -> dict:
    """Walk the durable log's hash chain and report integrity. Returns a dict:
      {available, exists, lines, ok, broken_at, sessions, error}
    `ok` is True iff EVERY raw line parses, recomputes its hash, and links to its predecessor.
    Unlike the display path (read_log), this walks the raw lines and treats an unparseable one as
    a broken chain — silently skipping it would let a garbled tail line (or a file replaced
    wholesale with garbage) verify as '✓ intact', exactly the tamper this log exists to expose.
    `broken_at` is the 1-based index of the first bad line."""
    path = _log_path()
    if path is None:
        return {"available": False, "exists": False, "ok": True, "lines": 0}
    if not path.exists():
        return {"available": True, "exists": False, "ok": True, "lines": 0, "sessions": []}

    try:
        raw = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as exc:
        return {"available": True, "exists": True, "ok": False, "broken_at": 1, "lines": 0,
                "error": f"unreadable: {exc}", "sessions": []}

    prev = ""
    sessions: list[str] = []

    def _broken(i: int, why: str) -> dict:
        return {"available": True, "exists": True, "ok": False, "broken_at": i, "error": why,
                "lines": len(raw), "sessions": sorted(set(sessions))}

    for i, line in enumerate(raw, start=1):
        try:
            row = json.loads(line)
        except Exception:
            return _broken(i, "unparseable line")
        if not isinstance(row, dict):
            return _broken(i, "not a record")
        payload = {k: v for k, v in row.items() if k not in ("prev", "h")}
        if row.get("prev", "") != prev or row.get("h") != _entry_hash(prev, payload):
            return _broken(i, "hash mismatch")
        s = row.get("session")
        if s:
            sessions.append(s)
        prev = row["h"]
    return {
        "available": True, "exists": True, "ok": True, "broken_at": None,
        "lines": len(raw), "sessions": sorted(set(sessions)),
    }


def log_summary(rows: "list[dict] | None" = None) -> dict:
    """Aggregate the DURABLE (cross-session) log for `/privacy egress log` and the trust report.
    Pass pre-read `rows` (from read_log()) to avoid a second full-file read when the caller
    already has them."""
    if rows is None:
        rows = read_log()
    sent = [r for r in rows if r.get("status") == SENT]
    blocked = [r for r in rows if r.get("status") == BLOCKED]
    hosts = sorted({r.get("host", "?") for r in sent})
    sessions = sorted({r.get("session", "") for r in rows if r.get("session")})
    return {
        "lines": len(rows),
        "sent": len(sent),
        "blocked": len(blocked),
        "bytes": sum(_safe_int(r.get("n_bytes")) for r in sent),
        "redactions": sum(_safe_int(r.get("redactions")) for r in sent),
        "hosts": hosts,
        "sessions": sessions,
        "first": rows[0].get("ts") if rows else None,
        "last": rows[-1].get("ts") if rows else None,
    }


def blocked_message(host: str, channel: str = "") -> str:
    """The refusal string a network tool returns to the model when air-gap blocks its op."""
    where = f" to {host}" if host and host != "?" else ""
    what = f" ({channel})" if channel else ""
    return (
        f"Air-gap is ON — this operation{what} would send data{where} over the network, which is "
        "currently blocked. Nothing was sent. The user can allow network access with "
        "`/privacy airgap off`."
    )


def check(channel: str, host: str, detail: str = "") -> "str | None":
    """Air-gap gate for a network op. Returns None when egress is allowed; when air-gap is on,
    records a `blocked` event and returns the refusal string for the caller to hand back (tools
    return it to the model as their observation)."""
    if airgap_on():
        record(channel, host, detail, status=BLOCKED)
        return blocked_message(_host_label(host), channel)
    return None


def events() -> list[EgressEvent]:
    """The ledger, oldest first (a copy — callers may filter/slice freely)."""
    return list(_LEDGER)


def next_seq() -> int:
    """The seq the NEXT recorded event will carry — capture at turn start, hand to events_since.
    Unlike a list index, a seq mark stays valid across the cap-trim and clear()."""
    return _SEQ + 1


def events_since(mark: int) -> list[EgressEvent]:
    """Events recorded at or after seq `mark`, oldest first. Seq-keyed (never an index into the
    ledger) so the _MAX_EVENTS trim or a mid-session `/privacy egress clear` can't shift a
    turn-start mark onto the wrong slice — the trust receipt must never read 'local-only' over a
    turn that actually sent."""
    out: list[EgressEvent] = []
    for e in reversed(_LEDGER):
        if e.seq < mark:
            break
        out.append(e)
    out.reverse()
    return out


def count() -> int:
    """Number of egress events recorded this session (for the status-bar indicator)."""
    return len(_LEDGER)


def summarize_events(events) -> dict:
    """Aggregate one slice of EgressEvents — THE one accounting every per-slice trust surface
    uses (the per-answer receipt, the Glass Box, the `/privacy egress` headline), so they can
    never report different byte/host numbers for the same events. Returns
    {sent, blocked, bytes, redactions, hosts (first-seen order), channels (sent, first-seen)}."""
    sent = [e for e in events if getattr(e, "status", "") == SENT]
    blocked = [e for e in events if getattr(e, "status", "") == BLOCKED]
    hosts: list[str] = []
    channels: list[str] = []
    for e in sent:
        h = getattr(e, "host", "?")
        if h not in hosts:
            hosts.append(h)
        c = getattr(e, "channel", "")
        if c and c not in channels:
            channels.append(c)
    return {
        "sent": len(sent),
        "blocked": len(blocked),
        "bytes": sum(_safe_int(getattr(e, "n_bytes", 0)) for e in sent),
        "redactions": sum(_safe_int(getattr(e, "redactions", 0)) for e in sent),
        "hosts": hosts,
        "channels": channels,
    }


def summary() -> dict:
    """Aggregate the ledger for the `/privacy egress` headline: totals, bytes, distinct hosts,
    blocked."""
    agg = summarize_events(_LEDGER)
    by_channel: dict[str, int] = {}
    for e in _LEDGER:
        if e.status == SENT:
            by_channel[e.channel] = by_channel.get(e.channel, 0) + 1
    return {
        "total": len(_LEDGER),
        "sent": agg["sent"],
        "blocked": agg["blocked"],
        "bytes": agg["bytes"],
        "redactions": agg["redactions"],
        "hosts": agg["hosts"],
        "by_channel": by_channel,
    }


def clear() -> None:
    """Empty the ledger (a deliberate operator reset via `/privacy egress clear`). The seq counter
    is NOT reset — outstanding turn-start marks must keep pointing past the cleared events, not
    get re-matched against new ones. The clear itself is remembered (cleared_since) so a per-turn
    consumer (the Glass Box) can tell an empty slice from a clear-emptied one instead of reporting
    'local-only' over a turn whose events were wiped."""
    global _CLEARED_AT
    _LEDGER.clear()
    _CLEARED_AT = _SEQ


_CLEARED_AT = 0  # highest seq wiped by clear(); 0 = never cleared


def cleared_since(mark: int) -> bool:
    """Whether a clear() has wiped events at/after seq `mark` — i.e. whether events_since(mark)
    may be missing events that really happened. A slice that may have been clear-emptied must be
    treated as UNKNOWN by truth-claiming surfaces, never as 'nothing was sent'."""
    return _CLEARED_AT >= mark > 0
