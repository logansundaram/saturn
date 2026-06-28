"""
Egress ledger + air-gap enforcement — the network boundary made visible.

The product's privacy proof point used to be *asserted* (`/privacy` lists what CAN leave this
machine) but never *shown* (what actually left). This module closes that gap. It is the single
chokepoint every outbound network operation reports through, so "nothing leaves your machine"
becomes an observable fact rather than a slogan:

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

The ledger is per-process (one Saturn session), like `budget.py` — a live boundary monitor.

This module also owns the inference-locality classifier (`_inference` + its display companions):
"where do the words come from" is fundamentally an egress question, and this is where the loopback
test (`ollama_is_local`) already lives. The posture line, `/privacy`, and the Glass Box all read
the one classifier here. Imports only leaves (config, diag, textutil), so any module (web tools,
mcp_client, llms, the TUI) can import it without a cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from config import MODEL_ROLES, get_config
from textutil import truncate

# Hard cap on retained events so a long session can't grow the ledger without bound (oldest drop).
_MAX_EVENTS = 5000

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


def ollama_endpoint() -> str:
    """The Ollama base URL the client libraries will actually talk to: `OLLAMA_HOST` when set
    (the one binding — nothing in config.yaml names it; ChatOllama/OllamaEmbeddings/ollama.list
    all read the same env var), else the daemon's local default."""
    return (os.environ.get("OLLAMA_HOST") or "").strip() or "http://127.0.0.1:11434"


def ollama_is_local() -> bool:
    """Whether Ollama traffic stays on this machine. The whole "local inference" story keys on
    this: an `OLLAMA_HOST` pointing off-machine makes the "local" models network egress like any
    cloud provider — recorded in the ledger, refused under air-gap, and disqualifying for the
    local-inference claim. Fails toward NOT local (an unparseable endpoint must never earn a
    'local' claim)."""
    endpoint = ollama_endpoint()
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    try:
        name = (urlparse(endpoint).hostname or "").lower()
    except Exception:
        return False
    return name in ("localhost", "::1", "0.0.0.0") or name.startswith("127.")


def _host_label(host: str) -> str:
    return (host or "?").strip() or "?"


def host_of(url: str) -> str:
    """Hostname of a URL for the egress ledger (falls back to the raw value). THE one
    derivation of "where did this go" every egress reporter shares (tools/web.py,
    tools/mcp_client.py) — a second copy could drift and label the same destination two ways.
    NOT used by ollama_is_local(): its fallback deliberately fails toward NOT-local
    (empty/False), whereas a ledger label must never be lost, so it falls back to the raw URL."""
    try:
        return urlparse(url).hostname or str(url)
    except Exception:
        return str(url)


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
    would bloat every render."""
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
    blocked. Carries `cleared`: whether a `/privacy egress clear` wiped events this session —
    the counts are then SINCE THE CLEAR, not the whole session, and any truth-claiming consumer
    must disclose that rather than imply an understated total. The same hazard `cleared_since`
    guards for per-turn slices, surfaced here for the whole-ledger aggregation."""
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
        "cleared": _CLEARED_AT > 0,
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


# ── inference-locality classifier ────────────────────────────────────────────────────────────────
# "Where do the words come from" — local (computed on this machine) vs off-machine (a cloud
# provider, or an Ollama daemon behind a remote OLLAMA_HOST). THE one classifier: the session
# posture line (receipt.posture_spans), `/privacy`, and the Glass Box all read this — never
# re-rolled. Lives here because locality IS an egress question and the loopback test
# (ollama_is_local) already lives in this module.


def _inference() -> dict:
    """Local-vs-off-machine binding map. 'local' means the words are computed ON THIS MACHINE: an
    Ollama binding only earns it when the endpoint is loopback — a remote OLLAMA_HOST is network
    inference and classifies 'remote' (off-machine, like 'cloud'), reported with the endpoint so
    the reader can see where."""
    cfg = get_config()
    ollama_local = ollama_is_local()
    ollama_loc = "local" if ollama_local else "remote"
    bindings = []
    for role in MODEL_ROLES:
        try:
            spec = cfg.model_for_role(role)
        except KeyError:
            continue
        bindings.append({
            "role": role,
            "provider": spec.provider,
            "model": spec.model,
            "locality": ollama_loc if spec.provider == "ollama" else "cloud",
        })
    try:
        bindings.append({"role": "embedder", "provider": "ollama",
                         "model": cfg.embedder_model, "locality": ollama_loc})
    except Exception:
        pass
    cloud = sorted({b["provider"] for b in bindings if b["locality"] == "cloud"})
    remote = any(b["locality"] == "remote" for b in bindings)
    out = {"bindings": bindings, "cloud_providers": cloud,
           "all_local": not cloud and not remote}
    if remote:
        out["remote_ollama"] = ollama_endpoint()
    return out


def remote_ollama_label(inf: dict) -> str:
    """The display label for a remote-Ollama destination (`ollama @ <endpoint>`) — one spelling
    for every surface that names it (posture line, /privacy tables, the report render)."""
    return f"ollama @ {inf.get('remote_ollama', '?')}"


def offmachine_destinations(inf: "dict | None" = None) -> list[str]:
    """The off-machine inference destinations as display labels: cloud providers plus a remote
    Ollama endpoint (`remote_ollama_label`). THE one assembly of the where-list — the session
    posture line (receipt.posture_spans), /privacy's verdict, all print this, so they can never
    name different destination sets for the identical posture. Takes the classifier's dict (or
    computes it fresh); empty when everything is local."""
    if inf is None:
        inf = _inference()
    where = list(inf.get("cloud_providers") or [])
    if inf.get("remote_ollama"):
        where.append(remote_ollama_label(inf))
    return where
