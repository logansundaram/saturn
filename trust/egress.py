"""
Egress ledger + air-gap enforcement — the network boundary made visible.

The single chokepoint every outbound network operation reports through, so "nothing leaves
your machine" is an observable fact rather than a slogan:

  - `record(...)`     every successful egress (a web search, a page fetch, a remote MCP call,
                      a remote-Ollama invocation) appends one `EgressEvent` to a process-wide,
                      append-only ledger. `/policy egress` renders it.
  - `UNTRACKED`       the ledger's honest gap: `run_shell`, `run_shortcut` and stdio MCP servers are processes
                      whose network use Saturn cannot observe, so each run is recorded with
                      this status. Under air-gap they are held for the human instead
                      (`policy.airgap_holds`) — the one boundary a string check cannot enforce.
  - `check(...)`      the air-gap gate. When `runtime.airgap` is on, an outbound op calls this
                      FIRST; it records a `blocked` event and returns a refusal string the caller
                      hands back instead of touching the network.

Air-gap is read live from `runtime.airgap` (toggled by `/policy airgap`), so flipping it applies
to the very next op. Exits that cannot hand a refusal string back (a remote-Ollama model or
embedder in core/llms.py) use `check_or_raise`. The ledger is per-process (one Saturn session).

This module also owns the inference-locality classifier (`_inference` + its display companions):
"where do the words come from" is an egress question, and the loopback test (`ollama_is_local`)
lives here. Imports only leaves (config, textutil), so any module (web tools, mcp_client, llms,
the TUI) can import it without a cycle.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from config import get_config
from textutil import truncate

# Hard cap on retained events so a long session can't grow the ledger without bound (oldest drop).
_MAX_EVENTS = 5000

# Egress statuses, for display + filtering.
SENT = "sent"        # left the machine
BLOCKED = "blocked"  # air-gap refused it before anything was sent
# A process Saturn cannot see inside ran — a shell command, a stdio MCP server. It may have used
# the network; nothing here can say. Never counted as a send, never left off the ledger: a turn
# that ran one must not read as "nothing left this machine".
UNTRACKED = "untracked"


@dataclass(frozen=True)
class EgressEvent:
    """One outbound network operation (or one air-gap refusal). `channel` is the kind of egress
    (web_search/web_extract/mcp/llm/embedding), `host` where it went, `detail` a short human
    label (the query, the URL, the model id), `provider` the backend when relevant, `n_bytes` the
    approximate size of what was SENT. `seq` is
    the session-wide ordinal (monotonic, never reused) — turn slices key on it, not list indexes,
    so the cap-trim and `clear()` can't shift a mark onto the wrong events."""

    ts: str
    channel: str
    host: str
    detail: str = ""
    provider: str = ""
    n_bytes: int = 0
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
    this: an `OLLAMA_HOST` pointing off-machine makes the "local" models network egress —
    recorded in the ledger, refused under air-gap, and disqualifying for the
    local-inference claim. Fails toward NOT local (an unparseable endpoint must never earn a
    'local' claim)."""
    endpoint = ollama_endpoint()
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    try:
        name = (urlparse(endpoint).hostname or "").lower()
    except Exception:
        return False
    return is_loopback_host(name)


def _inet_aton(name: str):
    """inet_aton's grammar, without the socket module (trust/ imports no network client): one
    to four dot-separated parts, each decimal, `0x` hex or leading-zero octal, the last part
    filling the remaining bytes. The IPv4Address it names, else None."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    vals = []
    for p in parts:
        if not p or not p.isascii() or not p.isalnum():
            return None
        try:
            v = (int(p, 16) if p[:2] in ("0x", "0X")
                 else int(p, 8) if len(p) > 1 and p[0] == "0"
                 else int(p, 10))
        except ValueError:
            return None
        vals.append(v)
    tail_bytes = 5 - len(vals)
    if any(v > 255 for v in vals[:-1]) or vals[-1] >= 256 ** tail_bytes:
        return None
    n = 0
    for v in vals[:-1]:
        n = (n << 8) | v
    return ipaddress.IPv4Address((n << (8 * tail_bytes)) | vals[-1])


def _parse_address(name: str):
    """The IP address `name` is a literal of, else None. `ipaddress` reads the canonical forms;
    the resolver also accepts inet_aton shorthand — `127.1`, hex or octal octets
    (`0x7f.0.0.1`, `0177.0.0.1`), a bare decimal — and connects to the same loopback, so those
    must parse here too or a string the checks below read as a public NAME reaches a local
    service. A name with letters other than a hex octet is left to DNS (None)."""
    try:
        return ipaddress.ip_address(name)
    except ValueError:
        return _inet_aton(name)


def is_loopback_host(name: str) -> bool:
    """Whether a hostname is THIS machine: `localhost`, or a literal address in the loopback
    range (127.0.0.0/8, ::1) or the unspecified address a local daemon binds. Parsed as an
    address (`_parse_address`), never matched as a string prefix — `127.evil.example.com` is a
    remote name."""
    name = (name or "").strip().lower()
    if name == "localhost":
        return True
    addr = _parse_address(name)
    if addr is None:
        return False
    return addr.is_loopback or addr.is_unspecified


# Names that resolve inside the local network by convention (mDNS, the loopback TLD, RFC 8375 /
# common router suffixes). A bare single-label name ("nas") is local by construction.
_PRIVATE_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home.arpa")


def is_private_host(name: str) -> bool:
    """Whether a hostname points at this machine or a private network: loopback, an RFC 1918 /
    link-local / reserved literal address (the cloud metadata address is link-local), or a name
    that only resolves locally. No DNS lookup — a public name that RESOLVES to a private
    address is out of this check's sight. Used to keep a model-chosen fetch off local services
    (trust/quarantine.url_hold, tools/web._fetch)."""
    name = (name or "").strip().lower().rstrip(".")
    if is_loopback_host(name):
        return True
    addr = _parse_address(name)
    if addr is None:
        return "." not in name or name.endswith(_PRIVATE_SUFFIXES)
    return addr.is_private or addr.is_link_local or addr.is_reserved or addr.is_multicast


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
           n_bytes: int = 0, status: str = SENT) -> None:
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
        "`/policy airgap off`."
    )


def check(channel: str, host: str, detail: str = "", *, provider: str = "") -> "str | None":
    """THE refusal gate for a network op — every rung of "may this leave the machine" lives here
    (air-gap today; any future rung lands here and reaches every exit).
    Returns None when egress is allowed; on refusal, records a `blocked` event and returns the
    refusal string for the caller to hand back (tools return it to the model as their
    observation)."""
    if airgap_on():
        record(channel, host, detail, provider=provider, status=BLOCKED)
        return blocked_message(_host_label(host), channel)
    return None


def check_or_raise(channel: str, host: str, detail: str = "", *, subject: str = "",
                   provider: str = "") -> None:
    """The raising twin of `check()`, for the exits that CANNOT hand a refusal string back: an
    LLM, an embedder. Delegates to check() — one gate, one
    recording site, so a future rung added inside check() refuses these exits too — then raises
    instead of returning. `subject` names what was refused ("the model (qwen3.5:9b)",
    "embedding document text") so the message stays specific.

    Callers must not re-implement this: an inference exit that hand-rolls the check is one the
    ledger can silently miss (and one a future rung inside check() would never reach)."""
    if check(channel, host, detail, provider=provider) is None:
        return
    what = subject or "this operation"
    raise RuntimeError(
        f"Air-gap is ON — {what} would cross the network to {host}. Nothing was sent. If "
        f"OLLAMA_HOST points off this machine, unset it to use the local daemon, or turn the "
        f"air-gap off with `/policy airgap off`."
    )


def events() -> list[EgressEvent]:
    """The ledger, oldest first (a copy — callers may filter/slice freely)."""
    return list(_LEDGER)


def count() -> int:
    """Boundary events this session — sends and air-gap blocks — for the status bar's egress
    counter. Untracked runs are not counted: the bar counts what crossed or was refused."""
    return sum(1 for e in _LEDGER if e.status != UNTRACKED)


def next_seq() -> int:
    """The seq the NEXT recorded event will carry — capture at turn start, hand to events_since.
    Unlike a list index, a seq mark stays valid across the cap-trim and clear()."""
    return _SEQ + 1


def events_since(mark: int) -> list[EgressEvent]:
    """Events recorded at or after seq `mark`, oldest first. Seq-keyed (never an index into the
    ledger) so the _MAX_EVENTS trim or a mid-session `/policy egress clear` can't shift a
    turn-start mark onto the wrong slice — the trust receipt must never read 'local-only' over a
    turn that actually sent."""
    out: list[EgressEvent] = []
    for e in reversed(_LEDGER):
        if e.seq < mark:
            break
        out.append(e)
    out.reverse()
    return out


def summarize_events(events) -> dict:
    """Aggregate one slice of EgressEvents — THE one accounting every per-slice trust surface
    uses (the per-answer receipt, the `/policy egress` headline), so they can
    never report different byte/host numbers for the same events. Returns
    {sent, blocked, untracked, bytes, hosts (first-seen order), channels (sent, first-seen)}."""
    sent = [e for e in events if getattr(e, "status", "") == SENT]
    blocked = [e for e in events if getattr(e, "status", "") == BLOCKED]
    untracked = [e for e in events if getattr(e, "status", "") == UNTRACKED]
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
        "untracked": len(untracked),
        "bytes": sum(_safe_int(getattr(e, "n_bytes", 0)) for e in sent),
        "hosts": hosts,
        "channels": channels,
    }


def summary() -> dict:
    """Aggregate the ledger for the `/policy egress` headline: totals, bytes, distinct hosts,
    blocked. Carries `cleared`: whether a `/policy egress clear` wiped events this session —
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
        "untracked": agg["untracked"],
        "bytes": agg["bytes"],
        "hosts": agg["hosts"],
        "by_channel": by_channel,
        "cleared": _CLEARED_AT > 0,
    }


def clear() -> None:
    """Empty the ledger (a deliberate operator reset via `/policy egress clear`). The seq counter
    is NOT reset — outstanding turn-start marks must keep pointing past the cleared events, not
    get re-matched against new ones. The clear itself is remembered (cleared_since) so a per-turn
    consumer can tell an empty slice from a clear-emptied one instead of reporting
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
# "Where do the words come from" — local (computed on this machine) vs off-machine (an Ollama
# daemon behind a remote OLLAMA_HOST). THE one classifier: the session posture line
# (receipt.posture_spans) and `/policy` both read this — never re-rolled.


def _inference() -> dict:
    """Local-vs-off-machine binding map. 'local' means the words are computed ON THIS MACHINE: an
    Ollama binding only earns it when the endpoint is loopback — a remote OLLAMA_HOST is network
    inference and classifies 'remote', reported with the endpoint so the reader can see where."""
    cfg = get_config()
    ollama_local = ollama_is_local()
    ollama_loc = "local" if ollama_local else "remote"
    bindings = []
    try:
        bindings.append({"role": "chat", "model": cfg.chat_model, "locality": ollama_loc})
    except KeyError:
        pass
    try:
        bindings.append({"role": "embedder", "model": cfg.embedder_model, "locality": ollama_loc})
    except Exception:
        pass
    out = {"bindings": bindings, "all_local": ollama_local}
    if not ollama_local:
        out["remote_ollama"] = ollama_endpoint()
    return out


def remote_ollama_label(inf: dict) -> str:
    """The display label for a remote-Ollama destination (`ollama @ <endpoint>`) — one spelling
    for every surface that names it (posture line, /policy tables, the report render)."""
    return f"ollama @ {inf.get('remote_ollama', '?')}"


def offmachine_destinations(inf: "dict | None" = None) -> list[str]:
    """The off-machine inference destinations as display labels: the remote Ollama endpoint
    (`remote_ollama_label`) when there is one. THE one assembly of the where-list — the session
    posture line (receipt.posture_spans), /policy's verdict, all print this, so they can never
    name different destination sets for the identical posture. Takes the classifier's dict (or
    computes it fresh); empty when everything is local."""
    if inf is None:
        inf = _inference()
    return [remote_ollama_label(inf)] if inf.get("remote_ollama") else []
