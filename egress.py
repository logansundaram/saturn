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
durable audit log (that is `/trace export`). Imports only config + diag, so any module (web tools,
mcp_client, llms) can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from config import get_config

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
    approximate size of what was SENT, `redactions` how many secrets were stripped first."""

    ts: str
    channel: str
    host: str
    detail: str = ""
    provider: str = ""
    n_bytes: int = 0
    redactions: int = 0
    status: str = SENT


_LEDGER: list[EgressEvent] = []


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
    machine is the one failure a boundary ledger must never have."""
    try:
        ev = EgressEvent(
            ts=datetime.now().isoformat(),
            channel=str(channel),
            host=_host_label(str(host) if host is not None else ""),
            detail=str(detail or ""),
            provider=str(provider or ""),
            n_bytes=_safe_int(n_bytes),
            redactions=_safe_int(redactions),
            status=str(status or SENT),
        )
    except Exception:
        return
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


def count() -> int:
    """Number of egress events recorded this session (for the status-bar indicator)."""
    return len(_LEDGER)


def summary() -> dict:
    """Aggregate the ledger for the `/privacy egress` headline: totals, bytes, distinct hosts,
    blocked."""
    sent = [e for e in _LEDGER if e.status == SENT]
    blocked = [e for e in _LEDGER if e.status == BLOCKED]
    hosts = sorted({e.host for e in sent})
    by_channel: dict[str, int] = {}
    for e in sent:
        by_channel[e.channel] = by_channel.get(e.channel, 0) + 1
    return {
        "total": len(_LEDGER),
        "sent": len(sent),
        "blocked": len(blocked),
        "bytes": sum(e.n_bytes for e in sent),
        "redactions": sum(e.redactions for e in sent),
        "hosts": hosts,
        "by_channel": by_channel,
    }


def clear() -> None:
    """Empty the ledger (a deliberate operator reset via `/privacy egress clear`)."""
    _LEDGER.clear()
