"""
Trust receipt — the per-answer, ambient proof of the privacy/control posture.

`/privacy egress` proves what left the machine on demand; the trust receipt proves it on EVERY
answer: the one-line stats receipt under each response gains a trust segment — `local-only` when
nothing left the machine this turn, or the bytes/host summary when something did, plus how many
calls faced the approval gate. The privacy claim stops being something the user has to go check
and becomes something every single answer carries.

Data sources: the egress ledger (`egress.py` — the turn's slice of it, marked at turn start) and
the gate-prompt counter the approval UI increments. `trust_parts` is the pure builder (testable
with synthetic events); `turn_parts` is the live wrapper the response renderer calls.

`runtime.receipt` (read live, default on) switches the segment off for users who want the plain
stats receipt back. Imports only config + egress + textutil (leaves), so the TUI can import it
freely.
"""

from __future__ import annotations

import egress
from config import get_config
from textutil import human_bytes


def enabled() -> bool:
    """Whether the trust segment renders on the answer receipt (`runtime.receipt`)."""
    return bool(get_config().get("runtime.receipt", True))


def mark() -> int:
    """The turn-start egress mark — the seq the next event will carry, NOT a ledger index (the
    cap-trim and `/privacy egress clear` shift indexes, and a stale index would make the receipt
    read 'local-only' over a turn that sent). Hand back to `turn_parts`."""
    return egress.next_seq()


# The live turn's mark. Receipt-domain state owned HERE (not a TUI module global): the turn
# lifecycle (statusbar.reset_turn in the interactive loop) calls reset_turn(); the response
# renderer and the Glass Box read turn_mark(). 0 = no turn marked yet (headless, or before the
# first turn) — consumers must treat that as UNKNOWN, never as "the whole ledger is this turn".
_TURN_MARK = 0


def reset_turn() -> None:
    """Record the egress mark for a turn that is about to run."""
    global _TURN_MARK
    _TURN_MARK = mark()


def turn_mark() -> int:
    """The current turn's egress mark (0 = none recorded)."""
    return _TURN_MARK


def trust_parts(events: list, gate_prompts: int = 0) -> list[str]:
    """The receipt's trust segment from one turn's egress events + gate-prompt count.

    `local-only` when nothing was sent; otherwise a compact send summary (count · bytes · first
    host, `+n` for more). Blocked attempts (air-gap) and gate prompts append when present.
    Accounting comes from egress.summarize_events — the same aggregation the Glass Box and
    /privacy egress use, so the receipt can never disagree with them."""
    agg = egress.summarize_events(events)

    parts: list[str] = []
    if not agg["sent"]:
        parts.append("local-only")
    else:
        hosts = agg["hosts"]
        label = f"⇅ {agg['sent']} send{'' if agg['sent'] == 1 else 's'}"
        if agg["bytes"]:
            label += f" · {human_bytes(agg['bytes'])}"
        if hosts:
            label += f" → {hosts[0]}"
            if len(hosts) > 1:
                label += f" +{len(hosts) - 1}"
        parts.append(label)
    if agg["blocked"]:
        parts.append(f"⛔ {agg['blocked']} blocked")
    if gate_prompts:
        parts.append(f"{gate_prompts} gated")
    return parts


def turn_parts(since_mark: int, gate_prompts: int = 0) -> list[str]:
    """The live trust segment for the turn whose first event would carry seq `since_mark`
    (from `mark()` at turn start)."""
    return trust_parts(egress.events_since(since_mark), gate_prompts)
