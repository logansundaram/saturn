"""
Trust receipt — the per-answer, ambient proof of the privacy/control posture.

`/privacy egress` proves what left the machine on demand; the trust receipt proves it on EVERY
answer: the one-line stats receipt under each response gains a trust segment — `local-only` when
nothing left the machine this turn, or the bytes/host summary when something did, plus how many
calls faced the approval gate. The privacy claim stops being something the user has to go check
and becomes something every single answer carries.

Data sources: the egress ledger (`egress.py` — the turn's slice of it, marked at turn start) and
the gated-call counter the approval UI increments. `trust_spans` is the pure builder (testable
with synthetic events) — it returns `(text, kind)` spans so the renderer can color each fact
semantically (the same green/yellow/red vocabulary the Glass Box uses for the identical facts);
`trust_parts` is its plain-text view, `turn_spans`/`turn_parts` the live wrappers the response
renderer calls. The live wrappers treat an unusable mark (0, or one a `/privacy egress clear`
wiped events past) as UNKNOWN — the receipt must never assert 'local-only' over a slice that may
be missing real sends (the same contract `/trace answer` applies before trusting the slice).

`runtime.receipt` (read live, default on) switches the segment off for users who want the plain
stats receipt back. Imports only config + egress + textutil (leaves), so the TUI can import it
freely.
"""

from __future__ import annotations

from trust import egress
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


def _gated_span(gated_calls: int) -> "tuple[str, str]":
    return (f"{gated_calls} call{'' if gated_calls == 1 else 's'} gated", "gated")


def trust_spans(events: list, gated_calls: int = 0) -> list[tuple[str, str]]:
    """The receipt's trust segment from one turn's egress events + gated-call count, as
    `(text, kind)` spans — kind ∈ `local`|`sent`|`blocked`|`gated` — so the styled renderer can
    color each fact semantically while the plain path prints the identical bare text.

    `local-only` when nothing was sent; otherwise a compact send summary (count · bytes · first
    host, `+n` for more). Blocked attempts (air-gap) and gated calls append when present.
    Accounting comes from egress.summarize_events — the same aggregation the Glass Box and
    /privacy egress use, so the receipt can never disagree with them."""
    agg = egress.summarize_events(events)

    spans: list[tuple[str, str]] = []
    if not agg["sent"]:
        spans.append(("local-only", "local"))
    else:
        hosts = agg["hosts"]
        label = f"⇅ {agg['sent']} send{'' if agg['sent'] == 1 else 's'}"
        if agg["bytes"]:
            label += f" · {human_bytes(agg['bytes'])}"
        if hosts:
            label += f" → {hosts[0]}"
            if len(hosts) > 1:
                label += f" +{len(hosts) - 1}"
        spans.append((label, "sent"))
    if agg["blocked"]:
        spans.append((f"⛔ {agg['blocked']} blocked", "blocked"))
    if gated_calls:
        spans.append(_gated_span(gated_calls))
    return spans


def trust_parts(events: list, gated_calls: int = 0) -> list[str]:
    """Plain-text view of trust_spans — the same words with the kinds dropped (the no-rich
    receipt path and anything that just needs the text)."""
    return [text for text, _ in trust_spans(events, gated_calls)]


def turn_spans(since_mark: int, gated_calls: int = 0) -> list[tuple[str, str]]:
    """The live trust spans for the turn whose first event would carry seq `since_mark` (from
    `mark()` at turn start). A mark of 0 (no turn recorded — headless, or before the first turn)
    or one that `/privacy egress clear` wiped events past means the slice may be MISSING real
    sends — render the honest unknown (kind `unknown`) instead of asserting 'local-only', the
    same guard `/trace answer` applies before trusting the live slice."""
    if since_mark <= 0 or egress.cleared_since(since_mark):
        spans: list[tuple[str, str]] = [("egress unknown", "unknown")]
        if gated_calls:
            spans.append(_gated_span(gated_calls))
        return spans
    return trust_spans(egress.events_since(since_mark), gated_calls)


def turn_parts(since_mark: int, gated_calls: int = 0) -> list[str]:
    """Plain-text view of turn_spans (same unknown-mark guard)."""
    return [text for text, _ in turn_spans(since_mark, gated_calls)]


# ── one-time discovery hints ───────────────────────────────────────────────────────────────────
# The receipt is where the trust surfaces first become visible, so it owns the tiny "has this
# hint fired yet" sentinels the response renderer consults — the same mechanism as the first-run
# `.setup_done` sentinel (a marker file in the database dir, so deleting the database resets
# discovery along with first-run). Session-level set as the fail-safe: an unwritable dir degrades
# to once per session instead of crashing or repeating every answer.
_HINTS_SHOWN: set[str] = set()


def take_hint(name: str) -> bool:
    """True exactly once per install for hint `name`; the caller renders the hint iff True.
    Consuming touches `database/.hint_<name>`; if the sentinel can't be read or written, the
    in-memory guard still bounds the hint to once per session."""
    if name in _HINTS_SHOWN:
        return False
    _HINTS_SHOWN.add(name)
    try:
        sentinel = get_config().path("database") / f".hint_{name}"
        if sentinel.exists():
            return False
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
    except Exception:
        pass  # unwritable sentinel dir — the session set above still makes this once-per-session
    return True
