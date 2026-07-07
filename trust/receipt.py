"""
Trust receipt — the ambient trust surfaces: the per-answer receipt segment, the session-start
posture line, and the one-time discovery hints.

**Calm by default, loud on deviation (2026-07-06 declutter — owner call):** the ambient
surfaces speak only when something actually crossed the boundary or was loosened. The receipt's
trust segment appears when the turn SENT something, was BLOCKED by air-gap, or faced the gate —
a fully-local turn adds nothing to the stats line. `posture_spans` is the session-level twin: a
facet at its safe default (gate read_only, local inference, quarantine gate) says nothing, so a
stock local install renders no posture line at all. The affirmative reassurance ("everything is
local, here's proof") lives on demand behind `/privacy` and `/glass` — silence in the ambient
flow means the defaults hold.

Data sources: the egress ledger (`egress.py` — the turn's slice of it, marked at turn start) and
the gated-call counter the approval UI increments. `trust_spans` is the pure builder (testable
with synthetic events) — it returns `(text, kind)` spans so the renderer can color each fact
semantically (the same green/yellow/red vocabulary the Glass Box uses for the identical facts);
`trust_parts` is its plain-text view, `turn_spans`/`turn_parts` the live wrappers the response
renderer calls. The live wrappers treat an unusable mark (0, or one a `/privacy egress clear`
wiped events past) as UNKNOWN — silence never makes a claim, but a slice that may be HIDING
sends still says `egress unknown` rather than blending into the calm (the same contract
`/trace answer` applies before trusting the slice).

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
    stay silent over a turn that sent). Hand back to `turn_parts`."""
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
    `(text, kind)` spans — kind ∈ `sent`|`blocked`|`gated` — so the styled renderer can
    color each fact semantically while the plain path prints the identical bare text.

    Deviation-only: EMPTY when nothing was sent, blocked, or gated (the calm local turn — the
    receipt is then just the dim run stats); otherwise a compact send summary (count · bytes ·
    first host, `+n` for more), blocked attempts (air-gap), and the gated count. Accounting
    comes from egress.summarize_events — the same aggregation the Glass Box and /privacy egress
    use, so the receipt can never disagree with them."""
    agg = egress.summarize_events(events)

    spans: list[tuple[str, str]] = []
    if agg["sent"]:
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
    sends — render the honest unknown (kind `unknown`) instead of blending into the calm
    no-deviation silence, the same guard `/trace answer` applies before trusting the live
    slice."""
    if since_mark <= 0 or egress.cleared_since(since_mark):
        spans: list[tuple[str, str]] = [("egress unknown", "unknown")]
        if gated_calls:
            spans.append(_gated_span(gated_calls))
        return spans
    return trust_spans(egress.events_since(since_mark), gated_calls)


def turn_parts(since_mark: int, gated_calls: int = 0) -> list[str]:
    """Plain-text view of turn_spans (same unknown-mark guard)."""
    return [text for text, _ in turn_spans(since_mark, gated_calls)]


# ── session posture line ───────────────────────────────────────────────────────────────────────
# The startup twin of the per-answer receipt: one line under the banner stating the live trust
# posture — but DEVIATION-ONLY (2026-07-06 declutter): a stock local install prints nothing at
# all, and the line speaks only when something is loosened or leaves the machine. Same
# (text, kind) span shape as trust_spans so the renderer colors semantically and the plain path
# prints identical words.


def posture_spans() -> list[tuple[str, str]]:
    """The session's live trust posture as (text, kind) spans — kind ∈ ok|warn|risk|accent|dim —
    deviation-only: a facet at its safe default (gate read_only · local inference · quarantine
    gate) says NOTHING, so the default posture renders no line at all; silence means the
    defaults hold. What speaks: a loosened/open gate, the air-gap seal, off-machine inference,
    a weakened quarantine, and the redaction mode once an off-machine boundary exists. The
    affirmative readout lives behind /privacy. Every read is live and best-effort: a facet that
    can't be derived is OMITTED rather than guessed — this line must never claim a posture it
    didn't read."""
    spans: list[tuple[str, str]] = []
    try:
        cfg = get_config()
    except Exception:
        return spans

    # Gate tier — only above the read_only default. At `destructive` the gate is not "at a
    # tier", it's OPEN — same loud label the status bar uses for as long as that holds.
    try:
        from trust import policy

        tier = policy.tier()
        if tier == "destructive":
            spans.append(("⚠ GATE OFF", "risk"))
        elif tier != "read_only":
            spans.append((f"gate {tier}", "warn"))
    except Exception:
        pass

    # Boundary modes — shown only while active, like the status bar flags.
    try:
        if bool(cfg.get("runtime.airgap", False)):
            spans.append(("⛓ airgap", "accent"))
    except Exception:
        pass

    # Inference locality — the headline privacy fact, but only when the words LEAVE the
    # machine; all-local is the expected default and stays silent.
    all_local = None
    try:
        # The one locality classifier + the one where-list assembly — never re-rolled.
        from trust.egress import _inference, offmachine_destinations

        inf = _inference()
        all_local = bool(inf.get("all_local"))
        if not all_local:
            if inf.get("remote_ollama"):
                # A remote OLLAMA_HOST: the words come from another machine even though the
                # provider says "ollama" — name the endpoint, never let it read as local.
                spans.append(
                    (f"inference off-machine: {', '.join(offmachine_destinations(inf))}", "warn")
                )
            else:
                cloud = ", ".join(offmachine_destinations(inf)) or "cloud"
                spans.append((f"inference cloud: {cloud}", "warn"))
    except Exception:
        pass

    # Quarantine — only when weakened below the `gate` default. The EFFECTIVE mode, not the raw
    # config string: quarantine.mode() lowercases and falls back to "gate" on an invalid value,
    # so `runtime.quarantine: none` runs gated and renders as the silent default it actually is
    # — never an echoed string the system ignored. (Lazy import of a leaf — no cycle.)
    try:
        from trust import quarantine

        q = quarantine.mode()
        if q != "gate":
            spans.append((f"quarantine {q}", "warn" if q == "off" else "dim"))
    except Exception:
        pass

    # Redaction — only meaningful once an off-machine boundary exists to redact for: `off` on a
    # live boundary is the warning; an active mode is the calm qualifier of the inference span.
    try:
        from trust import redaction

        if all_local is False:
            mode = redaction.mode()
            spans.append((f"redaction {mode}", "warn" if mode == "off" else "dim"))
    except Exception:
        pass
    return spans


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
