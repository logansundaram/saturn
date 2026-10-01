"""
The bottom-pinned live status bar, plus `reset_turn` (per-turn state seeding). One
high-signal `rich.live.Live` line — posture · progress · session — re-evaluated on every
refresh so the elapsed clock ticks even between node updates. The `Live` handle and the
type-ahead preview stay private here; only the per-turn timing/plan state (in `_base`) is shared
with the trace/plan/response renderers.
"""

import time

from . import _base
from ._base import (
    Live, Text, _console,
    _ACCENT, _DIM, _NODE_STARTING, _POSTURE_STYLE, _RAIL,
    _active_ctx_window, _fmt_dur, _meter_color, _mini_bar, _posture_flags,
)


# ── live status bar (bottom-pinned) ───────────────────────────────────────────
# `_live` holds the active rich.live.Live (None when torn down for input).
_live = None

# Type-ahead preview: the line the user is currently typing mid-turn + how many completed lines are
# already queued. Fed by typeahead.InputQueue's on_change callback (set_input_preview); rendered in
# the pinned status bar so queuing follow-ups while the agent works has live feedback.
_input_state = {"buffer": "", "queued": 0}


def set_input_preview(buffer: str, queued: int) -> None:
    """Update the status bar's type-ahead readout (current in-progress line + queue depth) and
    repaint the bar immediately so typing feels live, not capped at the bar's idle refresh rate.
    No-op visually when no bar is up (between turns) — the state is still stored for the next bar."""
    _input_state["buffer"] = buffer
    _input_state["queued"] = queued
    _live_refresh()


class _StatusBar:
    """Renderable for the pinned bar. `__rich__` is re-evaluated on every Live refresh, so the
    elapsed clock ticks even when no node update has fired. Quiet zones — posture · [type-ahead]
    · progress · session — plus the trailing key legend; most appear only when they have
    something to say. No-wrap + ellipsis so a narrow terminal trims the right edge instead of
    wrapping to two rows (the bar must stay exactly one line for the Live region)."""

    def __rich__(self) -> "Text":
        elapsed = time.perf_counter() - _base._turn_start if _base._turn_start else 0.0
        status = _base._status
        n = status["tools"]
        tps = status["tok_per_sec"]
        bar = Text(no_wrap=True, overflow="ellipsis")
        started = False  # whether any zone has rendered yet (separators sit between zones only)

        def dot():   # within-zone separator (tight)
            bar.append(" · ", style=_DIM)

        def zone():  # between-zone separator: a quiet rule so the groups read as groups
            nonlocal started
            if started:
                bar.append("  │  ", style=_RAIL)
            started = True

        # ── posture ── deviation-only: the calm default renders nothing. Leftmost on purpose:
        # the bar trims from the right edge, and "the gate is open" must be the last thing a
        # narrow terminal sacrifices. An UNREADABLE posture still marks itself: under
        # silence-means-default, omitting it would show a SAFER posture than reality on exactly
        # the surface that exists to shout ⚠ GATE OFF.
        bar.append("  ", style=_DIM)
        posture = _posture_flags()
        if posture is None:
            posture = [("posture ?", "dim")]
        if posture:
            zone()
            for i, (label, kind) in enumerate(posture):
                if i:
                    dot()
                bar.append(label, style=_POSTURE_STYLE[kind])

        # ── type-ahead ── only present while the user is queuing input mid-turn. Ahead of
        # progress so the line being typed is never the part trimmed by the bar's ellipsis
        # overflow — seeing your own keystrokes matters more than the gauges.
        buf, queued = _input_state["buffer"], _input_state["queued"]
        if buf or queued:
            zone()
            if buf:
                bar.append(buf, style=_ACCENT)  # the line being typed, highlighted in cyan
                bar.append("▏", style=f"bold {_ACCENT}")  # block cursor on the typed line
            if queued:
                label = f" ({queued} queued)" if buf else f"{queued} queued"
                bar.append(label, style=_DIM)

        # ── progress ── how far the turn has got, then counts · time · rate. The node is named in
        # the PAST tense: show_node is fed from a node's *update* event, which LangGraph emits
        # when the node COMPLETES (app/turn.py), so this is the last node that FINISHED — not the
        # one running now. There is no active-node signal to render, so it says what it knows.
        zone()
        if status["node"] == _NODE_STARTING:
            bar.append(_NODE_STARTING, style=_DIM)
            dot()
        elif status["node"]:
            bar.append(f"✓ {status['node']}", style=_DIM)
            dot()
        for i, label in enumerate((f"iter {status['iteration']}",
                                   f"{n} tool{'' if n == 1 else 's'}",
                                   _fmt_dur(elapsed).strip())):
            if i:
                dot()
            bar.append(label, style="default")
        if tps > 0:
            dot()
            bar.append(f"{tps:.0f} tok/s", style="default")

        # ── session ── the turn-spanning gauges, one zone: context fill (it drives the agent, so
        # it keeps its meter) and the egress counter (the live twin of /policy egress — the
        # boundary, visible). Egress appears only once non-zero, so a fresh, fully-local session
        # stays calm.
        window = status["ctx_window"]
        try:
            from trust import egress as _eg
            _ne = _eg.count()
        except Exception:
            _ne = 0
        if window or _ne:
            zone()
            if window:
                _append_meter(bar, "ctx", status["ctx_used"] / window * 100, cells=4)
            if _ne:
                if window:
                    dot()
                bar.append("⇅ ", style=_DIM)
                bar.append(f"{_ne} egress", style="default")

        # ── key legend ── the turn-time keys, taught ambiently while they're usable. Trails the
        # whole line ON PURPOSE: the bar trims from the right edge on a narrow terminal (no-wrap
        # + ellipsis), so the hint is the first thing sacrificed — never the posture or progress.
        zone()
        bar.append("esc pause · ctrl-c cancel", style=_DIM)
        return bar


def _append_meter(bar: "Text", label: str, pct: float, cells: int = 0) -> None:
    """`label NN%` (load-colored), optionally trailed by a tiny `▰▱` fill bar when `cells > 0` —
    the compact gauge form used in the bar."""
    col = _meter_color(pct)
    bar.append(f"{label} ", style=_DIM)
    bar.append(f"{pct:.0f}%", style=col)
    if cells:
        bar.append(f" {_mini_bar(pct, cells)}", style=col)


def _live_start() -> None:
    """Pin a fresh status bar at the bottom. No-op if one is already running.
    `transient=True` erases the bar on stop (the scrolling trace stays); rich's default
    stdout/stderr redirect keeps node `print()`s flowing above the live region."""
    global _live
    if _live is not None:
        return
    _live = Live(_StatusBar(), console=_console, transient=True,
                 auto_refresh=True, refresh_per_second=4)
    _live.start()


def _live_stop() -> None:
    """Tear the bar down (before any input()) so it never fights a blocking prompt."""
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def _live_refresh() -> None:
    if _live is not None:
        _live.refresh()


def reset_turn() -> None:
    """Call once at the start of each user turn: resets node timing + plan-diff state and
    starts the bottom-pinned status bar for the turn."""
    _base._t_last = time.perf_counter()
    _base._turn_start = _base._t_last
    _base._plan_seen = {}
    _base._trace_started = False  # next node line leads with a blank to part it from the prompt
    # Carry the last measured context fill across turns (it only grows; refreshed once the agent
    # runs) but re-read the window in case the model/tier changed since the last turn.
    # Seed the node zone with `starting`: the turn is underway but no node has COMPLETED yet, so
    # there is no finished node to name and an empty zone would read as a stalled bar.
    _base._status = {"node": _NODE_STARTING, "iteration": 0, "tools": 0, "tok_per_sec": 0.0,
                     "ctx_used": _base._status.get("ctx_used", 0), "ctx_window": _active_ctx_window(),
                     "gates": 0}
    # Mark the egress ledger so the trust receipt can summarize exactly this turn's slice.
    # receipt.py owns the mark (receipt-domain state, not UI state); on failure the mark keeps
    # its previous value rather than being forced to 0 — readers treat 0 as "unknown", and a
    # forced 0 would make events_since(0) attribute the WHOLE session's egress to this turn.
    try:
        from trust import receipt

        receipt.reset_turn()
    except Exception:
        pass
    _live_start()
