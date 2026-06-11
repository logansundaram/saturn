"""
The bottom-pinned live status bar + its off-thread system-metrics sampler, plus `reset_turn`
(per-turn state seeding). One high-signal `rich.live.Live` line — identity · progress · resources —
re-evaluated on every refresh so the elapsed clock and the sampled gauges tick even between node
updates. The `Live` handle, the metrics snapshot, and the type-ahead preview stay private here; only
the per-turn timing/plan state (in `_base`) is shared with the trace/plan/response renderers.
"""

import time

from . import _base
from ._base import (
    Live, Text, _console, _RICH,
    _ACCENT, _DIM, _RAIL, _RISK,
    _active_ctx_window, _active_model_short, _fmt_dur, _meter_color, _mini_bar,
)


# ── live system-metrics sampler ───────────────────────────────────────────────
# cpu/ram/gpu/vram are sampled off the render path: nvidia-smi can block up to 2s, which must
# never stall the trace or the 4 Hz bar refresh. A lone daemon thread refreshes `_metrics` on a
# slow cadence; the bar just reads the latest cached snapshot (None until the first sample lands).
_METRICS_INTERVAL = 1.5  # seconds between samples
_metrics = None          # latest system_monitor.SystemMetrics (or None)
_metrics_thread = None


def _metrics_loop(interval: float) -> None:
    from tui.system_monitor import get_system_metrics

    global _metrics
    while True:
        try:
            _metrics = get_system_metrics()
        except Exception:
            pass
        time.sleep(interval)


def _metrics_start() -> None:
    """Lazily spin up the sampler (once per process). Daemon, so it dies with the interpreter;
    cheap enough at 1 sample / 1.5s to just run for the session's lifetime."""
    global _metrics_thread
    if _metrics_thread is not None:
        return
    import threading

    _metrics_thread = threading.Thread(
        target=_metrics_loop, args=(_METRICS_INTERVAL,), daemon=True
    )
    _metrics_thread.start()


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
    elapsed clock and the sampled system gauges tick even when no node update has fired. One
    high-signal line: identity · run progress · token/context · live hardware load · active node.
    Set no-wrap + ellipsis so a narrow terminal trims the right edge instead of wrapping to two
    rows (the bar must stay exactly one line for the Live region)."""

    def __rich__(self) -> "Text":
        elapsed = time.perf_counter() - _base._turn_start if _base._turn_start else 0.0
        status = _base._status
        n = status["tools"]
        tps = status["tok_per_sec"]
        bar = Text(no_wrap=True, overflow="ellipsis")

        def dot():   # within-zone separator (tight)
            bar.append(" · ", style=_DIM)

        def zone():  # between-zone separator: a quiet rule so the groups read as groups
            bar.append("   │   ", style=_RAIL)

        # ── identity ──
        bar.append("  ", style=_DIM)
        bar.append("saturday", style=f"bold {_ACCENT}")
        dot()
        bar.append(_active_model_short(), style="default")
        dot()
        # The gate policy, read live (policy.py — /autoapprove, Shift+Tab and /config are all
        # views of the same threshold). At `destructive` the gate is not "at a tier", it's OPEN,
        # and the bar must say so loudly for as long as that is true — the /autoapprove banner
        # scrolls away; this indicator doesn't.
        try:
            from config import get_config
            _perm = get_config().auto_approve
        except Exception:
            _perm = "read_only"
        if _perm == "destructive":
            bar.append("⚠ GATE OFF", style=f"bold {_RISK.get('destructive', 'red')}")
        else:
            bar.append(_perm, style=_RISK.get(_perm, _DIM))

        # Mode flags — air-gap (network sealed) and dry-run (nothing executes). Both are loud,
        # persistent states the user must be able to see at a glance for as long as they hold.
        try:
            from config import get_config as _gc
            _cfg = _gc()
            if bool(_cfg.get("runtime.airgap", False)):
                dot()
                bar.append("⛓ AIRGAP", style=f"bold {_ACCENT}")
            if bool(_cfg.get("runtime.dry_run", False)):
                dot()
                bar.append("DRY-RUN", style=f"bold {_RISK.get('side_effecting', 'yellow')}")
        except Exception:
            pass

        # ── type-ahead ── only present while the user is queuing input mid-turn. Placed right after
        # identity (ahead of progress) so the line being typed is never the part trimmed by the
        # bar's ellipsis overflow — seeing your own keystrokes matters more than the gauges here.
        buf, queued = _input_state["buffer"], _input_state["queued"]
        if buf or queued:
            zone()
            if buf:
                bar.append(buf, style=_ACCENT)  # the line being typed, highlighted in cyan
                bar.append("▏", style=f"bold {_ACCENT}")  # block cursor on the typed line
            if queued:
                label = f"  ({queued} queued)" if buf else f"{queued} queued"
                bar.append(label, style=_DIM)

        # ── progress ── the active stage leads: it's the highest-value live datum, and keeping it
        # left means a narrow terminal trims resources off the right rather than "where am I".
        zone()
        if status["node"]:
            bar.append(f"▸ {status['node']}", style=f"bold {_ACCENT}")
            dot()
        for i, label in enumerate((f"iter {status['iteration']}", _fmt_dur(elapsed).strip(),
                                   f"{n} tool{'' if n == 1 else 's'}")):
            if i:
                dot()
            bar.append(label, style="default")
        if tps > 0:
            dot()
            bar.append(f"{tps:.0f} tok/s", style="default")

        # ── egress ── how many times anything has left the machine this session (the boundary
        # counter). Only shown once non-zero, so a fully-local session stays calm; it's the live
        # twin of /privacy egress, and the whole point of the privacy proof point being *visible*.
        try:
            import egress as _eg
            _ne = _eg.count()
        except Exception:
            _ne = 0
        if _ne:
            zone()
            bar.append("⇅ ", style=_DIM)
            bar.append(f"{_ne} egress", style="default")

        # ── resources ── tertiary; ctx keeps its meter (it drives the agent), hardware is bare %.
        window = status["ctx_window"]
        m = _metrics
        if window or m is not None:
            zone()
        if window:
            _append_meter(bar, "ctx", status["ctx_used"] / window * 100, cells=4)
        if m is not None:
            if window:
                bar.append("  ", style=_DIM)
            _append_meter(bar, "cpu", m.cpu_usage_percent)
            ram_pct = m.ram_used_gb / m.total_ram_gb * 100 if m.total_ram_gb else 0.0
            bar.append("  ", style=_DIM)
            _append_meter(bar, "ram", ram_pct)
            if m.gpu_usage_percent is not None:
                bar.append("  ", style=_DIM)
                _append_meter(bar, "gpu", m.gpu_usage_percent)
            if m.vram_used_gb is not None and m.total_vram_gb:
                bar.append("  ", style=_DIM)
                _append_meter(bar, "vram", m.vram_used_gb / m.total_vram_gb * 100)
        return bar


def _append_meter(bar: "Text", label: str, pct: float, cells: int = 0) -> None:
    """`label NN%` (load-colored), optionally trailed by a tiny `▰▱` fill bar when `cells > 0` —
    the compact gauge form used in the bar. Meters are opt-in: only the context gauge carries one
    (it's what drives the agent); the hardware readouts stay bare percentages so the resources zone
    reads calm rather than like a dashboard."""
    col = _meter_color(pct)
    bar.append(f"{label} ", style=_DIM)
    bar.append(f"{pct:.0f}%", style=col)
    if cells:
        bar.append(f" {_mini_bar(pct, cells)}", style=col)


def _live_start() -> None:
    """Pin a fresh status bar at the bottom. No-op without rich or if one is already running.
    `transient=True` erases the bar on stop (the scrolling trace stays); rich's default
    stdout/stderr redirect keeps node `print()`s flowing above the live region."""
    global _live
    if not _RICH or _live is not None:
        return
    _metrics_start()  # ensure the off-thread cpu/ram/gpu sampler is running
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
    _base._status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0,
                     "ctx_used": _base._status.get("ctx_used", 0), "ctx_window": _active_ctx_window(),
                     "gates": 0}
    # Mark the egress ledger so the trust receipt can summarize exactly this turn's slice.
    # receipt.py owns the mark (receipt-domain state, not UI state); on failure the mark keeps
    # its previous value rather than being forced to 0 — readers treat 0 as "unknown", and a
    # forced 0 would make events_since(0) attribute the WHOLE session's egress to this turn.
    try:
        import receipt

        receipt.reset_turn()
    except Exception:
        pass
    _live_start()
