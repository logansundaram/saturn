"""
On-demand readouts (`/config context`, `/models`) and the one-off log lines (notes, warnings,
steering/pause acknowledgements, queued-line echoes). All render in the trace-rail style and reuse the
shared meter vocabulary (`_mini_bar`/`_meter_color`), so a gauge reads identically here, in the
status bar, and in the live trace. None of these touch per-turn state.
"""

from ._base import (
    Text, _console, _RICH,
    _ACCENT, _DIM, _RAIL_GLYPH,
    _emit, _meter_color, _mini_bar, _rail, _truncate,
)
from .listing import section


# ── system metrics display ───────────────────────────────────────────────────────
def show_system_metrics(metrics) -> None:
    """Display a compact system-resource readout in the trace-rail style. Shares the one meter
    glyph + threshold vocabulary (`_mini_bar` / `_meter_color`) with the status bar and /config context,
    so a hot gauge reads identically everywhere; percentages are whole numbers (no false precision)."""

    def _row(label: str, pct: float, detail: str = "") -> None:
        bar = _mini_bar(pct, 20)
        col = _meter_color(pct)
        if _RICH:
            line = _rail()
            line.append(f"{label:<6}", style=_DIM)
            line.append(f"  {bar}", style=col)
            line.append(f"  {pct:>3.0f}%", style=col)
            if detail:
                line.append(f"   {detail}", style=_DIM)
            _console.print(line)
        else:
            print(f"  {_RAIL_GLYPH} {label:<6}  {bar}  {pct:>3.0f}%{'   ' + detail if detail else ''}")

    section("system")

    _row("cpu", metrics.cpu_usage_percent)
    ram_pct = metrics.ram_used_gb / metrics.total_ram_gb * 100
    _row("ram", ram_pct, f"{metrics.ram_used_gb:.1f} / {metrics.total_ram_gb:.1f} GB")
    if metrics.gpu_usage_percent is not None:
        _row("gpu", metrics.gpu_usage_percent)
    if metrics.vram_used_gb is not None and metrics.total_vram_gb is not None:
        vram_pct = metrics.vram_used_gb / metrics.total_vram_gb * 100
        _row("vram", vram_pct, f"{metrics.vram_used_gb:.1f} / {metrics.total_vram_gb:.1f} GB")


# ── context-window readout (the /config context command) ──────────────────────────────────
def show_context(window: int, used: int, source: str, per_role: dict[str, int]) -> None:
    """Detailed context-window readout for /config context: the active window + where it comes from, a
    wide fill bar for the last measured usage, and the per-role windows. Same trace-rail
    vocabulary as show_system_metrics; the compact form of this fill gauge also rides the live
    status bar during a turn."""
    pct = (used / window * 100) if window else 0.0
    col = _meter_color(pct)
    bar = _mini_bar(pct, width=28)

    section("context")

    if _RICH:
        win = Text("  ")
        win.append("window ", style=_DIM)
        win.append(f"{window:,}", style="default")
        win.append(" tokens", style=_DIM)
        win.append(f"   ({source})", style=_DIM)
        _console.print(win)

        usage = _rail()
        usage.append("usage ", style=_DIM)
        usage.append(f" {bar}", style=col)
        usage.append(f"  {pct:>4.0f}%", style=col)
        usage.append(f"   {used:,} / {window:,}", style=_DIM)
        _console.print(usage)
    else:
        print(f"  window {window:,} tokens   ({source})")
        print(f"  {_RAIL_GLYPH} usage  {bar}  {pct:>4.0f}%   {used:,} / {window:,}")

    if per_role:
        roles_txt = " · ".join(f"{r} {w:,}" for r, w in per_role.items())
        _emit(f"  roles: {roles_txt}")
    _emit("  set with /config context <size> (or /config context auto for per-model capability)")


# ── log lines (startup notices, warnings) ────────────────────────────────────────
# Every line below is the same shape: a glyph, a message, and an optional dim tail — rendered as
# styled spans under rich and as plain text without it. One helper so the rich/plain fallback
# contract lives in ONE place (it was hand-copied six times here and once in response.py, so a
# fix to the fallback had to land seven times). The plain form is always glyph + text + tail,
# which is exactly what each copy printed.
def _glyph_line(glyph: str, glyph_style: str, text: str,
                text_style: str = "default", tail: str = "", tail_style: str = "") -> None:
    if _RICH:
        t = Text()
        t.append(glyph, style=glyph_style)
        t.append(text, style=text_style)
        if tail:
            t.append(tail, style=tail_style or _DIM)
        _console.print(t)
    else:
        print(f"{glyph}{text}{tail}")


def note(msg: str) -> None:
    """A quiet informational line (dim) — e.g. the `@file` attachment notice. Distinct from
    `warn` (yellow), which flags a problem; a note is just neutral context."""
    _glyph_line("  · ", _DIM, msg, _DIM)


def warn(msg: str) -> None:
    _glyph_line("  ! ", "yellow", msg, "yellow")


def steer_note(text: str) -> None:
    """Acknowledge a mid-turn steering correction the moment it's captured (Esc with typed text).
    The correction is injected into the running turn at the next step boundary (see plan_gate); this
    is the immediate feedback that it landed, printed above the live status bar."""
    _glyph_line("  ↪ ", f"bold {_ACCENT}", "steering — applies at the next step: ",
                _ACCENT, _truncate(text, 80), _DIM)


def pause_note() -> None:
    """Acknowledge an empty-line Esc the moment it's captured. The plan-review pause lands at the
    next step boundary (see plan_gate), which on a local model can be a long wait — this is the
    immediate feedback that the keypress registered, printed above the live status bar exactly
    like steer_note's steering acknowledgement."""
    _glyph_line("  ⏸ ", f"bold {_ACCENT}", "pausing for plan review at the next step…", _ACCENT)


def freeze_note() -> None:
    """Acknowledge an Esc that froze the streaming answer (interrupt-and-correct) the moment
    it's captured — the stream stops at the next token and the freeze editor opens, but on a
    slow local model that beat can lag the keypress; this is the immediate feedback, printed
    above the live answer region exactly like steer_note/pause_note."""
    _glyph_line("  ✂ ", f"bold {_ACCENT}",
                "freezing the answer — the editor opens when the stream stops…", _ACCENT)


def echo_queued(line: str) -> None:
    """Echo a type-ahead line as the REPL pulls it off the queue to run, so a query/command the
    user typed while a previous turn was working shows up in the transcript just like a line typed
    live at the `»` prompt (with a quiet `queued` tag to mark where it came from)."""
    _glyph_line("» ", f"bold {_ACCENT}", line, "default", "   (queued)", _DIM)
