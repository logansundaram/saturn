"""
CLI rendering for the agent console.

Design target: a serious local-agent console — git status / htop / pytest / a trace viewer,
not a chatbot. Dense, fast, keyboard-first, low-noise, inspectable. The aesthetic comes from
*structure*, not decoration:

  - A dim vertical rail (`│`) carries the execution trace. Consecutive node lines form one
    continuous gutter, so a turn reads as a single inspectable block (the htop/tree feel). Each
    node line leads with a green `✓` (the node has finished by the time it prints) then a dim
    `name  elapsed`. At normal verbosity the plumbing nodes (`ground`, `update_plan`, `plan_gate`)
    fold out of the rail — their *output* still prints and the trace DB keeps every node — so a
    turn reads as the user's mental model, `plan → agent → tools → … → synthesize`;
    `set_verbosity("verbose")` (via `/verbose full`) restores every node line and full timings.
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk tier. Structure
    is dim. Nothing is colored just to look nice — if it has color, it means something.
  - The plan prints **once** as the intended route, then emits a single line per status change
    as steps advance — a log/trace, not a re-rendered panel. This is the transparency surface
    and the main noise source, so it's diffed.
  - The `tools` node renders a **tool-I/O sub-tree** under its header: one `├─ name(args)  dur`
    branch per call, the call repr sized to the terminal width and durations column-aligned. Raw
    result previews are **hidden** by default (noisy JSON) — a failed call still shows its error
    inline (wrapped under the rail with a hanging indent), and `/calls` or `/verbose full` surfaces
    full outputs on demand. What the agent *did* (inputs · cost · ok/fail) stays visible; the
    workflow other tools hide.
  - LLM nodes annotate their trace line with the live **metrics for that step** (iteration,
    context tokens ingested, tok/s) — rendered **dim**: metrics are tertiary and must never
    out-shout the trace they ride on, let alone the response. The eye flows response → trace →
    metrics without having to parse the screen.
  - The approval gate deliberately breaks out of the rail with a heavy rule. It's a blocking
    safety decision and *should* draw the eye; everything else recedes. Each gated call shows its
    risk tier, every argument on its own line, and a one-line "what allowing this means" hint.
  - The final **response** renders as real markdown (headings, bold, lists, fenced code with
    syntax highlighting), so the answer reads as finished output, not a log line.
  - A single-line **status bar** is pinned at the bottom of the screen for the duration of a turn
    (`rich.live.Live`), grouped into three zones parted by a quiet `│` rule so it reads as
    deliberate groups, not a value stream: **identity** (`saturday · model`) │ **progress**
    (`▸node · iter · elapsed · tools · tok/s` — the active stage leads, kept left so it survives a
    narrow terminal) │ **resources** (`ctx NN% ▰▱` with a meter — it's what drives the agent — then
    bare `cpu/ram/gpu/vram NN%` load-colored percentages; gpu/vram sampled off-thread by a daemon so
    nvidia-smi never stalls the render). It's no-wrap +
    ellipsis so a narrow terminal trims the right edge rather than wrapping. The trace lines above
    it keep scrolling normally (rich routes `console.print` and captured `stdout` above the live
    region). It's `transient`, so it vanishes when the turn ends — the scrolling trace is the
    permanent record, the bar is just a live "where are we now" readout. Because `input()` can't
    run inside an active `Live`, the bar is torn down around the `»` prompt, the approval gate,
    and the final response, then restarted as the loop continues.

The agent emits node/plan/state updates; this module is one subscriber that renders them
(SATURDAY_MVP_PLAN.md §6). Swapping it for a Textual/Electron surface needs no graph change.
Degrades to plain ASCII-ish output if `rich` is absent (still UTF-8: stdout is reconfigured in
agent.py, so box-drawing glyphs are safe even on the no-color path).
"""

import io
import math
import os
import shutil
import time

try:
    from rich.console import Console
    from rich.text import Text
    from rich.live import Live
    from rich.markdown import Markdown

    _console = Console(highlight=False)
    _RICH = True
except Exception:  # pragma: no cover - fallback path
    _console = None
    _RICH = False

# prompt_toolkit drives the `»` input line so a typed `/command` is highlighted live, character
# by character — valid commands glow cyan, typos go red. Independent of rich: if it's missing we
# fall back to rich's (or plain) input(), just without the live highlight.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import (
        Completer as _PTKCompleter,
        Completion as _PTKCompletion,
        PathCompleter as _PTKPathCompleter,
    )
    from prompt_toolkit.document import Document as _PTKDocument
    from prompt_toolkit.key_binding import KeyBindings as _PTKKeyBindings
    from prompt_toolkit.lexers import Lexer as _PTKLexer
    from prompt_toolkit.styles import Style as _PTKStyle

    _PTK = True
except Exception:  # pragma: no cover - fallback path
    _PTK = False


# ── palette ──────────────────────────────────────────────────────────────────
# One accent, semantic status colors, everything else dim. Change here to retheme.
_ACCENT = "cyan"
_RAIL = "grey39"  # the trace gutter; quiet but visible
_DIM = "grey46"
_FAINT = "grey30"  # fainter than the rail — for the most incidental annotations (plan ::tool tags)

# status -> (glyph, style). Markers carry the state; color reinforces it.
_PLAN = {
    "pending": ("·", _DIM),
    "active": ("▸", f"bold {_ACCENT}"),
    "done": ("✓", "green"),
    "skipped": ("⨯", "grey30 strike"),
}
# risk tier -> style for the approval gate. Read-only never reaches the gate, but kept for parity.
_RISK = {
    "read_only": "green",
    "side_effecting": "yellow",
    "destructive": "bold red",
}
# one-line "what allowing this means" hint per tier, shown under each gated call.
_RISK_HINT = {
    "read_only": "no side effects",
    "side_effecting": "writes or calls out — review before allowing",
    "destructive": "irreversible — review carefully",
}

_RAIL_GLYPH = "│"
_NODE_W = 12  # node-name column width, keeps timings aligned

# Tree glyphs for the tool-I/O sub-trace (one branch per executed call).
_TREE_MID, _TREE_END, _TREE_PIPE, _TREE_LEAF = "├─", "└─", "│", "└"


# ── metric formatting (shared by the status bar and the readout commands) ─────
def _human_tokens(n: int) -> str:
    """Compact token count: 980 -> '980', 1842 -> '1.8k', 8192 -> '8k'."""
    if n < 1000:
        return str(int(n))
    k = n / 1000
    return f"{k:.0f}k" if k >= 10 or k == int(k) else f"{k:.1f}k"


def _meter_color(pct: float) -> str:
    """Load -> semantic color. Used for every gauge (context fill, cpu/ram/gpu) so a hot meter
    reads the same way everywhere: green ok, yellow warm, red hot."""
    if pct < 60:
        return "green"
    if pct < 85:
        return "yellow"
    return "bold red"


def _mini_bar(pct: float, width: int = 6) -> str:
    """A compact ▰▱ fill bar, `width` cells, clamped to [0, width]."""
    filled = max(0, min(width, round(pct / 100 * width)))
    return "▰" * filled + "▱" * (width - filled)


def _active_ctx_window() -> int:
    """The agent model's context window — the fill gauge's denominator. Lazily imports llms so
    ui stays a leaf module; best-effort (0 if the factory/config is unavailable)."""
    try:
        from llms import active_context_window

        return active_context_window()
    except Exception:
        return 0


def _active_model() -> str:
    """The current `tier:model` label for the status bar, resolved live each render — `/model`
    re-points config + drops the model caches but can't reach back into ui, so a value captured
    once at banner() goes stale. Lazily imports config/llms (ui stays a leaf); falls back to the
    banner-captured `_model` if the factory/config is unavailable."""
    try:
        from config import get_config
        from llms import model_id

        return f"{get_config().active_tier}:{model_id('tool_caller')}"
    except Exception:
        return _model


def _active_model_short() -> str:
    """The status-bar model label: the model id without the `tier:` prefix `_active_model` adds.
    The tier already rides the banner, and the bar trims from the right on a narrow terminal, so
    the prefix is pure cost here."""
    full = _active_model()
    return full.split(":", 1)[-1] if ":" in full else full


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


# ── trace verbosity ───────────────────────────────────────────────────────────
# How much of the execution trace scrolls live. The trace DB keeps everything regardless, so
# /trace and /calls stay full-fidelity no matter what this is set to:
#   "normal"  (default) — plumbing nodes (ground, update_plan) are folded out of the live rail;
#                         their *output* still prints (update_plan's plan diff is driven by
#                         show_plan), and their timing rolls into the next visible node.
#   "verbose"           — every node line, including the folded plumbing ones and full timings.
# Whether the trace renders at all is a separate switch (commands' show_ui / `/verbose off`).
_VERBOSITY = "normal"
_FOLD_NODES = ("ground", "update_plan")  # hidden from the live rail unless verbosity == "verbose"


def set_verbosity(level: str) -> str:
    """Set live-trace verbosity (\"normal\" | \"verbose\"); returns the level now in effect."""
    global _VERBOSITY
    if level in ("normal", "verbose"):
        _VERBOSITY = level
    return _VERBOSITY


def verbosity() -> str:
    return _VERBOSITY


# ── per-turn state (timing + plan diff). Reset via reset_turn() each turn. ─────
_t_last = None
_plan_seen: dict = {}
_trace_started = False  # False until the turn's first node line prints (gates one lead-in blank)

# ── live status bar (bottom-pinned) ───────────────────────────────────────────
# `_status` is the live readout the bar renders; `_turn_start` anchors the elapsed
# clock; `_live` holds the active rich.live.Live (None when torn down for input).
# `_model` is captured once in banner() so the bar needs no model passed per turn.
_turn_start = None
_status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0,
           "ctx_used": 0, "ctx_window": 0}
_model = "unknown"
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
        elapsed = time.perf_counter() - _turn_start if _turn_start else 0.0
        n = _status["tools"]
        tps = _status["tok_per_sec"]
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
        if _status["node"]:
            bar.append(f"▸ {_status['node']}", style=f"bold {_ACCENT}")
            dot()
        for i, label in enumerate((f"iter {_status['iteration']}", _fmt_dur(elapsed).strip(),
                                   f"{n} tool{'' if n == 1 else 's'}")):
            if i:
                dot()
            bar.append(label, style="default")
        if tps > 0:
            dot()
            bar.append(f"{tps:.0f} tok/s", style="default")

        # ── resources ── tertiary; ctx keeps its meter (it drives the agent), hardware is bare %.
        window = _status["ctx_window"]
        m = _metrics
        if window or m is not None:
            zone()
        if window:
            _append_meter(bar, "ctx", _status["ctx_used"] / window * 100, cells=4)
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
    global _t_last, _plan_seen, _turn_start, _status, _trace_started
    _t_last = time.perf_counter()
    _turn_start = _t_last
    _plan_seen = {}
    _trace_started = False  # next node line leads with a blank to part it from the prompt
    # Carry the last measured context fill across turns (it only grows; refreshed once the agent
    # runs) but re-read the window in case the model/tier changed since the last turn.
    _status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0,
               "ctx_used": _status.get("ctx_used", 0), "ctx_window": _active_ctx_window()}
    _live_start()


# ── small helpers ──────────────────────────────────────────────────────────────
def _emit(text) -> None:
    if _RICH:
        _console.print(text)
    else:
        print(text if isinstance(text, str) else str(text))


def _rail(style: str = _RAIL) -> "Text":
    t = Text()
    t.append(f"  {_RAIL_GLYPH} ", style=style)
    return t


def _fmt_dur(seconds: float) -> str:
    """Human-scale duration, fixed 6-char field. Humans don't benefit from sub-second precision
    on a trace, so seconds carry one decimal and sub-millisecond steps collapse to `<1ms`."""
    if seconds < 0.001:
        return "  <1ms"
    if seconds < 1:
        return f"{seconds * 1000:>4.0f}ms"
    if seconds < 60:
        return f"{seconds:>5.1f}s"
    return f"{seconds / 60:>5.1f}m"


def _fmt_args(args: dict, cap: int = 48) -> str:
    parts = []
    for k, v in (args or {}).items():
        r = repr(v)
        if len(r) > cap:
            r = r[: cap - 1] + "…"
        parts.append(f"{k}={r}")
    return ", ".join(parts)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _term_width(default: int = 80) -> int:
    """Current console width, for width-responsive truncation/wrapping in the trace. Falls back
    safely so a detached or odd stdout never throws."""
    try:
        if _RICH:
            w = _console.width
            return w if w and w >= 20 else default
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _git_branch() -> str:
    """Current git branch for the banner, or "" if not a repo / git missing. Best-effort, short
    timeout — never blocks startup."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=0.5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _short_cwd() -> str:
    """Current working dir with $HOME collapsed to ~, for a compact banner line."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    return cwd


# ── startup splash (the one ornamental moment) ───────────────────────────────────
# A tilted cyan ring draws itself out around a dark, banded gas-giant — Saturn, for the `saturn`
# launch word. Deliberately the only decorative element in an otherwise utilitarian console: it
# plays once at startup, obeys the same palette (cyan = live/structure, grey = the inert body),
# and is fully skippable. The sphere is a real diffuse-shaded ball and the ring a real tilted
# ellipse — A ≫ B — so its back arc genuinely slips behind the body rather than reading as
# clip-art. Set SATURDAY_NO_SPLASH=1 to suppress it, SATURDAY_NO_ANIM=1 to print only the
# resting frame; it also auto-skips when stdout isn't a terminal or is too narrow.
_ART_R, _ART_C = 15, 56                   # canvas rows, cols
_ART_PCY, _ART_PCX = 7.0, 28.0            # planet centre
_ART_PRR, _ART_PRC = 4.5, 9.0             # planet radii (rows, cols) — cols ~2× for char aspect
_ART_RING_A, _ART_RING_B = 24.0, 5.7      # main ring semi-axes; B sets how open the tilt reads
_ART_GUIDE_A, _ART_GUIDE_B = 27.5, 6.5    # outer dashed guide ring, echoing the reference art
_ART_RING_SKEW = 0.12                     # slight rotation of the projected ring plane (radians)
_ART_RAMP = " .:-=+*#%@"                  # dark → light sphere shading ramp
_ART_LIGHT = (-0.5, -0.65, 0.6)           # light direction (x, y=screen-down, z)
# ramp index → grey shade. Kept dark and monochrome on purpose: the body recedes so the lone
# cyan ring is the only saturated thing on screen ("color means something", as the trace follows).
_ART_SHADE = ["", "grey15", "grey19", "grey23", "grey27",
              "grey30", "grey35", "grey39", "grey42", "grey46"]
_ART_STARS = [(0, 7), (1, 49), (2, 14), (12, 43), (11, 10), (3, 53), (9, 4)]  # faint backdrop


def _norm3(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / m for c in v)


_ART_LN = _norm3(_ART_LIGHT)


def _sphere_cell(r: int, c: int):
    """Diffuse-shaded gas-giant cell at (r, c), or None if the cell is outside the disc. Carries
    faint horizontal latitude bands and a lit rim along the top edge so the dark body still reads
    as a sphere."""
    nx = (c - _ART_PCX) / _ART_PRC
    ny = (r - _ART_PCY) / _ART_PRR
    rad = nx * nx + ny * ny
    if rad > 1.0:
        return None
    nz = math.sqrt(max(0.0, 1.0 - rad))
    diff = max(0.0, nx * _ART_LN[0] + ny * _ART_LN[1] + nz * _ART_LN[2])
    band = 0.84 + 0.16 * math.sin(ny * 6.5 + 0.5)   # gas-giant latitude banding
    b = 0.12 + diff * band * 0.78
    if rad > 0.80 and ny < 0.15:                    # faint lit rim along the upper silhouette
        b = max(b, 0.6)
    if rad > 0.90:                                   # soft edge fade — dims the outer rim
        b *= (1.0 - rad) / 0.10
    b = max(0.0, min(1.0, b))
    idx = int(b * (len(_ART_RAMP) - 1) + 0.5) or 1  # never blank inside the silhouette
    return _ART_RAMP[idx], _ART_SHADE[idx]



def _ring_path(a: float, b: float, n: int = 360):
    """Full set of projected ring samples `(i, t, rr, cc, front)`. The ellipse is rotated by
    `_ART_RING_SKEW` so the plane reads as skewed (one ansa lifts above the other); `front` (the
    near, lower arc that draws over the body) stays keyed to the un-skewed `sin(t)` depth."""
    cs, sn = math.cos(_ART_RING_SKEW), math.sin(_ART_RING_SKEW)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = a * math.cos(t), b * math.sin(t)
        cc = _ART_PCX + x * cs - y * sn
        rr = _ART_PCY + x * sn + y * cs
        out.append((i, t, rr, cc, math.sin(t) > 0))
    return out


_RING_PATH = _ring_path(_ART_RING_A, _ART_RING_B)
_GUIDE_PATH = _ring_path(_ART_GUIDE_A, _ART_GUIDE_B)


def _saturn_grid(main_style, guide_vis):
    """Layered (char, style) grid: backdrop → back ring → planet → front ring. `main_style(t)`
    returns the core style for a main-ring sample (or None to leave it un-drawn); `guide_vis(i,t)`
    gates the dashed outer ring. The planet is laid between the back and front arcs, so the tilted
    ring genuinely slips behind the body."""
    grid = [[(" ", None) for _ in range(_ART_C)] for _ in range(_ART_R)]

    def put(r, c, ch, st):
        r, c = int(round(r)), int(round(c))
        if 0 <= r < _ART_R and 0 <= c < _ART_C:
            grid[r][c] = (ch, st)

    for r, c in _ART_STARS:
        put(r, c, "·", "grey27")

    def draw_guide(want_front):              # dashed outer ring
        for i, t, rr, cc, front in _GUIDE_PATH:
            if front == want_front and guide_vis(i, t):
                put(rr, cc, "·", "bright_cyan")

    def draw_main(want_front):               # glowing main band (bright core + cyan halo)
        for i, t, rr, cc, front in _RING_PATH:
            if front != want_front:
                continue
            st = main_style(t)
            if st is None:
                continue
            put(rr + (1 if front else -1), cc, "·", "bright_cyan")  # halo, toward the outside
            put(rr, cc, "•", st)                                     # bright core

    draw_guide(False)                        # back arcs (behind the planet)
    draw_main(False)
    for r in range(_ART_R):                  # the planet itself
        for c in range(_ART_C):
            cell = _sphere_cell(r, c)
            if cell:
                put(r, c, cell[0], cell[1])
    draw_main(True)                          # front arcs (over the planet)
    draw_guide(True)
    return grid


def _saturn_cells(progress: float, final: bool):
    """Still / draw-out frame: the ring is filled up to angle `progress`·2π (full when `final`)."""
    cut = 2 * math.pi if final else max(progress, 0.0) * 2 * math.pi

    def main_style(t):
        if t > cut + 1e-9:
            return None
        if not final and (cut - t) < 0.16:  # bright leading edge as it draws in
            return "bold white"
        return "bold bright_cyan"

    def guide_vis(i, t):
        return t <= cut + 1e-9 and i % 7 < 4

    return _saturn_grid(main_style, guide_vis)


# Continuous loop: a fixed-length lit arc (with a bright comet head) chases a travelling gap
# around the ring, so it reads as the ring perpetually drawing itself out — seamless, no snap
# back to empty. `tail` lets the closing beat fill the gap to a complete ring.
_ANIM_TAIL = 2 * math.pi * 0.80


def _saturn_anim_cells(phase: float, tail: float = _ANIM_TAIL):
    head = phase % (2 * math.pi)

    def behind(t):                           # angular distance back from the head, in [0, 2π)
        return (head - t) % (2 * math.pi)

    def main_style(t):
        d = behind(t)
        if d > tail:                         # the travelling, un-drawn gap
            return None
        if d < 0.22:                         # bright comet head
            return "bold white"
        return "bold bright_cyan"

    def guide_vis(i, t):
        return behind(t) <= tail and i % 7 < 4

    return _saturn_grid(main_style, guide_vis)


def _grid_text(grid) -> "Text":
    # No trailing newline: a final "\n" makes a printed frame one line taller than its content.
    t = Text()
    for ri, row in enumerate(grid):
        if ri:
            t.append("\n")
        t.append("  ")  # left indent, in line with the trace rail
        for ch, st in row:
            t.append(ch, style=st or "default")
    return t


# In-place animation player. rich's Live (in the normal buffer) repaints its whole region every
# frame, which flickers on Windows consoles; the alternate-screen buffer hides that but is jarring
# to switch in and out of. Instead we stay in the shell scrollback and rewrite only the lines that
# actually changed since the previous frame, with the cursor hidden — so each frame touches just a
# few rows and there's nothing to tear.
class _InlinePlayer:
    def __init__(self, out):
        self._out = out
        self._buf = io.StringIO()
        self._render = Console(file=self._buf, force_terminal=True, highlight=False,
                               width=_ART_C + 4, color_system=_console.color_system or "standard")
        self._prev = None  # the previous frame's per-row ANSI strings

    def _rows(self, grid):
        rows = []
        for row in grid:
            t = Text("  ")  # left indent, in line with the trace rail
            for ch, st in row:
                t.append(ch, style=st or "default")
            self._buf.seek(0)
            self._buf.truncate(0)
            self._render.print(t, end="")
            rows.append(self._buf.getvalue())
        return rows

    def draw(self, grid):
        rows = self._rows(grid)
        n = len(rows)
        if self._prev is None:                       # first frame: lay the whole block down
            self._out.write("\x1b[?25l" + "\r\n".join(rows))
        else:                                        # later frames: rewrite only changed rows
            self._out.write(f"\r\x1b[{n - 1}A")      # cursor to the block's top line
            for i, row in enumerate(rows):
                if row != self._prev[i]:
                    self._out.write("\r" + row + "\x1b[K")
                if i < n - 1:
                    self._out.write("\x1b[B")        # step down a row without scrolling
            self._out.write("\r")
        self._out.flush()
        self._prev = rows

    def clear(self):
        """Wipe the block and park the cursor back at its origin, so the settled frame prints in
        the same spot — no jump — then restore the cursor."""
        if self._prev is not None:
            n = len(self._prev)
            self._out.write(f"\r\x1b[{n - 1}A")
            for i in range(n):
                self._out.write("\x1b[2K")
                if i < n - 1:
                    self._out.write("\x1b[B")
            self._out.write(f"\x1b[{n - 1}A\r")
        self._out.write("\x1b[?25h")
        self._out.flush()


def _saturn_text(progress: float, final: bool) -> "Text":
    return _grid_text(_saturn_cells(progress, final))


def _saturn_plain() -> str:
    lines = ["  " + "".join(ch for ch, _ in row).rstrip()
             for row in _saturn_cells(1.0, final=True)]
    return "\n".join(lines)


def splash(work=None):
    """Play the startup ring animation, then settle on its resting frame. If `work` (a zero-arg
    callable — the slow startup loading) is given, it runs on a background thread while the ring
    keeps drawing itself out in a smooth loop, and the animation holds until it finishes; its
    return value is passed back (its exception re-raised). Best-effort and non-fatal: a
    non-terminal stdout, a too-narrow window, or SATURDAY_NO_SPLASH still runs `work`, just
    without the art, so it can never wedge launch."""
    import sys
    import threading

    box = {"value": None, "exc": None}
    def _run():
        try:
            box["value"] = work() if work else None
        except BaseException as exc:        # noqa: BLE001 — surfaced to the caller below
            box["exc"] = exc

    def _finish():
        if box["exc"] is not None:
            raise box["exc"]
        return box["value"]

    quiet = bool(os.environ.get("SATURDAY_NO_SPLASH")) or not _RICH \
        or _console.size.width < _ART_C + 2 \
        or not _console.is_terminal or bool(os.environ.get("SATURDAY_NO_ANIM"))

    if quiet:
        if not _RICH and not os.environ.get("SATURDAY_NO_SPLASH"):
            print(_saturn_plain())
        _run()                              # no animation: just do the work, then settle
        if _RICH and not os.environ.get("SATURDAY_NO_SPLASH") \
                and _console.size.width >= _ART_C + 2:
            _console.print(_saturn_text(1.0, final=True))
        return _finish()

    # The worker's stdout/stderr (ingest logs, etc.) is captured for the duration of the animation
    # and replayed *beneath* the settled ring afterward — otherwise those prints would land in the
    # middle of the in-place ring region from a second thread and tear it apart.
    captured = io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    player = _InlinePlayer(real_out)
    worker = threading.Thread(target=_run, daemon=True)
    phase, step = 0.0, 2 * math.pi / 80     # ~one lap every 1.8s
    try:
        sys.stdout = sys.stderr = captured
        worker.start()
        # Loop the draw-out until the work is done (or, with no work, a couple of laps).
        while worker.is_alive() or (work is None and phase < 4 * math.pi):
            player.draw(_saturn_anim_cells(phase))
            time.sleep(0.022)
            phase += step
        # Closing beat: fill the travelling gap so the ring completes to a whole circle.
        for k in range(20):
            tail = _ANIM_TAIL + (2 * math.pi - _ANIM_TAIL) * (k + 1) / 20
            player.draw(_saturn_anim_cells(phase, tail=tail))
            time.sleep(0.018)
            phase += step
    except Exception:
        pass  # a half-drawn splash must never break startup; fall through to the still frame
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        try:
            player.clear()                  # wipe the animation, park the cursor at its origin
        except Exception:
            pass
    worker.join()
    _console.print(_saturn_text(1.0, final=True))   # the settled ring, in the same spot …
    spill = captured.getvalue()
    if spill.strip():                                # … then the loading output beneath it
        real_out.write(spill if spill.endswith("\n") else spill + "\n")
        real_out.flush()
    return _finish()


def play_animation() -> None:
    """Loop the Saturn ring animation until the user presses Ctrl+C, then settle the resting frame."""
    import sys

    _live_stop()

    if not _RICH or not _console.is_terminal or _console.size.width < _ART_C + 2:
        if _RICH:
            _console.print(_saturn_text(1.0, final=True))
        else:
            print(_saturn_plain())
        return

    hint = Text("  ")
    hint.append("Ctrl+C", style=f"bold {_ACCENT}")
    hint.append(" to stop", style=_DIM)
    _console.print(hint)

    out = sys.stdout
    player = _InlinePlayer(out)
    phase, step = 0.0, 2 * math.pi / 80
    try:
        while True:
            player.draw(_saturn_anim_cells(phase))
            time.sleep(0.022)
            phase += step
    except KeyboardInterrupt:
        pass
    finally:
        try:
            player.clear()
        except Exception:
            pass
    _console.print(_saturn_text(1.0, final=True))


# ── startup banner ─────────────────────────────────────────────────────────────
def banner(model: str, n_tools: int, n_docs: int, db_path: str) -> None:
    """Session header: a light two-line identity block — model/tier · context window, then tool +
    doc counts · git branch · working dir, with the `/help` hint folded onto the second line. No
    box and no absolute trace-DB path (it widened the old card for a value that's a `/config` away);
    just alignment and whitespace, in the dim/accent palette. `db_path` is accepted for callers /
    future relocation but isn't rendered here."""
    global _model
    _model = model  # captured here so the live status bar needs no model passed per turn
    win = _active_ctx_window()
    ctx = _human_tokens(win) if win else "?"
    branch = _git_branch()
    cwd = _short_cwd()

    if _RICH:
        head = Text("  ")
        head.append("saturday.ai", style=f"bold {_ACCENT}")
        head.append("   ", style=_DIM)
        head.append(model, style="default")
        head.append("  ·  ctx ", style=_DIM)
        head.append(ctx, style="default")
        _console.print(head)

        info = Text("  ")
        info.append(f"{n_tools} tools", style="default")
        info.append("  ·  ", style=_DIM)
        info.append(f"{n_docs} docs", style="default")
        if branch:
            info.append("  ·  ", style=_DIM)
            info.append(f"git {branch}", style="default")
        info.append("  ·  ", style=_DIM)
        info.append(cwd, style=_DIM)
        info.append("      ", style=_DIM)
        info.append("/help", style=_ACCENT)
        info.append(" for commands", style=_DIM)
        _console.print(info)
    else:
        git = f"  ·  git {branch}" if branch else ""
        print(f"saturday.ai  {model}  ·  ctx {ctx}")
        print(f"{n_tools} tools  ·  {n_docs} docs{git}  ·  {cwd}      /help for commands")


# ── input prompt ───────────────────────────────────────────────────────────────
# Live highlight for the `»` line: a `/token` is colored by how it matches the command set, so
# a typo never blends in with a real command. Valid -> cyan, a prefix of some command (mid-type)
# -> yellow, anything else -> red. Args after the token stay dim. Built only when prompt_toolkit
# is present; the palette mirrors the rest of ui.py (cyan accent, semantic status colors).
if _PTK:
    _PTK_STYLE = _PTKStyle.from_dict({
        "prompt": "ansicyan bold",
        "prompt.cont": "ansibrightblack",
        "cmd.valid": "ansicyan bold",
        "cmd.partial": "ansiyellow",
        "cmd.unknown": "ansired bold",
        "cmd.args": "ansibrightblack",
        "mention": "ansibrightblue",
    })

    # An `@mention` inside a normal (non-slash) line: `@` at a word boundary + a run of
    # non-space/non-`@`. Highlighted live so the user sees which token will attach a file
    # (mentions.expand resolves them for real at submit; see mentions.py).
    import re as _re
    _MENTION_LEX_RE = _re.compile(r"(?<!\S)@[^\s@]+")
    # Same grammar but capturing just the path fragment up to the cursor, for Tab completion.
    _AT_FRAGMENT_RE = _re.compile(r"(?:^|\s)@([^\s@]*)$")

    def _mention_fragments(line: str):
        """Split one plain line into prompt_toolkit (style, text) fragments, coloring any
        `@mention` runs. Used for normal turns so file mentions stand out as they're typed."""
        frags = []
        pos = 0
        for m in _MENTION_LEX_RE.finditer(line):
            if m.start() > pos:
                frags.append(("", line[pos:m.start()]))
            frags.append(("class:mention", m.group(0)))
            pos = m.end()
        if pos < len(line):
            frags.append(("", line[pos:]))
        return frags or [("", line)]

    def _slash_token(text: str):
        """Split a prompt line into `(lead, token, args)` around the leading `/command` word —
        `lead` is any whitespace before the slash, `token` the command word (no slash, original
        case), `args` the remainder (its leading space included). Returns `None` for a non-slash
        line. The single definition of the `/token` grammar, shared by the lexer and the completer."""
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return None
        lead = text[: len(text) - len(stripped)]  # preserve leading whitespace verbatim
        body = stripped[1:]
        cut = len(body)
        for i, ch in enumerate(body):
            if ch.isspace():
                cut = i
                break
        return lead, body[:cut], body[cut:]

    class _CommandLexer(_PTKLexer):
        """Colors the first `/token` of the line against a known-command set, live as it's typed.
        Only the command token is styled; normal (non-slash) turns render plain."""

        def __init__(self, names):
            self._names = names  # canonical names + aliases, lowercased, no leading slash

        def _style_for(self, key: str) -> str:
            if key in self._names:
                return "class:cmd.valid"
            if not key or any(n.startswith(key) for n in self._names):
                return "class:cmd.partial"  # lone "/" or still typing a real command
            return "class:cmd.unknown"      # a typo — make it loud

        def lex_document(self, document):
            # Per-line so multiline input (Alt+Enter / paste) highlights each line. Only the
            # FIRST line is treated as a possible `/command` (a slash command is always single-
            # line); every line gets `@mention` highlighting.
            lines = document.lines

            def get_line(lineno):
                line = lines[lineno] if 0 <= lineno < len(lines) else ""
                parsed = _slash_token(line) if lineno == 0 else None
                if parsed is None:
                    return _mention_fragments(line)
                lead, token, args = parsed
                frags = []
                if lead:
                    frags.append(("", lead))
                frags.append((self._style_for(token.lower()), "/" + token))
                if args:
                    frags.append(("class:cmd.args", args))
                return frags

            return get_line

    class _CommandCompleter(_PTKCompleter):
        """Tab-completes two things, depending on where the cursor sits:
          - the leading `/command` token of a slash line (against the known command set), and
          - an `@path` mention anywhere in a normal line (against the filesystem, so a file can be
            pulled into the turn inline — see mentions.py).
        Command completion fires only on the first token of a slash line (a space ends it); `@path`
        completion fires on the partial path after an `@` at a word boundary. `display_meta` carries
        each command's one-line summary into the menu."""

        def __init__(self, meta):
            self._meta = meta  # list of (token, summary), tokens lowercased, no leading slash
            # Delegate the actual path matching to prompt_toolkit's PathCompleter; we only carve
            # out the `@`-prefixed fragment and re-base its replacement offset.
            self._paths = _PTKPathCompleter(expanduser=True)

        def get_completions(self, document, complete_event):
            # `@path` mention completion takes precedence: it can occur anywhere on the line,
            # including inside what would otherwise be a slash command's args.
            before = document.text_before_cursor
            at = _AT_FRAGMENT_RE.search(before)
            if at is not None:
                frag = at.group(1)  # the partial path typed after `@` (no `@`)
                sub = _PTKDocument(frag, len(frag))
                for comp in self._paths.get_completions(sub, complete_event):
                    yield comp  # offsets are relative to `frag`'s end == the cursor, so they map 1:1
                return

            parsed = _slash_token(before)
            if parsed is None:
                return
            _lead, token, args = parsed
            if args:  # past the command token, into the args
                return
            word = token.lower()
            for tok, summary in self._meta:
                if tok.startswith(word):
                    # Replace just the typed token (the leading "/" stays put).
                    yield _PTKCompletion(
                        tok,
                        start_position=-len(token),
                        display="/" + tok,
                        display_meta=summary,
                    )

    # Multiline input: Enter submits, Alt+Enter inserts a newline. So a pasted multi-line block
    # (bracketed paste inserts its newlines directly, never submitting) or a deliberately-composed
    # multi-paragraph turn survives instead of being chopped at the first newline. When the
    # completion menu is open, Enter accepts the highlighted completion rather than submitting.
    _PTK_KB = _PTKKeyBindings()

    @_PTK_KB.add("enter")
    def _ptk_enter(event):
        buf = event.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
        else:
            buf.validate_and_handle()  # submit the line

    @_PTK_KB.add("escape", "enter")  # Alt/Option+Enter (and Esc then Enter) -> hard newline
    def _ptk_newline(event):
        event.current_buffer.insert_text("\n")

    def _ptk_continuation(width, line_number, is_soft_wrap):
        """Gutter for continuation lines of a multiline entry — a dim `·` aligned under the `»`."""
        return [("class:prompt.cont", "· ".rjust(width))] if not is_soft_wrap else ""

    _ptk_session = None  # one PromptSession for the process -> free line history across turns


def prompt(command_meta=None) -> str:
    """Read the `»` input line. With prompt_toolkit and a `command_meta` list of `(token, summary)`
    pairs, a typed `/command` is highlighted live (valid=cyan, typo=red), Tab completes the leading
    `/command` token or an `@path` mention, and the line is multiline (Enter submits, Alt+Enter adds
    a newline — so pasted/multi-paragraph input survives). Without prompt_toolkit, falls back to
    rich/plain input. Returns the raw line (slash-command + @mention handling happen upstream)."""
    _live_stop()  # never read a line under an active Live (also clears a bar left by an error)
    if _PTK and command_meta is not None:
        global _ptk_session
        if _ptk_session is None:
            _ptk_session = PromptSession()
        names = {token for token, _ in command_meta}  # valid-command set for the live highlight
        return _ptk_session.prompt(
            [("class:prompt", "» ")],
            lexer=_CommandLexer(names),
            style=_PTK_STYLE,
            completer=_CommandCompleter(command_meta),
            complete_while_typing=False,  # Tab-triggered, so the menu never fights live typing
            multiline=True,
            key_bindings=_PTK_KB,
            prompt_continuation=_ptk_continuation,
        )
    if _RICH:
        return _console.input(f"[bold {_ACCENT}]»[/] ")
    return input("» ")


# ── execution trace ─────────────────────────────────────────────────────────────
def _metric_parts(delta: dict) -> list[str]:
    """Per-node metric annotations (iteration · context tokens · tok/s) pulled from a node delta —
    shared by the live trace (show_node) and the recorded replay (show_run)."""
    parts = []
    if "iteration" in delta:
        parts.append(f"iter {delta['iteration']}")
    used = delta.get("context_tokens") or 0
    if used > 0:
        parts.append(f"{_human_tokens(used)} ctx")
    tps = delta.get("tok_per_sec") or 0.0
    if tps > 0:
        parts.append(f"{tps:.0f} tok/s")
    return parts


def _node_line(node: str, dur: float, delta: dict) -> "Text | str":
    """Build one `│ ✓ node  elapsed  metrics` trace row (metrics dim) — the shared format for the
    live trace and the /trace replay. Returns a rich Text, or a plain string without rich."""
    extra = " · ".join(_metric_parts(delta))
    if _RICH:
        line = _rail()
        line.append("✓ ", style="green")  # the node has finished by the time its line prints
        line.append(f"{node:<{_NODE_W}}", style="default")
        line.append(f"{_fmt_dur(dur):>7}", style=_DIM)
        if extra:
            line.append(f"   {extra}", style=_DIM)  # metrics are tertiary — dim, never the accent
        return line
    tail = f"   {extra}" if extra else ""
    return f"  {_RAIL_GLYPH} ✓ {node:<{_NODE_W}}{_fmt_dur(dur):>7}{tail}"


def show_node(node: str, delta: dict | None = None) -> None:
    """One trace line per node execution — `│ <node>  <elapsed>  <annotation>` — with the elapsed
    measured since the previous node emitted (htop-style). LLM nodes annotate with iter / context
    tokens / tok-per-sec; the `tools` node renders a sub-tree of its calls (args · timing · result
    preview) beneath the header, so the agent's actual actions are fully visible, not hidden."""
    global _t_last, _trace_started
    now = time.perf_counter()
    dur = now - _t_last if _t_last is not None else 0.0
    _t_last = now

    # plan_gate is a control checkpoint, not an informative node — skip its rail line so the
    # per-step pass-throughs don't double the trace. Its effects still surface elsewhere: a plan
    # edit via show_plan (the on_update subscriber calls it on a `plan` delta), a pause via the
    # plan-review prompt. _t_last is already advanced, so the next node's timing excludes the gate.
    # The plumbing nodes (ground, update_plan) fold the same way at normal verbosity: their timing
    # rolls into the next visible node, and update_plan's plan diff still prints (show_plan is
    # driven separately by on_update). Everything stays in the trace DB for /trace and /calls.
    if node == "plan_gate":
        return
    if node in _FOLD_NODES and _VERBOSITY != "verbose":
        return

    delta = delta or {}
    # Feed the pinned status bar from whatever this delta carried.
    called = delta.get("tools_called") or []
    _status["tools"] += len(called)
    if "iteration" in delta:
        _status["iteration"] = delta["iteration"]
    tps = delta.get("tok_per_sec") or 0.0
    if tps > 0:
        _status["tok_per_sec"] = tps
    used = delta.get("context_tokens") or 0
    if used > 0:
        _status["ctx_used"] = used
    _status["node"] = node

    # The synthesize node's output IS the answer — it's streamed/printed by ResponseStream (or
    # `response`), so it gets no rail line of its own (that line used to land between the response
    # rule and the streamed answer). Its metrics above still feed the live bar + the post-turn
    # receipt, and the /trace replay still renders it as a node.
    if node == "synthesize":
        _live_refresh()
        return

    # Per-node trace row: `│ ✓ node  elapsed  metrics` (metrics dim). The metric annotations are
    # built from the delta by the shared _node_line helper (the live trace + the /trace replay
    # render identical rows).
    if not _trace_started:
        _emit("")  # one blank line parting the turn's trace from the prompt above it
        _trace_started = True
    _emit(_node_line(node, dur, delta))

    if delta.get("tool_events"):
        _render_tool_events(delta["tool_events"])

    _live_refresh()  # repaint the bar with the new node/iter/tools immediately


def _emit_result_leaf(cont: str, text: str, style: str) -> None:
    """Emit a tool result/error leaf under its call branch, word-wrapped to the terminal with a
    HANGING INDENT: the first line carries the `└` leaf glyph, continuation lines indent to sit
    under the text (keeping the `cont` rail gutter), so a long output stays inside the trace rail
    instead of spilling to column 0."""
    import textwrap

    first = f"{cont}  {_TREE_LEAF} "   # "│  └ " / "   └ " — leaf glyph, under the call text
    rest = f"{cont}    "               # "│    " / "     " — aligns continuation under the text
    avail = max(20, _term_width() - (4 + 2 + len(first)))  # minus rail(4) + nest(2) + leaf prefix
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        prefix = first if i == 0 else rest
        if _RICH:
            row = _rail()
            row.append("  ", style=_RAIL)
            row.append(prefix, style=_RAIL)
            row.append(ln, style=style)
            _console.print(row)
        else:
            print(f"  {_RAIL_GLYPH}   {prefix}{ln}")


def _render_tool_events(events: list[dict], *, always_show_results: bool = False) -> None:
    """Draw the tool-I/O sub-tree under the `tools` node header: one `├─ name(args)  dur` branch
    per call, the call repr sized to the terminal and durations column-aligned within the round so
    they read as a column. The raw result preview is **hidden** by default — it's noisy JSON, and
    `/calls` (or `/verbose full`) surfaces full outputs on demand — but a FAILED call still shows
    its error leaf inline (signal, not noise). What the agent *did* (name · args · cost · ok/fail)
    always stays visible. `always_show_results=True` (the /trace replay) shows every output, word-
    wrapped under the rail with a hanging indent."""
    n = len(events)
    # Width-responsive: size the call repr to the room left after the tree prefix (~9) and the right
    # `   dur` column (~9), then align durations to the widest call in this round.
    call_cap = max(24, _term_width() - 18)
    calls = [_truncate(f"{ev.get('name', '?')}({_fmt_args(ev.get('args', {}))})", call_cap)
             for ev in events]
    col_w = max((len(c) for c in calls), default=0)
    for i, ev in enumerate(events):
        last = i == n - 1
        branch = _TREE_END if last else _TREE_MID
        cont = " " if last else _TREE_PIPE  # gutter under the branch for the result/error leaf
        call = calls[i]
        dur = _fmt_dur(ev.get("dur", 0.0))
        ok = ev.get("ok", True)
        result = ev.get("result", "")
        # Outputs are hidden by default; show only errors, or everything under /verbose full or in
        # the /trace replay (always_show_results).
        show_result = bool(result) and (always_show_results or not ok or _VERBOSITY == "verbose")

        if _RICH:
            line = _rail()
            line.append("  ", style=_RAIL)            # nest under the node column
            line.append(f"{branch} ", style=_RAIL)
            line.append(f"{call:<{col_w}}", style="default" if ok else "red")
            line.append(f"   {dur}", style=_DIM)
            _console.print(line)
        else:
            print(f"  {_RAIL_GLYPH}   {branch} {call:<{col_w}}   {dur}")
        if show_result:
            _emit_result_leaf(cont, result, _DIM if ok else "red")


_MSG_ROLE = {"AIMessage": "ai", "HumanMessage": "in", "SystemMessage": "sys"}


def _msg_kind_content(m) -> tuple[str, str]:
    """Normalize one delta message to `(kind, content)` — handling BOTH forms it can take:
      - a live LangChain message OBJECT (the live trace: `delta["messages"]` straight off the
        graph), or
      - the trace DB's pre-serialized `"AIMessage: <text> [tool_calls: ...]"` STRING (the /trace
        replay: deltas are JSON, and `stores.trace._json_default` flattened each message to a string).
    Returning the same `(kind, content)` for both keeps the live and replay rendering identical.
    The object branch mirrors `_json_default`'s format (content + a `[tool_calls: …]` suffix) so a
    content-less tool-calling turn still records WHAT the agent decided."""
    if not isinstance(m, str):
        kind = type(m).__name__
        content = str(getattr(m, "content", "") or "")
        calls = getattr(m, "tool_calls", None)
        if calls:
            names = ", ".join(c.get("name", "?") for c in calls)
            content = (content + " " if content else "") + f"[tool_calls: {names}]"
        return kind, content
    kind, _, content = m.partition(": ")
    return kind.strip(), content


def _emit_message_leaf(label: str, text: str) -> None:
    """One message/reasoning leaf under a node row: `└ <role>  <wrapped text>`, hanging-indented
    under the rail so a long thought stays inside the trace gutter. Dim — it's narrative, not the
    accent. Used by the /trace replay to surface the agent's actual thinking between tool calls."""
    import textwrap

    head = f"  {_TREE_END} {label:<3} "   # nest(2) + leaf glyph + fixed-width role tag
    rest = " " * len(head)               # continuation lines align under the text
    avail = max(20, _term_width() - (4 + len(head)))
    for i, ln in enumerate(textwrap.wrap(text, width=avail) or [text]):
        if _RICH:
            row = _rail()
            row.append(head if i == 0 else rest, style=_RAIL)
            row.append(ln, style=_DIM)
            _console.print(row)
        else:
            print(f"  {_RAIL_GLYPH} {head if i == 0 else rest}{ln}")


def _render_trace_messages(node: str, delta: dict, max_chars: int | None = None) -> None:
    """Render the messages a node ADDED — chiefly the agent's reasoning text and its tool-call
    decisions — as dim leaves under its trace row. This is the piece the default tool tree never
    surfaces, and what turns the /trace replay from a reprint of the answer into a real execution
    log. ToolMessages are skipped (the tool sub-tree already carries their output) and the
    synthesize node's message is skipped (it's the final answer, shown once in the response section
    below). Used by BOTH the /trace replay (full fidelity, `max_chars=None`) and the live trace
    (`show_reasoning`, a bounded preview) — `_msg_kind_content` normalizes the two message forms so
    they render identically. `max_chars` clips each leaf to a preview (the full text lives in the
    /trace replay)."""
    if node == "synthesize":
        return
    for m in (delta.get("messages") or []):
        kind, content = _msg_kind_content(m)
        if "ToolMessage" in kind:
            continue
        content = " ".join(content.split())  # collapse to a compact one-block preview
        if not content:
            continue
        if max_chars and len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        _emit_message_leaf(_MSG_ROLE.get(kind, kind.lower() or "msg"), content)


# Live-trace reasoning preview cap. The agent's between-tool reasoning can run long (a thinking
# model especially), so live we show a bounded preview — enough to read the model's intent without
# flooding the rail; the full text is always in the /trace replay. Lifted at "verbose".
_LIVE_REASONING_CAP = 500


def show_reasoning(node: str, delta: dict | None = None) -> None:
    """Live counterpart to the /trace replay's reasoning leaves: render the agent's between-tool
    reasoning (and its tool-call decision) under the node's trace row, as it happens — the
    "show me the thinking" detail the live rail otherwise omits until you open /trace. Shown at
    normal verbosity as a bounded preview (_LIVE_REASONING_CAP) and in full at "verbose"; skipped
    for nodes that print no rail row (synthesize, plan_gate, and the folded plumbing nodes) so a
    leaf never dangles under a row that isn't there. The caller only invokes this when show_ui is
    on, so `/verbose off` (trace hidden) suppresses it for free."""
    if node in ("synthesize", "plan_gate"):
        return
    if node in _FOLD_NODES and _VERBOSITY != "verbose":
        return
    cap = None if _VERBOSITY == "verbose" else _LIVE_REASONING_CAP
    _render_trace_messages(node, delta or {}, max_chars=cap)


def _plan_line(step: dict, *, show_tool: bool) -> "Text | str":
    status = step.get("status", "pending")
    glyph, style = _PLAN.get(status, _PLAN["pending"])
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")

    # Width-responsive: drop the ::tool tag on a very narrow terminal, then size the label to what's
    # left after the prefix (rail+nest+glyph+id ≈ 12) and the tag, so a step stays on one row.
    tw = _term_width()
    tag = f"  ::{tool}" if (show_tool and tool and tw >= 56) else ""
    label = _truncate(str(step.get("label", "")), max(20, tw - 14 - len(tag)))

    if _RICH:
        line = _rail()
        line.append("  ", style=_RAIL)  # nest steps under the node
        line.append(f"{glyph} ", style=style)
        line.append(f"{str(sid):>2}  ", style=_DIM)
        line.append(label, style=style if status in ("active", "skipped") else "default")
        if tag:
            line.append(tag, style=_FAINT)  # the most incidental annotation — faintest
        return line
    return f"  {_RAIL_GLYPH}   {glyph} {str(sid):>2}  {label}{tag}"


def render_plan(plan) -> None:
    """Print the full plan unconditionally — every step, with its intended tool. Unlike
    `show_plan` this does no diffing and touches no per-turn state, so it's the right call for the
    `/plan` command (inspect the last plan on demand, outside the live trace)."""
    if not plan:
        _emit("  (no plan yet — run a turn first)")
        return
    for step in plan:
        _emit(_plan_line(step, show_tool=True))


def show_plan(plan) -> None:
    """First call this turn: print the whole plan as the intended route (with tools).
    Later calls: print only the steps whose status changed — one line each, like a trace.
    This keeps the plan transparent without re-rendering a panel on every node update."""
    global _plan_seen
    if not plan:
        return

    first_render = not _plan_seen
    for step in plan:
        sid = step.get("step_id")
        status = step.get("status", "pending")
        if first_render:
            _emit(_plan_line(step, show_tool=True))
            _plan_seen[sid] = status
        elif _plan_seen.get(sid) != status:
            _emit(_plan_line(step, show_tool=False))
            _plan_seen[sid] = status


# ── approval gate (the one place that gets to shout) ─────────────────────────────
# Cap the diff preview so a huge rewrite can't flood the gate; the agent still sees the full
# content, this is just the human-facing safety preview.
_MAX_DIFF_LINES = 60


def _workspace_old_text(file_path: str) -> "tuple[str, bool]":
    """Current contents of a workspace file (for the write_file diff preview) + whether it exists.
    Resolved exactly like the write_file tool (sandboxed to the workspace), so the preview matches
    what the write will actually touch. Any failure degrades to ('', False) — the preview is
    best-effort and must never block the gate."""
    try:
        from config import get_config

        workspace = get_config().path("workspace")
        target = (workspace / file_path).resolve()
        if not target.is_relative_to(workspace) or not target.exists():
            return "", False
        return target.read_text(encoding="utf-8", errors="replace"), True
    except Exception:
        return "", False


def _diff_lines(file_path: str, content: str, overwrite: bool) -> "tuple[list, bool, int]":
    """Build the unified-diff rows for a pending write_file. Returns (rows, is_new_file,
    hidden_count) where each row is (kind, text), kind ∈ {add, del, hunk, ctx}. An append
    (overwrite=False) diffs old-vs-(old+content) so the appended text reads as additions."""
    import difflib

    old, existed = _workspace_old_text(file_path)
    new = content if overwrite else (old + content)
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    rows: list = []
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=2))
    for line in diff[2:]:  # skip the two file-name headers positionally (content may start with +++/---)
        if line.startswith("@@"):
            rows.append(("hunk", line))
        elif line.startswith("+"):
            rows.append(("add", line[1:]))
        elif line.startswith("-"):
            rows.append(("del", line[1:]))
        else:
            rows.append(("ctx", line[1:] if line.startswith(" ") else line))
    hidden = max(0, len(rows) - _MAX_DIFF_LINES)
    return rows[:_MAX_DIFF_LINES], not existed, hidden


_DIFF_STYLE = {"add": "green", "del": "red", "hunk": _ACCENT, "ctx": _DIM}
_DIFF_SIGN = {"add": "+", "del": "-", "hunk": "", "ctx": " "}


def _render_write_diff(args: dict) -> None:
    """Render the colored unified diff for a pending write_file inside the approval frame, so the
    user sees exactly what changes before approving an overwrite (write_file overwrites by default —
    see gotcha #2). Falls back to a plain +/- listing without rich."""
    file_path = str(args.get("file_path", ""))
    content = args.get("content", "")
    overwrite = args.get("overwrite", True)
    rows, is_new, hidden = _diff_lines(file_path, str(content), bool(overwrite))

    mode = "new file" if is_new else ("overwrite" if overwrite else "append")
    if _RICH:
        head = Text()
        head.append("  ┃ ", style="bold")
        head.append(f"    ↳ diff ({mode}) ", style=_DIM)
        head.append(file_path, style="default")
        _console.print(head)
        if not rows:
            empty = Text()
            empty.append("  ┃ ", style="bold")
            empty.append("        (no textual change)", style=_DIM)
            _console.print(empty)
        width = max(20, _term_width() - 12)  # loop-invariant — compute once
        for kind, text in rows:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append(f"      {_DIFF_SIGN[kind]} ", style=_DIFF_STYLE[kind])
            row.append(_truncate(text, width), style=_DIFF_STYLE[kind])
            _console.print(row)
        if hidden:
            more = Text()
            more.append("  ┃ ", style="bold")
            more.append(f"        … {hidden} more diff line(s)", style=_DIM)
            _console.print(more)
    else:
        print(f"  ┃     -> diff ({mode}) {file_path}")
        for kind, text in rows:
            print(f"  ┃       {_DIFF_SIGN[kind]} {text}")
        if hidden:
            print(f"  ┃        … {hidden} more diff line(s)")


def ask_approval(value: dict) -> bool:
    """Compact, high-signal gate. Heavy rule + risk-colored tier so it breaks out of the dim
    trace rail. A write_file call additionally renders a colored unified diff of what it will
    change (gotcha #2: write_file overwrites by default). Returns True to approve the whole batch."""
    global _t_last
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    _live_stop()  # the gate blocks on input(); the bar can't be live while it does

    def arg_repr(v) -> str:
        r = repr(v)
        return r if len(r) <= 80 else r[:79] + "…"

    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("approval required", style=f"bold {_ACCENT}")
        top.append(" " + "━" * 36, style="bold")
        _console.print(top)
        for tc in tool_calls:
            risk = str(tc.get("risk", "destructive"))
            risk_style = _RISK.get(risk, "bold red")
            head = Text()
            head.append("  ┃ ", style="bold")
            head.append(f"{risk:<14} ", style=risk_style)  # tier chip, risk-colored
            head.append(f"{tc.get('name')}", style="default")
            _console.print(head)
            is_write = tc.get("name") == "write_file"
            for k, v in (tc.get("args") or {}).items():  # one line per argument — full clarity
                # For write_file the `content` arg is shown as a diff below, not as a truncated
                # repr — the diff is the real safety surface, the 80-char repr would just be noise.
                if is_write and k == "content":
                    continue
                arow = Text()
                arow.append("  ┃ ", style="bold")
                arow.append(f"    {k} = ", style=_DIM)
                arow.append(arg_repr(v), style="default")
                _console.print(arow)
            if is_write:
                _render_write_diff(tc.get("args") or {})
            hint = _RISK_HINT.get(risk)
            if hint:
                hrow = Text()
                hrow.append("  ┃ ", style="bold")
                hrow.append(f"    ↳ {hint}", style=risk_style)
                _console.print(hrow)
        resp = _console.input("  [bold]┗━[/] approve? [bold]y/N[/] » ").strip().lower()
    else:
        print("  ┏━ approval required " + "━" * 30)
        for tc in tool_calls:
            print(f"  ┃ [{tc.get('risk')}] {tc.get('name')}")
            is_write = tc.get("name") == "write_file"
            for k, v in (tc.get("args") or {}).items():
                if is_write and k == "content":
                    continue
                print(f"  ┃     {k} = {arg_repr(v)}")
            if is_write:
                _render_write_diff(tc.get("args") or {})
            hint = _RISK_HINT.get(str(tc.get("risk")))
            if hint:
                print(f"  ┃     -> {hint}")
        resp = input("  ┗━ approve? y/N » ").strip().lower()

    _t_last = time.perf_counter()  # don't bill the human's decision time to the next node
    _live_start()  # the turn continues (tools -> agent -> …); re-pin the bar
    return resp in ("y", "yes")


# ── plan-review editor (the human-in-the-loop pause) ─────────────────────────────
# Reached when a turn pauses at the plan_gate (keyboard pause, /plan pause|review, or an in-graph
# request). Renders the live plan with the current step marked, then runs a small edit loop on the
# shared plan_ops grammar until the user continues (resume with the edited plan) or aborts (end the
# turn). Mirrors ask_approval's Live teardown/restart so it composes with the bottom status bar.
def _review_emit(text) -> None:
    _emit(text)


def _review_header(reason: str) -> None:
    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("plan review", style=f"bold {_ACCENT}")
        top.append(" — execution paused", style=_DIM)
        top.append(" " + "━" * 22, style="bold")
        _console.print(top)
        if reason:
            r = Text()
            r.append("  ┃ ", style="bold")
            r.append(reason, style=_DIM)
            _console.print(r)
    else:
        print("  ┏━ plan review — execution paused " + "━" * 16)
        if reason:
            print(f"  ┃ {reason}")


def _render_review_plan(plan: list[dict], active_id) -> None:
    """List the plan inside the review block: every step with status + intended tool, the current
    step flagged so the user sees where execution will resume."""
    if not plan:
        _review_emit("  ┃   (empty plan — add steps with `add <label>`)")
        return
    for step in plan:
        line = _plan_line(step, show_tool=True)
        marker = "  ← current" if step.get("step_id") == active_id else ""
        if _RICH:
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append_text(line if isinstance(line, Text) else Text(str(line)))
            if marker:
                row.append(marker, style=f"bold {_ACCENT}")
            _console.print(row)
        else:
            print(f"  ┃ {line}{marker}")


def _review_help() -> None:
    import plan_ops

    _review_emit("  ┃ edit the plan, then `go` to run it (or `abort` to stop):")
    for h in plan_ops.COMMAND_HELP:
        _review_emit(f"  ┃     {h}")
    _review_emit("  ┃     go / <enter>          run the (edited) plan")
    _review_emit("  ┃     abort                 stop this turn")
    _review_emit("  ┃     show · help           reprint the plan · this help")


def _review_note(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ┃   ", style="bold")
        t.append(msg, style=_DIM if not msg.startswith("!") else "yellow")
        _console.print(t)
    else:
        print(f"  ┃   {msg}")


def _review_input() -> str:
    if _RICH:
        return _console.input("  [bold]┗━[/] edit» ", markup=True)
    return input("  ┗━ edit» ")


def review_plan(value: dict) -> dict:
    """Handle a plan-review interrupt. Returns `{"action": "continue"|"abort", "plan": <edited>}`
    — the resume value the plan_gate node applies. The edited plan is normalized + renumbered by
    plan_ops, so step ids the user typed always match what's rendered."""
    import plan_ops

    global _t_last
    plan = plan_ops.normalize(value.get("plan") or [])
    reason = value.get("reason", "")
    active = value.get("active_step") or {}
    active_id = active.get("step_id")

    _live_stop()  # the editor blocks on input(); the bar can't be live while it does

    _review_header(reason)
    _render_review_plan(plan, active_id)
    _review_help()

    action = "continue"
    while True:
        try:
            raw = _review_input()
        except (EOFError, KeyboardInterrupt):
            raw = ""  # treat as "continue" — never strand the turn on an empty read
        cmd = raw.strip()
        low = cmd.lower()
        if low in ("", "go", "c", "continue", "run", "resume"):
            action = "continue"
            break
        if low in ("abort", "q", "quit", "cancel", "stop"):
            action = "abort"
            break
        if low in ("help", "h", "?"):
            _review_help()
            continue
        if low in ("show", "ls", "plan"):
            _render_review_plan(plan, active_id)
            continue
        try:
            plan, note = plan_ops.apply_command(plan, cmd)
            _review_note(note)
            _render_review_plan(plan, active_id)
        except ValueError as exc:
            _review_note(f"! {exc}")

    if _RICH:
        tail = Text()
        verb = "running the plan" if action == "continue" else "aborting the turn"
        tail.append("  ┗━ ", style="bold")
        tail.append(verb, style=_ACCENT)
        _console.print(tail)
    else:
        print(f"  ┗━ {'running the plan' if action == 'continue' else 'aborting the turn'}")

    _t_last = time.perf_counter()  # don't bill the human's edit time to the next node
    _live_start()  # the turn continues; re-pin the bar
    return {"action": action, "plan": plan}


# ── final answer ─────────────────────────────────────────────────────────────────
def _turn_summary_parts() -> list[str]:
    """The post-response receipt: a permanent one-line echo of the (transient) status bar's run
    stats — the bar vanishes when the turn ends, so this is what survives in the scrollback."""
    elapsed = time.perf_counter() - _turn_start if _turn_start else 0.0
    n = _status["tools"]
    parts = [f"{_status['iteration']} iter", f"{n} tool{'' if n == 1 else 's'}",
             _fmt_dur(elapsed).strip()]
    if _status["tok_per_sec"] > 0:
        parts.append(f"{_status['tok_per_sec']:.0f} tok/s")
    window = _status["ctx_window"]
    if window and _status["ctx_used"]:
        parts.append(f"ctx {_status['ctx_used'] / window * 100:.0f}%")
    return parts


def response(text: str) -> None:
    """The payload. Leaves the trace rail behind a short labeled rule and renders the answer as
    real markdown — headings, bold, lists, and fenced code with syntax highlighting — so it reads
    like a finished answer, not a log line. Falls back to plain text if markdown rendering raises
    (arbitrary model output), and to plain print without rich."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    if _RICH:
        _console.print()  # part the answer from the trace rail above it
        rule = Text()
        rule.append("  ── ", style=_DIM)
        rule.append("response", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
        _console.print()  # let the answer breathe beneath its rule
        try:
            # Markdown parses markdown, not Rich console markup, so bracketed tokens like
            # `list[str]` or citations `[1]` are safe literal text here.
            _console.print(Markdown(text))
        except Exception:
            # Arbitrary model output can occasionally trip the markdown parser; never lose the
            # answer over formatting. markup=False so brackets aren't eaten as Rich tags.
            _console.print(text, markup=False)
        _console.print()  # let the answer breathe before the receipt
        _console.print(Text("  ╶ " + " · ".join(_turn_summary_parts()), style=_DIM))
        _console.print()  # trailing whitespace before the next prompt
    else:
        print()
        print("  ── response " + "─" * 36)
        print()
        print(text)
        print()
        print("  ╶ " + " · ".join(_turn_summary_parts()))
        print()


# ── streaming the final answer ─────────────────────────────────────────────────────
# The synthesize node streams its answer token-by-token (LangGraph messages mode -> run_turn ->
# on_token). ResponseStream renders those tokens live, then finishes with the same finished look as
# `response`. The hard part in a terminal is long output: a growing Live region that outgrows the
# screen can't be erased cleanly. So during streaming we show a *transient* Live of only the last
# screenful (a bounded tail — see `_tail`), which always fits and so always erases cleanly; on
# `finish` we tear that down and render the WHOLE answer once as real markdown (+ the receipt). The
# permanent scrollback record is that final rendered block, not the transient tail. Without rich we
# just type the raw tokens out incrementally. If the model yields no tokens, `started` stays False
# and the caller renders via `response` instead.
class ResponseStream:
    def __init__(self) -> None:
        self._chars: list[str] = []
        self._live = None
        self._started = False
        self._last = 0.0  # last repaint time (throttle)

    @property
    def started(self) -> bool:
        return self._started

    def feed(self, text: str) -> None:
        """Append a streamed answer token; opens the response section on the first one."""
        if not text:
            return
        if not self._started:
            self._begin()
        self._chars.append(text)
        if self._live is not None:
            now = time.perf_counter()
            if now - self._last >= 0.06:  # throttle (~16/s) so granular tokens don't thrash the live
                self._live.update(self._tail(), refresh=True)
                self._last = now
        else:  # plain (no-rich) path: just type it out
            print(text, end="", flush=True)

    def _begin(self) -> None:
        self._started = True
        _live_stop()  # drop the turn's status bar — the answer takes over the bottom of the screen
        if _RICH:
            _console.print()  # part the answer from the trace rail above it
            rule = Text()
            rule.append("  ── ", style=_DIM)
            rule.append("response", style=f"bold {_ACCENT}")
            rule.append(" " + "─" * 40, style=_DIM)
            _console.print(rule)
            _console.print()
            # transient + a screen-bounded tail => the live region always fits, so stop() erases it
            # cleanly no matter how long the answer runs. Manual refresh (throttled in feed).
            self._live = Live(console=_console, transient=True, auto_refresh=False)
            self._live.start()
        else:
            print()
            print("  ── response " + "─" * 36)
            print()

    def _tail(self) -> "Text":
        """The last screenful of the answer-so-far as plain Text, bounded to at most `rows` VISUAL
        lines so the transient live region always fits on screen — which is what lets `stop()` erase
        it cleanly before the full answer is re-rendered. The bound must count visual rows, which
        means accounting for BOTH hard newlines AND soft wrapping: a raw character budget undercounts
        badly when the answer is many short lines (lists, headings, code, blanks), because each
        newline ends a line early, so the same budget spans far more rows than the screen has. The
        region then scrolls off the top and the transient erase corrupts the final render — eating
        the first lines of the answer (the data is fine; only the on-screen handoff breaks)."""
        rows = max(4, (_console.size.height or 24) - 6)
        cols = max(20, _console.size.width or 80)
        avail = max(1, cols - 2)  # room for text after the 2-space indent below
        lines = "".join(self._chars).split("\n")
        # Walk from the bottom up, accumulating physical lines until their WRAPPED height fills the
        # row budget, so the rendered region can't exceed the screen no matter the line lengths.
        chosen: list[str] = []
        used = 0
        for ln in reversed(lines):
            h = max(1, -(-len(ln) // avail))  # ceil(len / avail); a blank line is still one row
            if used + h > rows:
                if chosen:
                    break
                ln = ln[-(rows * avail):]  # a lone line taller than the screen: keep its tail only
            chosen.append(ln)
            used += h
            if used >= rows:
                break
        chosen.reverse()
        t = Text()
        for i, ln in enumerate(chosen):
            if i:
                t.append("\n")
            t.append("  ")
            t.append(ln)
        return t

    def finish(self) -> None:
        """Close out a successful turn: tear down the live tail, render the full answer once as
        markdown, then the one-line receipt. Mirrors `response`'s final look exactly."""
        text = "".join(self._chars)
        if self._live is not None:
            self._live.stop()  # transient: erases the streaming tail
            self._live = None
        if _RICH:
            try:
                _console.print(Markdown(text))
            except Exception:
                _console.print(text, markup=False)
            _console.print()
            _console.print(Text("  ╶ " + " · ".join(_turn_summary_parts()), style=_DIM))
            _console.print()
        else:
            print()  # close the typed-out line
            print("  ╶ " + " · ".join(_turn_summary_parts()))
            print()

    def abort(self) -> None:
        """Tear down the live tail without a final render — a failed/cancelled turn. The transient
        Live erases the partial text; the caller surfaces the error (`warn`) separately."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None


# ── system metrics display ───────────────────────────────────────────────────────
def show_system_metrics(metrics) -> None:
    """Display a compact system-resource readout in the trace-rail style. Shares the one meter
    glyph + threshold vocabulary (`_mini_bar` / `_meter_color`) with the status bar and /context,
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

    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append("system", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
    else:
        print("  ╶── system " + "─" * 44)

    _row("cpu", metrics.cpu_usage_percent)
    ram_pct = metrics.ram_used_gb / metrics.total_ram_gb * 100
    _row("ram", ram_pct, f"{metrics.ram_used_gb:.1f} / {metrics.total_ram_gb:.1f} GB")
    if metrics.gpu_usage_percent is not None:
        _row("gpu", metrics.gpu_usage_percent)
    if metrics.vram_used_gb is not None and metrics.total_vram_gb is not None:
        vram_pct = metrics.vram_used_gb / metrics.total_vram_gb * 100
        _row("vram", vram_pct, f"{metrics.vram_used_gb:.1f} / {metrics.total_vram_gb:.1f} GB")


# ── context-window readout (the /context command) ──────────────────────────────────
def show_context(window: int, used: int, source: str, per_role: dict[str, int]) -> None:
    """Detailed context-window readout for /context: the active window + where it comes from, a
    wide fill bar for the last measured usage, and the per-role windows. Same trace-rail
    vocabulary as show_system_metrics; the compact form of this fill gauge also rides the live
    status bar during a turn."""
    pct = (used / window * 100) if window else 0.0
    col = _meter_color(pct)
    bar = _mini_bar(pct, width=28)

    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append("context", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)

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
        print("  ╶── context " + "─" * 42)
        print(f"  window {window:,} tokens   ({source})")
        print(f"  {_RAIL_GLYPH} usage  {bar}  {pct:>4.0f}%   {used:,} / {window:,}")

    if per_role:
        roles_txt = "  ·  ".join(f"{r} {w:,}" for r, w in per_role.items())
        _emit(f"  roles: {roles_txt}")
    _emit("  set with /context <size> (or /context auto for per-model capability)")


# ── run drill-down (the /trace expanded view) ─────────────────────────────────────
def _enrich_results(events: list[dict], results: list, cap: int = 1200) -> list[dict]:
    """Pair each recorded tool event with the fuller `call -> observation` from tool_results
    (collapsed to one line, capped), so the /trace replay shows real output where the live tree
    deliberately showed nothing. Falls back to the event's own preview when no pair exists."""
    out = []
    for i, ev in enumerate(events):
        ev = dict(ev)
        if i < len(results):
            _, _, obs = str(results[i]).partition(" -> ")
            obs = " ".join(obs.split())
            if obs:
                ev["result"] = (obs[: cap - 1] + "…") if len(obs) > cap else obs
        out.append(ev)
    return out


def show_run(run, events) -> None:
    """Replay one recorded run from the trace DB as an expanded drill-down (the default /trace view):
    the query, every node with its wall-clock step time + metrics, the plan as it advanced, the
    agent's reasoning + tool-call decisions per step (the `ai`/`in` leaves — the execution-log detail
    the live trace omits), each tool call WITH its output (the live trace hides these too), and last,
    de-emphasized, the recorded final answer. The full-fidelity counterpart to the live trace; same
    rail/glyph/tree vocabulary, but here the EXECUTION LOG is the subject, not the response.

    `run` is the row `(run_id, query, started_at, ended_at, status, response)`; `events` are its
    `(seq, ts, node, summary, data)` rows in order. Step times are wall-clock deltas between event
    timestamps, so a tool step that waited on the approval gate honestly includes that pause."""
    import json
    from datetime import datetime

    global _plan_seen
    run_id, query, started_at, ended_at, status, response_text = run

    def _parse(ts):
        try:
            return datetime.fromisoformat(ts) if ts else None
        except (TypeError, ValueError):
            return None

    # header: run id · query · when / status / total wall time
    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append(f"run #{run_id}", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
    else:
        print(f"  ╶── run #{run_id} " + "─" * 40)

    q = " ".join(str(query or "").split()) or "(empty)"
    start_dt, end_dt = _parse(started_at), _parse(ended_at)
    when = (started_at or "")[:19].replace("T", " ")
    total = _fmt_dur((end_dt - start_dt).total_seconds()).strip() if (start_dt and end_dt) else ""
    status_style = {"ok": "green", "error": "bold red", "running": "yellow"}.get(str(status), _DIM)
    if _RICH:
        qline = Text("  ")
        qline.append("query  ", style=_DIM)
        qline.append(q, style="default")
        _console.print(qline)
        meta = Text("  ")
        meta.append(when or "—", style=_DIM)
        meta.append("  ·  ", style=_DIM)
        meta.append(str(status), style=status_style)
        if total:
            meta.append("  ·  ", style=_DIM)
            meta.append(total, style=_DIM)
        _console.print(meta)
    else:
        print(f"  query  {q}")
        print(f"  {when}  ·  {status}" + (f"  ·  {total}" if total else ""))
    _emit("")

    # node-by-node replay. plan_gate is a control checkpoint with no info (folded in the live trace
    # too); everything else shows — this IS the full drill-down, plumbing and tool outputs included.
    saved_seen = _plan_seen
    _plan_seen = {}  # let show_plan diff afresh over this run's plan events
    prev = start_dt
    try:
        for _seq, ts, node, _summary, data in events:
            if node == "plan_gate":
                continue
            try:
                delta = json.loads(data or "{}")
            except (json.JSONDecodeError, TypeError):
                delta = {}
            cur = _parse(ts)
            dur = (cur - prev).total_seconds() if (cur and prev) else 0.0
            if cur:
                prev = cur
            _emit(_node_line(node, dur, delta))
            if delta.get("plan"):
                show_plan(delta["plan"])
            # the agent's reasoning / tool-call decisions for this step — the execution-log detail
            # the live trace omits; this is the point of the drill-down
            _render_trace_messages(node, delta)
            tev = delta.get("tool_events") or []
            if tev:
                _render_tool_events(_enrich_results(tev, delta.get("tool_results") or []),
                                    always_show_results=True)
    finally:
        _plan_seen = saved_seen

    # the run's final answer — subordinate in the replay. The execution log above is the subject of
    # /trace; the answer is just the recorded outcome, so it's rendered quietly (dim plaintext under
    # a faint label) rather than as the bold-accent markdown the LIVE turn already showed.
    if response_text:
        _emit("")
        if _RICH:
            rule = Text()
            rule.append("  ╶ ", style=_FAINT)
            rule.append("final answer", style=_DIM)
            rule.append(" (recorded)", style=_FAINT)
            _console.print(rule)
            for ln in response_text.splitlines() or [""]:
                row = Text("  ")
                row.append(ln, style=_DIM)
                _console.print(row)
        else:
            print("  ╶ final answer (recorded)")
            for ln in response_text.splitlines() or [""]:
                print(f"  {ln}")


# ── model picker / listing ───────────────────────────────────────────────────────
def show_models(models, bindings: dict, active_tier: str, embedder: str,
                *, numbered: bool = False) -> None:
    """Render the locally-installed (Ollama) models plus the live role bindings, in the
    trace-rail style. `models` is a list of `llms.LocalModel`; `bindings` maps role -> model id;
    `embedder` is the active embedder tag. With `numbered=True` each installed row gets a 1-based
    index (the selector the interactive picker reads). A `◂ <roles>` tail marks what each model
    currently drives, so the bindings are visible inline."""
    # role(s) / embedder each installed tag currently serves -> shown as a tail marker.
    serves: dict[str, list[str]] = {}
    for role, mid in (bindings or {}).items():
        serves.setdefault(mid, []).append(role)
    if embedder:
        serves.setdefault(embedder, []).append("embedder")

    all_roles = set(bindings or {})

    def _tail_for(name: str) -> str:
        """Compact 'what this tag drives' marker. Collapses every-role bindings to 'all roles'
        so a model serving the whole loop doesn't spill five role names across the line."""
        entries = serves.get(name, [])
        roles = [e for e in entries if e != "embedder"]
        parts = []
        if roles:
            parts.append("all roles" if all_roles and set(roles) == all_roles
                         else " ".join(roles))
        if "embedder" in entries:
            parts.append("embedder")
        return "  ".join(parts)

    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append("models", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
        sub = Text("  ")
        sub.append("tier ", style=_DIM)
        sub.append(active_tier, style="default")
        sub.append("  ·  embedder ", style=_DIM)
        sub.append(embedder or "—", style="default")
        _console.print(sub)
    else:
        print("  ╶── models " + "─" * 44)
        print(f"  tier {active_tier}  ·  embedder {embedder or '—'}")

    if not models:
        _emit("  (no local models — is the Ollama daemon running? `ollama list`)")
    else:
        for i, m in enumerate(models, start=1):
            meta = " ".join(p for p in (m.parameter_size, m.quantization) if p) or "·"
            tail = _tail_for(m.name)
            idx = f"{i:>2}  " if numbered else ""
            if _RICH:
                line = _rail()
                if numbered:
                    line.append(f"{i:>2}  ", style=_ACCENT)
                line.append(f"{m.name:<26}", style="default")
                line.append(f"{m.size_h:>7}  ", style=_DIM)
                line.append(f"{meta:<14}", style=_DIM)
                if m.is_embedding:
                    line.append("[embed] ", style="yellow")
                if tail:
                    line.append("◂ " + tail, style="green")
                _console.print(line)
            else:
                emb = "[embed] " if m.is_embedding else ""
                bound = ("◂ " + tail) if tail else ""
                print(f"  {_RAIL_GLYPH} {idx}{m.name:<26}{m.size_h:>7}  {meta:<14}{emb}{bound}")

    # Role bindings summary — the full role list, even for roles whose model isn't pulled locally
    # (e.g. a cloud-hybrid anthropic binding won't appear in the installed list above).
    if bindings:
        _emit("  bindings:")
        for role, mid in bindings.items():
            _emit(f"    {role:<12} {mid}")


def ask(prompt_text: str) -> str:
    """Read a single line for an interactive command prompt (e.g. the /models picker). Tears down
    any live status bar first — input() can't run under an active Live — and returns the raw,
    stripped reply. Degrades to plain input() without rich."""
    _live_stop()
    try:
        if _RICH:
            # markup=False: prompts carry literal brackets (e.g. "[all|planner|…]") that Rich
            # would otherwise eat as style tags.
            return _console.input(f"  {prompt_text}", markup=False).strip()
        return input(f"  {prompt_text}").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# ── log lines (startup notices, warnings) ────────────────────────────────────────
def note(msg: str) -> None:
    """A quiet informational line (dim) — e.g. the `@file` attachment notice. Distinct from
    `warn` (yellow), which flags a problem; a note is just neutral context."""
    if _RICH:
        t = Text()
        t.append("  · ", style=_DIM)
        t.append(msg, style=_DIM)
        _console.print(t)
    else:
        print(f"  · {msg}")


def warn(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ! ", style="yellow")
        t.append(msg, style="yellow")
        _console.print(t)
    else:
        print(f"  ! {msg}")


def steer_note(text: str) -> None:
    """Acknowledge a mid-turn steering correction the moment it's captured (Esc with typed text).
    The correction is injected into the running turn at the next step boundary (see plan_gate); this
    is the immediate feedback that it landed, printed above the live status bar."""
    msg = text if len(text) <= 80 else text[:79] + "…"
    if _RICH:
        t = Text()
        t.append("  ↪ ", style=f"bold {_ACCENT}")
        t.append("steering — applies at the next step: ", style=_ACCENT)
        t.append(msg, style=_DIM)
        _console.print(t)
    else:
        print(f"  ↪ steering — applies at the next step: {msg}")


def echo_queued(line: str) -> None:
    """Echo a type-ahead line as the REPL pulls it off the queue to run, so a query/command the
    user typed while a previous turn was working shows up in the transcript just like a line typed
    live at the `»` prompt (with a quiet `queued` tag to mark where it came from)."""
    if _RICH:
        t = Text()
        t.append("» ", style=f"bold {_ACCENT}")
        t.append(line, style="default")
        t.append("   (queued)", style=_DIM)
        _console.print(t)
    else:
        print(f"» {line}   (queued)")
