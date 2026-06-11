"""
Shared foundation for the `tui.ui` package: the console handle + capability flags, the palette,
the per-process / per-turn mutable state, and the low-level rendering primitives every other
submodule builds on. Imported as `from . import _base` (for the mutable state, which must be read
through the module so a rebind in one submodule is seen by the rest) and via `from ._base import …`
for the constants + stateless helpers (never rebound, safe to bind by name).

This module depends on nothing else in the package — it's the leaf the others import.
"""

import os
import shutil

from textutil import fmt_args, truncate as _truncate

try:
    from rich.console import Console
    from rich.text import Text
    from rich.live import Live
    from rich.markdown import Markdown

    _console = Console(highlight=False)
    _RICH = True
except Exception:  # pragma: no cover - fallback path
    # Define the names unconditionally so `from ._base import Text` resolves even without rich;
    # they're only ever *used* under an `if _RICH:` guard, so the None values are never called.
    Console = Text = Live = Markdown = None
    _console = None
    _RICH = False


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


# ── trace verbosity ───────────────────────────────────────────────────────────
# How much of the execution trace scrolls live. The trace DB keeps everything regardless, so
# /trace and /calls stay full-fidelity no matter what this is set to:
#   "normal"  (default) — plumbing nodes (ground, update_plan) are folded out of the live rail;
#                         their *output* still prints (update_plan's plan diff is driven by
#                         show_plan), and their timing rolls into the next visible node.
#   "verbose"           — every node line, including the folded plumbing ones and full timings.
# Whether the trace renders at all is a separate switch (commands' show_ui / `/trace off`).
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
# These are mutated across submodules, so they MUST be accessed through the module
# (`_base._t_last`), never bound by name — a `from ._base import _t_last` would copy the binding and
# writes wouldn't propagate. reset_turn (statusbar) seeds them; show_node (trace) / show_plan (plan)
# / the gates (approval) advance them.
_t_last = None
_plan_seen: dict = {}
_trace_started = False  # False until the turn's first node line prints (gates one lead-in blank)

# ── live status-bar state (shared with response/_StatusBar) ───────────────────
# `_status` is the live readout the bar renders; `_turn_start` anchors the elapsed clock.
# `_model` is captured once in banner() so the bar needs no model passed per turn. (The Live
# handle + the metrics sampler stay private to statusbar.py — only these cross submodules.)
_turn_start = None
_status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0,
           "ctx_used": 0, "ctx_window": 0}
_model = "unknown"


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


# ── small rendering helpers ──────────────────────────────────────────────────
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
    return fmt_args(args, cap)


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
