"""
CLI rendering for the agent console.

Design target: a serious local-agent console — git status / htop / pytest / a trace viewer,
not a chatbot. Dense, fast, keyboard-first, low-noise, inspectable. The aesthetic comes from
*structure*, not decoration:

  - A dim vertical rail (`│`) carries the execution trace. Consecutive node lines form one
    continuous gutter, so a turn reads as a single inspectable block (the htop/tree feel).
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk tier. Structure
    is dim. Nothing is colored just to look nice — if it has color, it means something.
  - The plan prints **once** as the intended route, then emits a single line per status change
    as steps advance — a log/trace, not a re-rendered panel. This is the transparency surface
    and the main noise source, so it's diffed.
  - The approval gate deliberately breaks out of the rail with a heavy rule. It's a blocking
    safety decision and *should* draw the eye; everything else recedes.
  - A single-line **status bar** is pinned at the bottom of the screen for the duration of a
    turn (`rich.live.Live`): `model · iter · elapsed · tools · ▸node`. The trace lines above it
    keep scrolling normally (rich routes `console.print` and captured `stdout` above the live
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
import time

try:
    from rich.console import Console
    from rich.text import Text
    from rich.live import Live

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
    from prompt_toolkit.completion import Completer as _PTKCompleter, Completion as _PTKCompletion
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

_RAIL_GLYPH = "│"
_NODE_W = 12  # node-name column width, keeps timings aligned
_LABEL_W = 46  # plan-label truncation (keeps step lines on one row at 80 cols)

# ── per-turn state (timing + plan diff). Reset via reset_turn() each turn. ─────
_t_last = None
_plan_seen: dict = {}

# ── live status bar (bottom-pinned) ───────────────────────────────────────────
# `_status` is the live readout the bar renders; `_turn_start` anchors the elapsed
# clock; `_live` holds the active rich.live.Live (None when torn down for input).
# `_model` is captured once in banner() so the bar needs no model passed per turn.
_turn_start = None
_status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0}
_model = "unknown"
_live = None


class _StatusBar:
    """Renderable for the pinned bar. `__rich__` is re-evaluated on every Live refresh, so the
    elapsed clock ticks even when no node update has fired."""

    def __rich__(self) -> "Text":
        elapsed = time.perf_counter() - _turn_start if _turn_start else 0.0
        n = _status["tools"]
        tps = _status["tok_per_sec"]
        bar = Text()
        bar.append("  ╶ ", style=_DIM)
        bar.append("saturday", style=f"bold {_ACCENT}")
        labels = [_model, f"iter {_status['iteration']}", _fmt_dur(elapsed).strip(),
                  f"{n} tool{'' if n == 1 else 's'}"]
        if tps > 0:
            labels.append(f"{tps:.0f} tok/s")
        for label in labels:
            bar.append("  ·  ", style=_DIM)
            bar.append(label, style="default")
        if _status["node"]:
            bar.append("  ·  ", style=_DIM)
            bar.append(f"▸ {_status['node']}", style=f"bold {_ACCENT}")
        return bar


def _live_start() -> None:
    """Pin a fresh status bar at the bottom. No-op without rich or if one is already running.
    `transient=True` erases the bar on stop (the scrolling trace stays); rich's default
    stdout/stderr redirect keeps node `print()`s flowing above the live region."""
    global _live
    if not _RICH or _live is not None:
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
    global _t_last, _plan_seen, _turn_start, _status
    _t_last = time.perf_counter()
    _turn_start = _t_last
    _plan_seen = {}
    _status = {"node": "", "iteration": 0, "tools": 0, "tok_per_sec": 0.0}
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
    if seconds < 1:
        return f"{seconds * 1000:>4.0f}ms"
    if seconds < 60:
        return f"{seconds:>5.2f}s"
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


# ── startup banner ─────────────────────────────────────────────────────────────
def banner(model: str, n_tools: int, n_docs: int, db_path: str) -> None:
    """Compact two-line header. Reads like a tool's startup line, not a splash screen."""
    global _model
    _model = model  # captured here so the live status bar needs no model passed per turn
    if _RICH:
        head = Text()
        head.append("saturday", style=f"bold {_ACCENT}")
        head.append(".ai", style=_DIM)
        for label in (model, f"{n_tools} tools", f"{n_docs} docs"):
            head.append("  ·  ", style=_DIM)
            head.append(label, style="default")
        _console.print(head)
        sub = Text()
        sub.append(f"sqlite:{db_path}", style=_DIM)
        sub.append("   ", style=_DIM)
        sub.append("/help", style=_ACCENT)
        sub.append(" for commands", style=_DIM)
        _console.print(sub)
    else:
        print(f"saturday.ai  ·  {model}  ·  {n_tools} tools  ·  {n_docs} docs")
        print(f"sqlite:{db_path}   /help for commands")


# ── input prompt ───────────────────────────────────────────────────────────────
# Live highlight for the `»` line: a `/token` is colored by how it matches the command set, so
# a typo never blends in with a real command. Valid -> cyan, a prefix of some command (mid-type)
# -> yellow, anything else -> red. Args after the token stay dim. Built only when prompt_toolkit
# is present; the palette mirrors the rest of ui.py (cyan accent, semantic status colors).
if _PTK:
    _PTK_STYLE = _PTKStyle.from_dict({
        "prompt": "ansicyan bold",
        "cmd.valid": "ansicyan bold",
        "cmd.partial": "ansiyellow",
        "cmd.unknown": "ansired bold",
        "cmd.args": "ansibrightblack",
    })

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
            text = document.text

            def get_line(_lineno):
                parsed = _slash_token(text)
                if parsed is None:
                    return [("", text)]
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
        """Tab-completes the leading `/command` token against the known command set. Fires only
        on the first token of a slash line (a space ends the token — args are left alone), so it
        never interferes with normal turns or command arguments. `display_meta` carries each
        command's one-line summary into the completion menu."""

        def __init__(self, meta):
            self._meta = meta  # list of (token, summary), tokens lowercased, no leading slash

        def get_completions(self, document, complete_event):
            parsed = _slash_token(document.text_before_cursor)
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

    _ptk_session = None  # one PromptSession for the process -> free line history across turns


def prompt(command_meta=None) -> str:
    """Read the `»` input line. With prompt_toolkit and a `command_meta` list of `(token, summary)`
    pairs, a typed `/command` is highlighted live (valid=cyan, typo=red) and Tab completes the
    leading `/command` token — the highlight set is derived from the same tokens. Without it, falls
    back to rich/plain input. Returns the raw line (slash-command detection happens upstream)."""
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
        )
    if _RICH:
        return _console.input(f"[bold {_ACCENT}]»[/] ")
    return input("» ")


# ── execution trace ─────────────────────────────────────────────────────────────
def show_node(node: str, delta: dict | None = None) -> None:
    """One trace line per node execution: `│ <node>   <elapsed>`, with the elapsed time
    measured since the previous node emitted (htop-style). The `tools` node also surfaces the
    tool names it ran, so the trace shows *what* happened, not just that something did."""
    global _t_last
    now = time.perf_counter()
    dur = now - _t_last if _t_last is not None else 0.0
    _t_last = now

    extra = ""
    if delta:
        called = delta.get("tools_called") or []
        if called:
            extra = ", ".join(called)
        # Feed the pinned status bar: latest node, running tool count, agent iteration.
        _status["tools"] += len(called)
        if "iteration" in delta:
            _status["iteration"] = delta["iteration"]
        tps = delta.get("tok_per_sec") or 0.0
        if tps > 0:
            _status["tok_per_sec"] = tps
            extra = (extra + "  " if extra else "") + f"{tps:.0f} tok/s"
    _status["node"] = node

    if _RICH:
        line = _rail()
        line.append(f"{node:<{_NODE_W}}", style="default")
        line.append(f"{_fmt_dur(dur):>7}", style=_DIM)
        if extra:
            line.append(f"  {extra}", style=_ACCENT)
        _console.print(line)
    else:
        tail = f"  {extra}" if extra else ""
        print(f"  {_RAIL_GLYPH} {node:<{_NODE_W}}{_fmt_dur(dur):>7}{tail}")

    _live_refresh()  # repaint the bar with the new node/iter/tools immediately


def _plan_line(step: dict, *, show_tool: bool) -> "Text | str":
    status = step.get("status", "pending")
    glyph, style = _PLAN.get(status, _PLAN["pending"])
    label = _truncate(str(step.get("label", "")), _LABEL_W)
    sid = step.get("step_id", "?")
    tool = step.get("intended_tool")

    if _RICH:
        line = _rail()
        line.append("  ", style=_RAIL)  # nest steps under the node
        line.append(f"{glyph} ", style=style)
        line.append(f"{str(sid):>2}  ", style=_DIM)
        line.append(label, style=style if status in ("active", "skipped") else "default")
        if show_tool and tool:
            line.append(f"  ::{tool}", style=_DIM)
        return line
    tooltxt = f"  ::{tool}" if (show_tool and tool) else ""
    return f"  {_RAIL_GLYPH}   {glyph} {str(sid):>2}  {label}{tooltxt}"


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
def ask_approval(value: dict) -> bool:
    """Compact, high-signal gate. Heavy rule + risk-colored tier so it breaks out of the dim
    trace rail. Returns True to approve the whole batch."""
    global _t_last
    tool_calls = value.get("tool_calls", []) if isinstance(value, dict) else []

    _live_stop()  # the gate blocks on input(); the bar can't be live while it does

    if _RICH:
        top = Text()
        top.append("  ┏━ ", style="bold")
        top.append("approval required", style=f"bold {_ACCENT}")
        top.append(" " + "━" * 36, style="bold")
        _console.print(top)
        for tc in tool_calls:
            risk = str(tc.get("risk", "destructive"))
            row = Text()
            row.append("  ┃ ", style="bold")
            row.append(f"{risk:<14} ", style=_RISK.get(risk, "bold red"))
            row.append(f"{tc.get('name')}", style="default")
            row.append(f"({_fmt_args(tc.get('args', {}))})", style=_DIM)
            _console.print(row)
        resp = _console.input("  [bold]┗━[/] approve? [bold]y/N[/] » ").strip().lower()
    else:
        print("  ┏━ approval required " + "━" * 30)
        for tc in tool_calls:
            print(f"  ┃ [{tc.get('risk')}] {tc.get('name')}({_fmt_args(tc.get('args', {}))})")
        resp = input("  ┗━ approve? y/N » ").strip().lower()

    _t_last = time.perf_counter()  # don't bill the human's decision time to the next node
    _live_start()  # the turn continues (tools -> agent -> …); re-pin the bar
    return resp in ("y", "yes")


# ── final answer ─────────────────────────────────────────────────────────────────
def response(text: str) -> None:
    """The payload. Leaves the trace rail (un-indented, copy-pasteable) behind a short labeled
    rule, so the answer is visually distinct from the trace above it."""
    _live_stop()  # turn's over: drop the status bar before printing the answer
    if _RICH:
        rule = Text()
        rule.append("  ╶── ", style=_DIM)
        rule.append("response", style=f"bold {_ACCENT}")
        rule.append(" " + "─" * 40, style=_DIM)
        _console.print(rule)
        # markup=False: the answer is arbitrary model output; bracketed tokens like `list[str]`,
        # citations `[1]`, or paths `[/etc/hosts]` must not be parsed as Rich tags (they get
        # stripped, or raise MarkupError and kill the turn). highlight=False already on _console.
        _console.print(text, markup=False)
    else:
        print("  ╶── response " + "─" * 36)
        print(text)


# ── system metrics display ───────────────────────────────────────────────────────
def show_system_metrics(metrics) -> None:
    """Display a compact system-resource readout in the trace-rail style."""

    def _pct_color(pct: float) -> str:
        if pct < 50:
            return "green"
        if pct < 80:
            return "yellow"
        return "bold red"

    def _bar(pct: float, width: int = 20) -> str:
        filled = round(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def _row(label: str, pct: float, detail: str = "") -> None:
        bar = _bar(pct)
        col = _pct_color(pct)
        if _RICH:
            line = _rail()
            line.append(f"{label:<6}", style=_DIM)
            line.append(f"  {bar}", style=col)
            line.append(f"  {pct:>5.1f}%", style=col)
            if detail:
                line.append(f"   {detail}", style=_DIM)
            _console.print(line)
        else:
            print(f"  {_RAIL_GLYPH} {label:<6}  {bar}  {pct:>5.1f}%{'   ' + detail if detail else ''}")

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
def warn(msg: str) -> None:
    if _RICH:
        t = Text()
        t.append("  ! ", style="yellow")
        t.append(msg, style="yellow")
        _console.print(t)
    else:
        print(f"  ! {msg}")
