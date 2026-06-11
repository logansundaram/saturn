"""
The startup splash — the one ornamental moment. A tilted cyan ring draws itself out around a dark,
banded gas-giant (Saturn, for the `saturn` launch word), plays once, and is fully skippable. Self-
contained: the geometry, the in-place animation player, and `splash` (the startup draw-out,
optionally holding for a background `work` callable) all live here.
"""

import io
import math
import os
import time

from ._base import Console, Text, _console, _RICH


# ── geometry / shading constants ─────────────────────────────────────────────
# Set SATURDAY_NO_SPLASH=1 to suppress it, SATURDAY_NO_ANIM=1 to print only the resting frame; it
# also auto-skips when stdout isn't a terminal or is too narrow.
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
