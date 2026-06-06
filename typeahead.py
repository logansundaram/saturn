"""
Type-ahead input queue — the "keep typing while the agent works" surface (à la Claude Code).

While a turn is executing the REPL is blocked driving the graph, so historically every keystroke
was thrown away (the old `interrupts.KeyWatcher` consumed them purely to keep them out of the next
prompt) and the only way to line up a follow-up was to wait for the answer. This module lets the
user type ahead: a daemon thread reads the console *during* execution, echoes the in-progress line
into the status bar, and on Enter pushes the completed line onto a thread-safe FIFO the REPL drains
the moment the turn finishes — so follow-up queries AND slash commands can be queued without
waiting.

It subsumes the pause trigger, too: a single console can't be read by two threads, so `InputQueue`
is the *one* reader live during a turn. The keys, by what they do to the line you're typing:

  - **Enter** — commit the line to the queue (runs after the current turn).
  - **Esc with text typed** — submit that text as a *mid-turn steering correction*: it's injected
    into the running turn at the next step boundary (see `node_registry/plan_gate.py`) so the agent
    adjusts course WITHOUT losing the turn. The line is consumed (cleared), not queued.
  - **Esc on an empty line** — ask the shared `interrupts.PauseController` for a plan-review pause
    at the next step boundary (the role the `p` key used to play, moved off a letter so letters are
    free to type).

Steering vs. queuing is thus the same key story as Enter vs. Esc: Enter defers, Esc acts now.

Degrades to a no-op when the console can't be polled (not a TTY, or neither msvcrt nor POSIX
termios is available): the queue simply stays empty and the REPL blocks on the prompt exactly as
before. The repo's platform is win32, so the msvcrt path is primary; the POSIX path is best-effort
(cbreak via termios, restored on stop).
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from typing import Callable, Optional

from interrupts import PauseController, get_pause_controller

# Windows console key polling — the primary path on this repo's platform (win32). Absent
# elsewhere, where we fall back to a best-effort POSIX termios reader, then to a no-op.
try:
    import msvcrt  # type: ignore

    _HAS_MSVCRT = True
except Exception:  # pragma: no cover - non-Windows
    _HAS_MSVCRT = False


# Control characters the reader special-cases.
_ENTER = ("\r", "\n")
_BACKSPACE = ("\x08", "\x7f")
_ESC = "\x1b"
# Windows getwch() returns one of these as the first half of a two-char sequence for arrow / F-keys;
# we read and discard the trailing scancode so a stray key never lands in the buffer.
_WIN_PREFIX = ("\x00", "\xe0")


class InputQueue:
    """The single console reader for the duration of a turn: captures type-ahead lines (Enter →
    queue) and handles the Esc key (with text → mid-turn steer; empty → plan-review pause).

    `run_turn` brackets graph execution with `start()` / `stop()`, so the reader is live only while
    the graph runs — never while a blocking `input()` (the `»` prompt, the approval gate, the
    plan-review editor) is reading a line, so it can't steal those keystrokes. The REPL calls
    `pop()` between turns to drain queued lines before it blocks on the prompt.

    `on_change(buffer, queued)` (optional) is invoked on every mutation so a UI can render the live
    in-progress line and queue depth (here: the pinned status bar). It must not raise — a display
    hiccup can't be allowed to kill the reader thread."""

    def __init__(
        self,
        on_change: Optional[Callable[[str, int], None]] = None,
        on_steer: Optional[Callable[[str], None]] = None,
        controller: Optional[PauseController] = None,
    ) -> None:
        self._on_change = on_change
        self._on_steer = on_steer  # called(text) when a mid-turn steer is captured (Esc + text)
        self._controller = controller or get_pause_controller()
        self._queue: deque[str] = deque()
        self._buffer = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        """True if we can actually poll the console for keys in this environment."""
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
        except Exception:
            return False
        return _HAS_MSVCRT or _posix_supported()

    def start(self) -> None:
        """Spin up the reader for one graph-execution segment. No-op if it can't poll the console
        or is already running."""
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader to exit and wait briefly for it. Any half-typed (never Enter-committed)
        line is dropped — only complete lines queue — so a partial buffer can't leak into the prompt
        that follows."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._buffer = ""
        self._notify()

    def pop(self) -> Optional[str]:
        """Pull the oldest queued line (FIFO), or None if nothing is queued. The REPL calls this
        between turns to run type-ahead before blocking on the prompt."""
        with self._lock:
            line = self._queue.popleft() if self._queue else None
        if line is not None:
            self._notify()  # keep the displayed queue depth honest as we drain
        return line

    def pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

    # ── internals ────────────────────────────────────────────────────────────────
    def _notify(self) -> None:
        if self._on_change is None:
            return
        with self._lock:
            buf, n = self._buffer, len(self._queue)
        try:
            self._on_change(buf, n)
        except Exception:
            pass  # a display hiccup must never kill the reader thread

    def _on_escape(self) -> None:
        """Esc handling. With text already typed, consume it as a mid-turn steering correction
        (injected into the running turn at the next step boundary — see plan_gate); with an empty
        line, request a plan-review pause instead. Either way it routes through the shared
        PauseController, distinguished by source ('steer' vs 'user')."""
        with self._lock:
            text = self._buffer.strip()
            self._buffer = ""
        if text:
            self._controller.request("steer", text)
            self._notify()  # the typed line became a steer — clear it off the status bar
            if self._on_steer is not None:
                try:
                    self._on_steer(text)
                except Exception:
                    pass  # a display hiccup must never kill the reader thread
        else:
            self._controller.request("user", "you pressed Esc to review the plan")

    def _on_char(self, ch: str) -> None:
        """Fold one captured character into the line buffer / queue. (Esc is handled by the loops
        directly so the POSIX path can distinguish a lone Esc from an arrow-key escape sequence.)"""
        if ch in _ENTER:
            with self._lock:
                line = self._buffer.strip()
                self._buffer = ""
                if line:
                    self._queue.append(line)
            self._notify()
        elif ch in _BACKSPACE:
            with self._lock:
                self._buffer = self._buffer[:-1]
            self._notify()
        elif ch >= " ":  # any printable character — control chars (Tab, Ctrl-*) are ignored
            with self._lock:
                self._buffer += ch
            self._notify()

    def _loop(self) -> None:
        if _HAS_MSVCRT:
            self._loop_windows()
        else:
            self._loop_posix()

    def _loop_windows(self) -> None:  # pragma: no cover - platform/IO specific
        while not self._stop.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in _WIN_PREFIX:
                        if msvcrt.kbhit():
                            msvcrt.getwch()  # discard the special key's trailing scancode
                        continue
                    if ch == _ESC:  # lone Esc on Windows — special keys arrive via _WIN_PREFIX
                        self._on_escape()
                        continue
                    self._on_char(ch)
                else:
                    self._stop.wait(0.03)
            except Exception:
                return

    def _loop_posix(self) -> None:  # pragma: no cover - platform/IO specific
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # char-at-a-time, so select+read(1) returns keys before Enter
        except Exception:
            saved = None
        try:
            while not self._stop.is_set():
                try:
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if not r:
                        continue
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    if ch == _ESC:
                        # Arrow / function keys arrive as an ESC-led sequence; if more bytes are
                        # immediately available it's a sequence (drain + ignore), else a lone Esc.
                        if select.select([fd], [], [], 0.0005)[0]:
                            while select.select([fd], [], [], 0.0)[0]:
                                sys.stdin.read(1)
                            continue
                        self._on_escape()
                        continue
                    self._on_char(ch)
                except Exception:
                    return
        finally:
            if saved is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                except Exception:
                    pass


def _posix_supported() -> bool:  # pragma: no cover - platform specific
    """Best-effort check that we can put a POSIX TTY into cbreak and do single-char reads."""
    try:
        import select  # noqa: F401
        import termios  # noqa: F401
        import tty  # noqa: F401

        return True
    except Exception:
        return False
