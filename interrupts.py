"""
Pause-and-review mechanism for the living-plan loop — the modular seam that lets execution be
*interrupted at a step boundary* so the plan can be inspected and corrected, then resumed.

Two layers, deliberately decoupled so the *trigger* can change without touching the *handling*:

  PauseController   — a tiny, thread-safe, process-level singleton. ANY source asks for a pause
                      by calling `request(source, reason)`; the `plan_gate` node consults it at
                      each step boundary (`pending()` / `peek()`) and clears it once handled.
                      This is the one place the rest of the system reads "should we pause?".

  KeyWatcher        — the v1 *user-initiated* trigger: a daemon thread that watches the console
                      for the pause key while a turn executes and calls `controller.request(...)`.
                      Entirely optional and isolated — if the console can't be polled (not a TTY,
                      no msvcrt/POSIX termios) it degrades to a no-op and the rest still works via
                      `/plan pause`, `/plan review`, or the in-graph `state["pause_requested"]`.

Why a singleton rather than threading the controller through graph state/config: the CLI runs
exactly one turn at a time (blocking), so a single shared controller is unambiguous, needs no
serialization through the checkpointer, and keeps the gate node a pure `state -> updates`
function. The graph-state field `pause_requested` is the *other* seam — an in-graph source (e.g.
a future LLM `request_plan_review` tool/node) can set it and the same gate handles it identically.
See `node_registry/plan_gate.py`.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Optional

# Windows console key polling. The primary path on this repo's platform (win32); absent
# elsewhere, where we fall back to a best-effort POSIX reader, then to a no-op.
try:
    import msvcrt  # type: ignore

    _HAS_MSVCRT = True
except Exception:  # pragma: no cover - non-Windows
    _HAS_MSVCRT = False


@dataclass(frozen=True)
class PauseRequest:
    """A single request to pause at the next step boundary. `source` is who asked ('user',
    'review', and — later — 'llm'); `reason` is the human-readable why, shown at the prompt."""

    source: str
    reason: str = ""


class PauseController:
    """Thread-safe latch for 'pause at the next boundary'. A source sets it via `request()`; the
    gate reads it non-destructively (`pending()`/`peek()`) and `clear()`s it once it has handled
    the interrupt.

    The read is intentionally non-destructive: the `plan_gate` node re-executes from the top when
    a LangGraph `interrupt()` resumes, so the path to the interrupt must be identical both times.
    `clear()` runs only *after* the interrupt returns, so `pending()` stays true across the
    pause/resume boundary and the control flow is deterministic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: Optional[PauseRequest] = None

    def request(self, source: str, reason: str = "") -> None:
        """Ask for a pause at the next step boundary. Latest request wins (cheap and harmless —
        only the most recent reason is shown)."""
        with self._lock:
            self._request = PauseRequest(source=source, reason=reason)

    def pending(self) -> bool:
        with self._lock:
            return self._request is not None

    def peek(self) -> Optional[PauseRequest]:
        """Read the pending request without clearing it."""
        with self._lock:
            return self._request

    def clear(self) -> None:
        with self._lock:
            self._request = None


# Process-level singleton — every source and the gate share this one instance.
_controller = PauseController()


def get_pause_controller() -> PauseController:
    return _controller


class KeyWatcher:
    """User-initiated pause trigger: while running, a daemon thread watches the console for the
    pause key and asks the shared `PauseController` to pause at the next step boundary.

    `run_turn` brackets graph execution with `start()`/`stop()` so the watcher is live only while
    the graph is running — never while a blocking `input()` (the `»` prompt, the approval gate, the
    plan-review editor) is reading a line, so it can't steal those keystrokes. Keystrokes typed
    during execution are consumed here (kept out of the next prompt's buffer); a press of the pause
    key arms the controller and surfaces at the next `plan_gate`.

    No-ops cleanly when the console can't be polled (not a TTY, or no msvcrt/POSIX termios), so the
    feature simply isn't available there and the rest of the loop is unaffected."""

    def __init__(self, keys: str = "p", controller: Optional[PauseController] = None) -> None:
        # The key(s) that trigger a pause, matched case-insensitively.
        self._keys = {k.lower() for k in keys}
        self._controller = controller or get_pause_controller()
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
        return _HAS_MSVCRT or _posix_poll_supported()

    def start(self) -> None:
        """Spin up the watcher for the duration of one graph-execution segment. No-op if it can't
        poll the console or is already running."""
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the watcher to exit and wait briefly for it (it polls on a short cadence)."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None

    def _trigger(self, ch: str) -> None:
        self._controller.request("user", f"you pressed '{ch}' to review the plan")

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
                    if ch and ch.lower() in self._keys:
                        self._trigger(ch)
                    # other keys are consumed silently so they don't leak into the next prompt
                else:
                    self._stop.wait(0.05)
            except Exception:
                return

    def _loop_posix(self) -> None:  # pragma: no cover - platform/IO specific
        import select

        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch and ch.lower() in self._keys:
                        self._trigger(ch)
            except Exception:
                return


def _posix_poll_supported() -> bool:  # pragma: no cover - platform specific
    """Best-effort check that we can do a non-blocking single-char read on a POSIX TTY."""
    try:
        import select  # noqa: F401

        return True
    except Exception:
        return False
