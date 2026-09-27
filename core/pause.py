"""The pause latch — the one place the rest of the system reads "should this turn pause or
steer?". ANY source asks with `request(source, reason)`: the Esc key (tui/typeahead.InputQueue —
empty line = pause, text = steer). The agent node (nodes/agent.py) consults it at the top of
every pass: a pause `interrupt()`s the graph for the pause prompt; steers are drained into the
conversation as STEER_PREFIX notes. Process-level singleton: the CLI runs one turn at a time.

Determinism across the interrupt: a resumed `interrupt()` re-executes its node from the top, so
the node reads the pause non-destructively (`pending()`/`peek()`) and `clear()`s only after the
interrupt returns. (Lived in core/plan_ops.py until the plan engine's removal, 2026-09-27.)"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


# ── the pause latch ──────────────────────────────────────────────────────────────────────────
# Pause-and-review mechanism for the living-plan loop: ANY source asks for a pause by calling
# `request(source, reason)`; the `plan_gate` node consults it at each step boundary
# (`pending()` / `peek()`) and clears it once handled. This is the one place the rest of the
# system reads "should we pause?".
#
# The *user-initiated* trigger — a daemon thread that watches the console during a turn and
# calls `controller.request(...)` when the pause key (**Esc**) is pressed — lives in
# `typeahead.py`'s `InputQueue`, the single console reader for the duration of a turn. A console
# that can't be polled degrades to a no-op there, and the gate still works via `/plan pause` and
# `/plan review`.
#
# Why a singleton rather than threading the controller through graph state/config: the CLI runs
# exactly one turn at a time (blocking), so a single shared controller is unambiguous, needs no
# serialization through the checkpointer, and keeps the gate node a pure `state -> updates`
# function. This is THE one pause seam — a future in-graph source (e.g. an LLM
# `request_plan_review` tool/node) calls `request()` on this same controller. See
# `nodes/plan_gate.py`.


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
        self._steers: list = []

    def request(self, source: str, reason: str = "") -> None:
        """Ask for a pause at the next step boundary — or, for source "steer", QUEUE a mid-turn
        correction. Two slots, not one (transplanted from the engine isolate, 2026-08-15): a
        pause is a request to INTERRUPT, a steer a request to adjust WITHOUT interrupting, and
        plan_gate handles them on different paths. Sharing one slot let a steer typed after an
        Esc-pause overwrite the pause (the user saw the ⏸ acknowledgement and never got the
        editor), so steers queue and are drained only when no pause is outstanding — the pause
        outranks the steer, and the path to interrupt() evaluates identically on both LangGraph
        passes. For pauses the latest request wins (only the most recent reason is shown)."""
        with self._lock:
            if source == "steer":
                self._steers.append(PauseRequest(source=source, reason=reason))
            else:
                self._request = PauseRequest(source=source, reason=reason)

    def pending(self) -> bool:
        """Whether a PAUSE is outstanding (queued steers don't count — they never interrupt)."""
        with self._lock:
            return self._request is not None

    def peek(self) -> Optional[PauseRequest]:
        """Read the pending pause without clearing it — or, when no pause is outstanding, the
        oldest queued steer (a read-only view for callers that report what is waiting)."""
        with self._lock:
            if self._request is not None:
                return self._request
            return self._steers[0] if self._steers else None

    def clear(self) -> None:
        """Consume the PAUSE request. Queued steers survive — they were never handled."""
        with self._lock:
            self._request = None

    def steers_pending(self) -> bool:
        with self._lock:
            return bool(self._steers)

    def take_steers(self) -> list:
        """Drain the queued steering corrections, oldest first."""
        with self._lock:
            out, self._steers = self._steers, []
            return out

    def reset(self) -> None:
        """Drop everything outstanding — the turn boundary, or a test starting clean."""
        with self._lock:
            self._request = None
            self._steers = []


# Process-level singleton — every source and the gate share this one instance.
_controller = PauseController()


def get_pause_controller() -> PauseController:
    return _controller
