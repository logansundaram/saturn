"""
Pause-and-review mechanism for the living-plan loop — the modular seam that lets execution be
*interrupted at a step boundary* so the plan can be inspected and corrected, then resumed.

  PauseController   — a tiny, thread-safe, process-level singleton. ANY source asks for a pause
                      by calling `request(source, reason)`; the `plan_gate` node consults it at
                      each step boundary (`pending()` / `peek()`) and clears it once handled.
                      This is the one place the rest of the system reads "should we pause?".

The *user-initiated* trigger — a daemon thread that watches the console during a turn and calls
`controller.request(...)` when the pause key (**Esc**) is pressed — now lives in `typeahead.py`'s
`InputQueue`, which is the single console reader for the duration of a turn (it also captures
type-ahead so the user can queue follow-up queries/commands). A console that can't be polled
degrades to a no-op there, and the gate still works via `/plan pause`, `/plan review`, or the
in-graph `state["pause_requested"]`.

Why a singleton rather than threading the controller through graph state/config: the CLI runs
exactly one turn at a time (blocking), so a single shared controller is unambiguous, needs no
serialization through the checkpointer, and keeps the gate node a pure `state -> updates`
function. The graph-state field `pause_requested` is the *other* seam — an in-graph source (e.g.
a future LLM `request_plan_review` tool/node) can set it and the same gate handles it identically.
See `nodes/plan_gate.py`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


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
