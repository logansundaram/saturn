"""
The one-off log lines (notes, warnings, steering/pause acknowledgements, queued-line echoes),
rendered in the trace-rail style. None of these touch per-turn state.
"""

from ._base import Text, _console, _ACCENT, _DIM, _truncate


# ── log lines (startup notices, warnings) ────────────────────────────────────────
# Every line below is the same shape: a glyph, a message, and an optional dim tail.
def _glyph_line(glyph: str, glyph_style: str, text: str,
                text_style: str = "default", tail: str = "", tail_style: str = "") -> None:
    t = Text()
    t.append(glyph, style=glyph_style)
    t.append(text, style=text_style)
    if tail:
        t.append(tail, style=tail_style or _DIM)
    _console.print(t)


def note(msg: str) -> None:
    """A quiet informational line (dim) — e.g. the `@file` attachment notice. Distinct from
    `warn` (yellow), which flags a problem; a note is just neutral context."""
    _glyph_line("  · ", _DIM, msg, _DIM)


def warn(msg: str) -> None:
    _glyph_line("  ! ", "yellow", msg, "yellow")


def steer_note(text: str) -> None:
    """Acknowledge a mid-turn steering correction the moment it's captured (Esc with typed text).
    The correction is injected into the running turn at the agent's next pass (nodes/agent.py);
    this is the immediate feedback that it landed, printed above the live status bar."""
    _glyph_line("  ↪ ", f"bold {_ACCENT}", "steering — applies at the next pass: ",
                _ACCENT, _truncate(text, 80), _DIM)


def pause_note() -> None:
    """Acknowledge an empty-line Esc the moment it's captured. The pause lands at the agent's
    next pass (nodes/agent.py), which on a local model can be a long wait — this is the immediate
    feedback that the keypress registered, printed above the live status bar exactly like
    steer_note's steering acknowledgement."""
    _glyph_line("  ⏸ ", f"bold {_ACCENT}", "pausing at the next pass…", _ACCENT)


def echo_queued(line: str) -> None:
    """Echo a type-ahead line as the REPL pulls it off the queue to run, so a query/command the
    user typed while a previous turn was working shows up in the transcript just like a line typed
    live at the `»` prompt (with a quiet `queued` tag to mark where it came from)."""
    _glyph_line("» ", f"bold {_ACCENT}", line, "default", "   (queued)", _DIM)
