"""
CLI rendering for the agent console — a serious local-agent console (git status / htop / a trace
viewer), not a chatbot: dense, keyboard-first, low-noise, inspectable. The rules every surface
follows:

  - Everything sits on a 2-space indent with one ` · ` rhythm, and exactly TWO rule glyphs exist:
    `── title` opens a block (`listing.section` is the one implementation) and `╶ ` marks a quiet
    meta-footer (the receipt, a recorded answer).
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk. Structure is dim.
    The MARKER carries the state and color only reinforces it — everything must still read under
    `NO_COLOR=1`.
  - A dim `│` rail carries the execution trace; metrics ride it dim, never louder than the trace.
  - The approval gate is the ONE surface that gets to shout (a heavy `┏━ ┃ ┗━` frame); everything
    else recedes.
  - The live status bar is torn down around every input() (the `»` prompt, the gate, the final
    response) — `input()` can't run inside an active `Live` — and restarted as the turn continues.

The agent emits node/plan/state updates; this package is one subscriber that renders them. Split
by screen concern: `_base` (console/palette/shared state/primitives), `statusbar`, `art`, `prompt`,
`trace`, `plan`, `approval`, `response`, `readouts`, `listing`. Callers use the flat
`from tui import ui; ui.foo()` surface re-exported below — nothing imports the submodules directly.
"""

# Trace verbosity (state + accessors live in _base alongside the rest of the shared state).
from ._base import set_verbosity, verbosity

# Status bar + per-turn reset.
from .statusbar import set_input_preview, set_thinking, reset_turn

# Startup splash.
from .art import splash

# Input prompt + banner (+ the session-start trust posture line, the ask_user answer prompt and
# the Esc pause prompt).
from .prompt import (prompt, banner, ask, answer_question, pause_prompt, posture_line,
                     line_was_pasted)

# Execution trace + recorded replays.
from .trace import show_node, show_run, show_llm_calls

# Plan rendering (the model's checklist).
from .plan import render_plan, show_plan

# Approval gate.
from .approval import ask_approval

# Final answer (streamed + non-streamed).
from .response import response, ResponseStream

# On-demand readouts + log lines.
from .readouts import (
    note, warn, steer_note, pause_note, echo_queued,
)

# Shared listing vocabulary (the section rule + aligned table every readout command uses).
from .listing import section, table, risk_style, status_glyph

__all__ = [
    "set_verbosity", "verbosity",
    "set_input_preview", "set_thinking", "reset_turn",
    "splash",
    "prompt", "banner", "ask", "answer_question", "pause_prompt", "posture_line",
    "line_was_pasted",
    "show_node", "show_run", "show_llm_calls",
    "render_plan", "show_plan",
    "ask_approval",
    "response", "ResponseStream",
    "note", "warn", "steer_note", "pause_note", "echo_queued",
    "section", "table", "risk_style", "status_glyph",
]
