"""
CLI rendering for the agent console.

Design target: a serious local-agent console — git status / htop / pytest / a trace viewer,
not a chatbot. Dense, fast, keyboard-first, low-noise, inspectable. The aesthetic comes from
*structure*, not decoration, and the structure is deliberately spare (2026-06-12 visual
refactor): everything sits on a 2-space indent, info lines share one tight ` · ` rhythm, and
exactly TWO rule glyphs exist — `── title` opens a block (dim dashes, accent title, never a
trailing bar of dashes; `listing.section` is the one implementation) and `╶ ` marks a quiet
meta-footer (the receipt, a recorded answer). The only thing allowed to be heavy is the approval
frame. Specifics:

  - A dim vertical rail (`│`) carries the execution trace. Consecutive node lines form one
    continuous gutter, so a turn reads as a single inspectable block (the htop/tree feel). Each
    node line leads with a green `✓` (the node has finished by the time it prints) then a dim
    `name  elapsed`. At normal verbosity the plumbing nodes (`ground`, `update_plan`, `plan_gate`)
    fold out of the rail — their *output* still prints and the trace DB keeps every node — so a
    turn reads as the user's mental model, `plan → agent → tools → … → synthesize`;
    `set_verbosity("verbose")` (via `/trace full`) restores every node line and full timings.
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk tier. Structure
    is dim. Nothing is colored just to look nice — if it has color, it means something.
  - The plan re-renders **in full** — every row carrying its status glyph AND intended tool —
    each time it materially changes (2026-07-06 faithful-rendering rework): the first draft,
    each completed step of the execute → update_plan loop, a replan's redraft, a rectify
    cancellation, a review edit. (The old one-line status diff hid tools after the first print
    and missed a redraft that kept ids/statuses.) The one fold: a step flipping to bare
    `active` — the execute rail line + reasoning leaf in the same delta already name the step
    being worked, so that flip rides silently into the next material render.
  - The `tools` node renders a **tool-I/O sub-tree** under its header: one `├─ name(args)  dur`
    branch per call, the call repr sized to the terminal width and durations column-aligned. Raw
    result previews are **hidden** by default (noisy JSON) — a failed call still shows its error
    inline (wrapped under the rail with a hanging indent), and `/trace calls` or `/trace full`
    surfaces full outputs on demand.
  - LLM nodes annotate their trace line with the live **metrics for that step** (iteration,
    context tokens ingested, tok/s) — rendered **dim**: metrics are tertiary and must never
    out-shout the trace they ride on, let alone the response. The eye flows response → trace →
    metrics without having to parse the screen.
  - The approval gate is the ONE surface that gets to shout: a heavy `┏━ ┃ ┗━` frame (no trailing
    fill — the weight is the frame itself), set off by a blank line, breaking out of the dim
    rail. It's a blocking safety decision and *should* draw the eye; everything else recedes.
    Each gated call shows its risk tier, every argument on its own line, and a one-line "what
    allowing this means" hint. The plan-review pause wears the same frame.
  - The final **response** renders under a `── response` rule as real markdown (headings, bold,
    lists, fenced code) at the app's 2-space indent, its measure capped (~100 cols) so prose
    stays readable on a wide terminal. The trust-colored Sources block and the receipt — trust
    facts first WHEN the turn deviated (sends / blocked / gated; a calm local turn shows just
    the dim run stats), corrections, then the stats — close every answer.
  - A single-line **status bar** is pinned at the bottom of the screen for the duration of a turn
    (`rich.live.Live`): **posture** (deviation-only: `⚠ GATE OFF` / a loosened tier / ⛓ AIRGAP;
    the calm read_only default renders nothing; leftmost so right-edge trimming sacrifices it
    last) │ **type-ahead**
    (only while queuing input) │ **progress** (`▸node · iter · tools · elapsed · tok/s`) │
    **session** (`ctx NN% ▰▱` meter · `⇅` egress count, each only once it has
    something to say) │ **hardware** (bare load-colored `cpu/ram/gpu/vram NN%`, sampled
    off-thread so nvidia-smi never stalls the render) │ the dim key legend, last on purpose. It's
    no-wrap + ellipsis so a narrow terminal trims the right edge rather than wrapping; transient,
    so it vanishes when the turn ends. Because `input()` can't run inside an active `Live`, the
    bar is torn down around the `»` prompt, the approval gate, and the final response, then
    restarted as the loop continues.

The agent emits node/plan/state updates; this module is one subscriber that renders them.
Swapping it for a Textual/Electron surface needs no graph change.
Degrades to plain ASCII-ish output if `rich` is absent (still UTF-8: stdout is reconfigured in
agent.py, so box-drawing glyphs are safe even on the no-color path).

This is a package, split by screen concern: `_base` (console/palette/shared state/primitives),
`statusbar`, `art`, `prompt`, `trace`, `plan`, `approval`, `response`, `readouts`, `listing`
(the shared section-rule + aligned-table vocabulary every readout command renders through).
Callers use the flat `from tui import ui; ui.foo()` surface re-exported below — nothing imports
the submodules directly.
"""

# Trace verbosity (state + accessors live in _base alongside the rest of the shared state).
from ._base import set_verbosity, verbosity

# Status bar + per-turn reset. (The gate indicator reads the live policy itself — there is no
# set_gate_off flag to keep in sync anymore.)
from .statusbar import set_input_preview, reset_turn

# Startup splash.
from .art import splash

# Input prompt + banner (+ the session-start trust posture line + the ask_user answer prompt).
from .prompt import prompt, banner, ask, answer_question, posture_line

# Execution trace + recorded replays.
from .trace import show_node, show_run, show_llm_calls, show_llm_context

# The Glass Box — answer-level provenance.
from .glass import show_glassbox

# Plan rendering + the plan-review editor.
from .plan import render_plan, show_plan, review_plan

# Approval gate (+ the diff helper the tests reach for).
from .approval import ask_approval, _diff_lines

# Final answer (streamed + non-streamed) + the per-turn provenance handoffs (Glass Box sources
# + the interrupt-and-correct answer buffer).
from .response import response, ResponseStream, set_turn_provenance, set_turn_buffer

# The freeze editor (interrupt-and-correct's freeze-then-edit interaction).
from .correction import edit_answer

# On-demand readouts + log lines.
from .readouts import (
    show_system_metrics, show_context, show_models,
    note, warn, steer_note, pause_note, freeze_note, echo_queued,
)

# Shared listing vocabulary (the section rule + aligned table every readout command uses).
from .listing import section, table, risk_style

__all__ = [
    "set_verbosity", "verbosity",
    "set_input_preview", "reset_turn",
    "splash",
    "prompt", "banner", "ask", "answer_question", "posture_line",
    "show_node", "show_run", "show_llm_calls", "show_llm_context",
    "show_glassbox",
    "render_plan", "show_plan", "review_plan",
    "ask_approval",
    "response", "ResponseStream", "set_turn_provenance", "set_turn_buffer",
    "edit_answer",
    "show_system_metrics", "show_context", "show_models",
    "note", "warn", "steer_note", "pause_note", "freeze_note", "echo_queued",
    "section", "table", "risk_style",
]
