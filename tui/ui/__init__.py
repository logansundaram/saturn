"""
CLI rendering for the agent console.

Design target: a serious local-agent console — git status / htop / pytest / a trace viewer,
not a chatbot. Dense, fast, keyboard-first, low-noise, inspectable. The aesthetic comes from
*structure*, not decoration:

  - A dim vertical rail (`│`) carries the execution trace. Consecutive node lines form one
    continuous gutter, so a turn reads as a single inspectable block (the htop/tree feel). Each
    node line leads with a green `✓` (the node has finished by the time it prints) then a dim
    `name  elapsed`. At normal verbosity the plumbing nodes (`ground`, `update_plan`, `plan_gate`)
    fold out of the rail — their *output* still prints and the trace DB keeps every node — so a
    turn reads as the user's mental model, `plan → agent → tools → … → synthesize`;
    `set_verbosity("verbose")` (via `/trace full`) restores every node line and full timings.
  - Color is **semantic only**: green = done, cyan = active, yellow/red = risk tier. Structure
    is dim. Nothing is colored just to look nice — if it has color, it means something.
  - The plan prints **once** as the intended route, then emits a single line per status change
    as steps advance — a log/trace, not a re-rendered panel. This is the transparency surface
    and the main noise source, so it's diffed.
  - The `tools` node renders a **tool-I/O sub-tree** under its header: one `├─ name(args)  dur`
    branch per call, the call repr sized to the terminal width and durations column-aligned. Raw
    result previews are **hidden** by default (noisy JSON) — a failed call still shows its error
    inline (wrapped under the rail with a hanging indent), and `/calls` or `/trace full` surfaces
    full outputs on demand.
  - LLM nodes annotate their trace line with the live **metrics for that step** (iteration,
    context tokens ingested, tok/s) — rendered **dim**: metrics are tertiary and must never
    out-shout the trace they ride on, let alone the response. The eye flows response → trace →
    metrics without having to parse the screen.
  - The approval gate deliberately breaks out of the rail with a heavy rule. It's a blocking
    safety decision and *should* draw the eye; everything else recedes. Each gated call shows its
    risk tier, every argument on its own line, and a one-line "what allowing this means" hint.
  - The final **response** renders as real markdown (headings, bold, lists, fenced code with
    syntax highlighting), so the answer reads as finished output, not a log line.
  - A single-line **status bar** is pinned at the bottom of the screen for the duration of a turn
    (`rich.live.Live`), grouped into three zones parted by a quiet `│` rule so it reads as
    deliberate groups, not a value stream: **identity** (`saturday · model`) │ **progress**
    (`▸node · iter · elapsed · tools · tok/s`) │ **resources** (`ctx NN% ▰▱` with a meter then
    bare `cpu/ram/gpu/vram NN%` load-colored percentages; gpu/vram sampled off-thread by a daemon
    so nvidia-smi never stalls the render). It's no-wrap + ellipsis so a narrow terminal trims the
    right edge rather than wrapping. The trace lines above it keep scrolling normally. It's
    `transient`, so it vanishes when the turn ends. Because `input()` can't run inside an active
    `Live`, the bar is torn down around the `»` prompt, the approval gate, and the final response,
    then restarted as the loop continues.

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

# Input prompt + banner.
from .prompt import prompt, banner, ask

# Execution trace + recorded replays.
from .trace import show_node, show_run, show_llm_calls

# The Glass Box — answer-level provenance.
from .glass import show_glassbox

# Plan rendering + the plan-review editor.
from .plan import render_plan, show_plan, review_plan

# Approval gate (+ the diff helper the tests reach for).
from .approval import ask_approval, _diff_lines

# Final answer (streamed + non-streamed).
from .response import response, ResponseStream

# On-demand readouts + log lines.
from .readouts import (
    show_system_metrics, show_context, show_models,
    note, warn, steer_note, echo_queued,
)

# Shared listing vocabulary (the section rule + aligned table every readout command uses).
from .listing import section, table, risk_style

__all__ = [
    "set_verbosity", "verbosity",
    "set_input_preview", "reset_turn",
    "splash",
    "prompt", "banner", "ask",
    "show_node", "show_run", "show_llm_calls",
    "show_glassbox",
    "render_plan", "show_plan", "review_plan",
    "ask_approval",
    "response", "ResponseStream",
    "show_system_metrics", "show_context", "show_models",
    "note", "warn", "steer_note", "echo_queued",
    "section", "table", "risk_style",
]
