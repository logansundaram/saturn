# Adaptive Thinking Implementation Plan

> **Status: executed 2026-10-04**, inline in the session that wrote it. The node-level tests
> landed in `tests/test_think.py` rather than `tests/test_agent_loop.py` / `tests/test_core.py`.
> The live measurement followed the same day (the spec's "Measured and decided"): `auto` is
> one rule, `act`, and `runtime.think_policy` and the candidate policies named below are gone.
> Not run: the sampling experiment.

**Goal:** the harness decides per pass whether the model thinks, by the kind of step; a thought
is bounded, visible and stoppable; the user has `/think` and three levels.

**Architecture:** a pure decision module (`core/think.py`) the agent node asks once per pass.
`nodes/agent._generate` reports and bounds the thought; the node records one `think` entry per
pass in state, which the rail, `/think`, `/trace why` and the benchmark read. The default policy
stays `recover` (today's behaviour) until the measurement in the spec's §7 is run.

**Tech stack:** Python 3.11+, LangGraph, langchain-ollama, pytest (offline; `_generate` seam).

**Spec:** `docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md`

## Global constraints

- No test may reach a model, the network or the embedder.
- `nodes.agent._generate(llm_input, *, tools, think=False) -> AIMessage` keeps its signature.
- A pass makes at most two model calls because of thinking.
- A thought never enters the conversation or the prompt.
- `runtime.think` accepts `fast | auto | deep` and the old `off | adaptive | on`, strings or
  YAML booleans. `runtime.think_policy: recover | decide | decide-draft`, default `recover`.
- `runtime.think_budget` default 1024.
- Model and tool text never reaches the terminal through a bare `print()`.
- Not in this plan: the live measurement (spec §7) and the sampling experiment (phase 5). Both
  need a running Ollama on mains power.

## Review focus

1. A config value nobody anticipated (`think: yes`, `think: 2`, a typo) — runs as `auto` and
   says so at startup; never crashes a turn. Test in Task 1.
2. A model that rejects the think flag mid-session — no widened bound, no identical rerun, the
   record says `unsupported`. Test in Task 3.
3. Esc pressed while a thought streams — the thought stops, the pass still answers, the pause
   still arrives at the next boundary. Test in Task 3.
4. A drafted tool call whose rethink comes back empty — the draft runs; never a third call.
   Test in Task 3.
5. `/think deep dive into X` — a request, not a level change. Test in Task 6.

## File map

| File | Change |
|---|---|
| `core/think.py` (new) | `normalise`, `level`, `policy`, `supported`, `latest_round`, `step_kind`, `decide`, `entry`, `describe`, `problem` |
| `tools/registry.py` | `is_action(name)` — the Sources rule, one place |
| `nodes/tools.py` | uses `is_action` |
| `core/llms.py` | no widened `num_predict` for an unsupported tag; budget default 1024; `check_models` reports a bad `runtime.think` |
| `core/state.py`, `app/session.py` | `think` (per-pass list) and `think_level` keys |
| `nodes/agent.py` | asks `core.think`; draft-first; thought metadata, budget and Esc cut in `_generate`; one `think` entry per pass |
| `stores/trace.py` | `think` flag on `llm_calls.output`; a cut stream keeps its reasoning |
| `app/turn.py`, `app/repl.py`, `app/headless.py` | `on_thinking`; the `/think <request>` one-turn override |
| `tui/ui/statusbar.py`, `tui/ui/trace.py`, `tui/ui/response.py`, `tui/ui/__init__.py` | `thinking Ns · esc stops`; `thought 1.8s` on the rail and the receipt; the thought leaf |
| `commands/think.py` (new), `commands/__init__.py` | `/think` |
| `commands/trace.py` | `/trace why` prints the kind and outcome |
| `benchmark.py` | `--think`, `--tier`, `--runs`; per-pass records |
| `config.default.yaml` | the three keys and their comment |
| `tests/test_think.py` (new), `tests/test_think_command.py` (new), `tests/test_agent_loop.py`, `tests/test_core.py` | below |
| docs | `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/engine.md`, `docs/OPTIMIZATIONS.md`, `CHANGELOG.md`, the spec's status |

## Tasks

### Task 1: the decision module

**Produces:**

```python
# core/think.py
LEVELS = ("fast", "auto", "deep"); POLICIES = ("recover", "decide", "decide-draft")
KINDS = ("capped", "recovery", "steered", "first", "wrap-up", "information")
def normalise(raw) -> tuple[str, bool]            # (level, recognised)
def level(state=None) -> str                      # state["think_level"] wins, else runtime.think
def policy() -> str                               # runtime.think_policy, unknown -> "recover"
def supported() -> bool                           # tag not in llms._NO_THINK_SUPPORT
def latest_round(this_turn: list) -> list         # moved from nodes/agent
def step_kind(this_turn: list, capped: bool, mechanical: tuple = ()) -> str
class Decision(NamedTuple): think: bool; draft: bool; why: str
def decide(level: str, kind: str, policy: str = "recover", supported: bool = True) -> Decision
def problem() -> str | None                       # the startup line for an unrecognised value
# tools/registry.py
def is_action(name: str) -> bool
```

- [x] Tests first (`tests/test_think.py`): the normaliser over `on`, `True`, `off`, `False`,
  `adaptive`, `Fast`, `yes`, `2`, `None`; `step_kind` as a table (first, clean read, failed
  read, mixed batch, write done, declined, blocked, steer behind a round, steer alone,
  `ASK_ALONE_TEXT` sibling, capped); `decide` over every kind × level × policy; unsupported.
- [x] Implement; `nodes/tools.py` calls `is_action`.
- [x] `pytest tests/test_think.py -q` green; full suite green.

### Task 2: config and the model seam

- [x] Tests: `invoke_kwargs(think=True)` for a tag in `_NO_THINK_SUPPORT` keeps
  `num_predict == 4096` and sends no `reasoning`; the budget default is 1024; `check_models`
  lists a bad `runtime.think`.
- [x] Implement in `core/llms.py`; rewrite the `config.default.yaml` block (`think: auto`,
  `think_policy: recover`, `think_budget: 1024`).

### Task 3: the pass

**Consumes:** Task 1. **Produces:** `state["think"]` entries
`{pass, kind, asked, draft, outcome, why, seconds, tokens, text, prompt_s}` with `outcome` in
`none | thought | empty | cut-budget | cut-esc | unsupported`; custom stream events
`{"type": "thinking", "phase": "start" | "end"}`;
`response_metadata["saturn_thought"] = {seconds, tokens, text, cut}` on `_generate`'s message.

- [x] Tests (`tests/test_agent_loop.py`): the existing think tests still pass under the old
  names; draft-first — a text draft is the answer in one call, a drafted call is rethought, an
  empty rethink keeps the draft, never a third call; a thought reported `cut: "budget"` or
  `"esc"` reruns think-off once and records the outcome; a tag that lands in
  `_NO_THINK_SUPPORT` during the call records `unsupported` and is not rerun; `ASK_ALONE_TEXT`
  wakes no thought; `state["think_level"] = "deep"` thinks on pass one; every pass appends one
  entry. `_generate` itself over a fake stream: reasoning chunks past the budget close the
  stream (`cut: "budget"`), a pending pause closes it (`cut: "esc"`) and stays pending, a
  normal thought reports seconds, tokens and text.
- [x] Implement in `nodes/agent.py`, `core/state.py`, `app/session.py`.

### Task 4: the record

- [x] Tests (`tests/test_core.py`): `llm_calls.output` carries `think` from
  `invocation_params["reasoning"]`; a closed stream records `cancelled` with its partial
  reasoning. `/trace why` prints `first move · thought 1.8s` from the events.
- [x] Implement in `stores/trace.py`, `commands/trace.py`.

### Task 5: what the user sees

- [x] Tests: `run_turn` hands a `thinking` custom event to `on_thinking`; the status bar reads
  `thinking` with a clock and `esc stops thinking` while set; `_metric_parts` adds
  `thought 1.8s`; the leaf reads `thought (first move): …`, and a cut one says so; the receipt
  echoes the turn's thinking time.
- [x] Implement in `app/turn.py`, `app/repl.py`, `tui/ui/`.

### Task 6: `/think`

**Produces:** `app/session.think_for_line(line) -> str | None` (the request of a
`/think <request>` line; None for a bare, level-setting or `--help` line).

- [x] Tests (`tests/test_think_command.py`): the readout; set and persist; `--session`; the
  old level names; `--help`; the level-word rule; the REPL and headless seams set
  `think_level` and strip the prefix.
- [x] Implement `commands/think.py`; wire `app/repl.py`, `app/headless.py`.

### Task 7: the benchmark

- [x] Tests: `--think decide-draft` sets `think=auto` + the policy in memory; `--think deep`
  sets the level; the loop summary counts thinking passes, drafts, empties, cuts and seconds;
  the report name carries tier and mode.
- [x] Implement in `benchmark.py`.

### Task 8: docs

- [x] `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/engine.md` (step 4, items 11–12),
  `docs/OPTIMIZATIONS.md`, `CHANGELOG.md` under `[Unreleased]`, the spec's status and an
  "As built" section naming every deviation.
- [x] Full suite green.
