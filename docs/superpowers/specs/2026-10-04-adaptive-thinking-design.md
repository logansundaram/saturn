# Adaptive thinking — which passes think, and a thought the user can see, bound and stop

Date: 2026-10-04. Status: **built and measured 2026-10-04** (`core/think.py`, `nodes/agent.py`,
`stores/trace.py`, `tui/ui/`, `commands/think.py`, `benchmark.py`). **`auto` is one rule — think
before a pass acts, never before a text answer (`act`)** — chosen on the loop benchmark; the
candidate policies and `runtime.think_policy` described in the design below were removed once
it was chosen. Phase 5 (sampling) is not run. Supersedes pivot "Improve" 4 ("think on
evidence", shipped 2026-09-29). Plan: `../plans/2026-10-04-adaptive-thinking.md`. Read
"Measured and decided" and "As built" at the end before the design: they override it.

## The problem

Thinking is the one engine decision made on every pass, and the user feels both sides of it: a
pass that thinks costs seconds, a pass that should have thought costs the task. Logan's ask
(2026-10-04): adaptive thinking is a pillar feature of Saturn, not a tuning knob.

Today (`nodes/agent._wants_think`) a pass thinks only right after a tool round with an error.
What the review found (2026-10-04):

- **The trigger misses where the mistakes are.** One loop-benchmark run per mode on the 4b:

  | `runtime.think` | passed | thinking passes | empty (rerun) | suite time |
  |---|---|---|---|---|
  | `off` | 25/34 | 0 | 0 | 155 s |
  | `adaptive` (today) | 23/34 | 6 | 0 | 160 s |
  | `on` | 29/34 | 81 | 7 | 245 s |

  The three `adaptive` runs of 2026-10-03 were also 23/34. The failures are first-move
  mistakes — a wrong tool, a needless tool, no question asked — on passes the trigger never
  reaches. `on` fixed 8 of `adaptive`'s 11 failures and broke 2. The 9b is unmeasured.
- **After an error the thought restates the error.** All 7 thinking passes in the trace DB are
  30–60 tokens paraphrasing an error text that already names its remedy.
- **The empties are predictable.** All 7 empty thinking passes under `on` were the wrap-up
  answer; 5 came right after a completed write, edit or move.
- **Bugs.** `think: on` written by hand in `config.yaml` is YAML `True` and silently runs as
  adaptive; a model that rejects the think flag still gets a doubled `num_predict` and an
  identical "think-off" rerun; `ASK_ALONE_TEXT` is stamped `error`, so the pass after the user
  answers a question thinks although nothing failed.
- **A thought is invisible and unbounded.** `app/turn.py` forwards only `content`, so a
  thinking pass is silence; nothing records that a pass was asked to think; the only bound is
  `num_predict`, 8× the longest thought observed (509 tokens); Esc waits for the thought to end.
- **The only control is a config key.**

## What the outside does (survey 2026-10-04)

- Where the weights allow it the model decides per request (Anthropic adaptive mode, Gemini
  dynamic thinking, GPT-5-Codex). Qwen3.5 cannot: its card says the `/think` soft switch is not
  supported, so the harness decides.
- OpenAI's ChatGPT router (August 2025) is the cautionary tale: silent switching read as a
  broken model, the added waits reportedly hurt usage, a manual picker returned within a week,
  and auto-switching was retired tier by tier through September 2026. The lesson: default fast,
  make the decision visible, keep a manual level.
- Thinking helps small models in tool loops: Qwen3-8B gains 8–13 points on BFCL and τ-bench
  with thinking on (arxiv.org/html/2511.05951, Table 2).
- Where in the loop it pays is mixed. Anthropic's docs: the first request after user input
  carries most of the reasoning and passes that only process tool results can skip it. ARES
  (arxiv.org/html/2603.07915): a per-step router matches always-high effort with 35% fewer
  reasoning tokens, spending on recovery steps and complex later observations, little on simple
  early ones. No published comparison of "think before the first action" against "every step".
- Every product shows a timer and a summary and offers a level plus a one-turn override; no
  open-source agent checked decides per step.
- Thinking counts against the output cap on every stack, and empty answers after thinking with
  tools on Qwen3 are a known Ollama issue (ollama/ollama#10976).

## Goals

1. `auto` beats today's rule on the loop benchmark on both the 4b and the 9b without making a
   chat turn slower. Measured, three runs per tier.
2. Whether a pass thought, why, and for how long is on screen while it happens and in the
   record afterwards.
3. A thought is bounded in tokens and stops on Esc. A pass makes at most two model calls
   because of thinking, whatever goes wrong.
4. The user has one front door: a session level and a one-turn override.

Not goals: a learned router, a classifier call, reading the request text for "hard" words.

## Decisions

- **The harness decides, by the kind of step.** A pure function of this turn's messages — no
  model call, no request-text heuristics (v1's regex quick path is not coming back). The
  decision is deterministic, so it can be tested, shown and explained.
- **The default is chosen by measurement, under a rule written down first** (below). Until the
  runs are in, `auto` behaves exactly as today.
- **A chat turn must not pay.** Hence the draft-first candidate: the first pass runs think-off;
  a text answer stands at zero cost, and only a pass that is about to *act* is rethought.
- **Three levels in the user's words: `fast` / `auto` / `deep`.** The old `off` / `adaptive` /
  `on` keep working.
- **A thought never enters the conversation** (unchanged; the Qwen3.5 card asks for the same).
  It is shown and recorded, not replayed to the model.
- **Esc keeps its one meaning** (pause at the next boundary). The addition: a pending pause ends
  a thought in flight, so the boundary arrives now.

## Design

### 1. The decision (`core/think.py`, new)

Three pure functions, no model, no I/O beyond config:

- `level(state) -> "fast" | "auto" | "deep"` — the turn's override if set, else `runtime.think`
  normalised (§5).
- `step_kind(this_turn, capped) -> str` — first match wins:

  | kind | when |
  |---|---|
  | `capped` | the pass is at or past `runtime.max_iterations` |
  | `recovery` | the latest tool round has an `error` stamp (a tool failure or a hygiene refusal; `ASK_ALONE_TEXT` excluded — it is mechanical) |
  | `steered` | a steer note arrived after the turn's last agent message |
  | `first` | no tool round yet this turn |
  | `wrap-up` | the latest round has no error and every completed call was an action, or nothing completed (all declined or blocked) |
  | `information` | anything else: a read, a search, a `plan`, an `ask_user` answer came back |

  "Action" is the rule `nodes/tools.py` already applies for Sources (`side_effecting`, or
  `destructive` and not untrusted). It moves into one helper both call.
- `decide(level, kind, policy, supported) -> (think: bool, draft: bool, why: str)`:

  | kind | `fast` | `auto` · `recover` (today) | `auto` · `decide` | `auto` · `decide-draft` | `deep` |
  |---|---|---|---|---|---|
  | `first` | – | – | think | draft, think if it calls a tool | think |
  | `information` | – | – | think | think | think |
  | `recovery` | – | think | think | think | think |
  | `steered` | – | – | think | think | think |
  | `wrap-up` | – | – | – | – | think |
  | `capped` | – | – | – | – | – |

  `supported` is false once the daemon has rejected the think flag for this tag
  (`llms._NO_THINK_SUPPORT`, learned on the first rejection as today): the answer is then
  never-think, `num_predict` is not widened, and no rerun happens.

  `wrap-up` is a prediction, not a restriction: a think-off pass can still call a tool.

`nodes/agent._wants_think`, `_latest_round` and `_EVIDENCE_STATUS` move here; the node keeps one
call.

### 2. The pass (`nodes/agent.py`)

One rule bounds everything: **a pass makes at most two model calls because of thinking.**

- **think** — the thinking call. If it comes back empty, over budget, or stopped by Esc, the
  pass is rerun once think-off (today's empty rerun, widened to the two new causes).
- **draft** — the think-off call first. A text answer is the answer. A tool call is retracted
  from the stream (`RETRACT`), discarded, and the pass is rerun thinking; if that thought comes
  back empty or cut, the *draft* stands. No third call.
- The capped pass, the malformed-output retry and the unbound hard stop are unchanged.

`_generate` keeps its seam signature. It reports the thought on the message it returns:
`response_metadata["saturn_thought"] = {"seconds", "tokens", "text", "cut": None | "budget" | "esc"}`.

### 3. A bounded thought (`nodes/agent._generate`)

The stream loop already folds chunks. While chunks carry `reasoning_content` and no content or
call has begun, it counts them (one token per chunk from Ollama; an approximation is fine for a
circuit breaker) and closes the stream when:

- the count passes `runtime.think_budget` — now "the most tokens one thought may spend",
  default **1024** (twice the longest thought observed), down from 4096; `num_predict` on a
  thinking pass stays `NUM_PREDICT[task] + think_budget`, so the answer still fits; or
- a pause is pending on `core.pause` (the user pressed Esc).

Either way the pass falls to its think-off rerun (§2). The pause itself is not consumed: it is
handled at the next pass boundary exactly as today, or dropped when the rerun is the answer.
`llms.stream`'s generator already closes the underlying stream when its consumer stops.

### 4. What the user sees

- **While it thinks.** `_generate` emits `{"type": "thinking", "phase": "start" | "end"}` on the
  custom stream (the channel `RETRACT` uses); `app/turn.run_turn` hands it to a new
  `on_thinking` callback; the status bar's node cell reads `thinking 3s · esc stops` on its own
  clock. Headless prints nothing.
- **After the pass.** The agent's rail row gains `thought 1.8s`; under a tool-calling pass a dim
  leaf shows the reason and the first 280 characters: `└ thought (first move): …`. A thought
  that was cut or empty says so: `thought cut at 1024 tokens — answered without it`. The answer
  pass's row stays quiet as today; its `thought 1.8s` joins the receipt line.
- **Afterwards.** `/trace why` already prints `thought:`; it gains the kind and the outcome.

### 5. The levels and `/think` (`commands/think.py`, new)

- `runtime.think: fast | auto | deep`. Read through one normaliser: `off`/`false`/`False` →
  `fast`, `adaptive` → `auto`, `on`/`true`/`True` → `deep`. Any other value → `auto`, with a
  line in the startup config problems. This fixes the YAML-boolean bug.
- `runtime.think_policy: recover | decide | decide-draft` — what `auto` means. Template default
  `recover` until the measurement says otherwise; it stays as the escape hatch afterwards.
- `/think` — the readout: the level, what `auto` does in one line per step kind, whether the
  bound model can think, and the last turn pass by pass
  (`pass 1 · first move · thought 1.8s`, `pass 3 · wrap-up · no thought`).
- `/think fast|auto|deep [--session]` — sets `runtime.think`, persisted like `/config`.
- `/think <request>` — runs that one request at `deep`. The rule: exactly one level word (plus
  flags) sets the level; anything else is a request. It sets `state["think_level"]` (a new
  `AgentState` key, reset each turn like `skill`) through `app/session`, the seam `/name`
  skills use, so `saturn -p "/think …"` works too.
- `/think --help`, as every command.

### 6. The record

- `AgentState.think` — one dict per pass, last write wins, reset each turn:
  `{kind, asked, draft, outcome, seconds, tokens, text}`; `outcome` is one of `none`, `thought`,
  `empty`, `cut-budget`, `cut-esc`, `unsupported`. The rail, `/think` and the trace events read
  it; it never enters the prompt. `text` is clipped like every other recorded model text.
- `llm_calls.output` gains `think: true|false` — what the call was *sent* with, read from
  `invocation_params["reasoning"]` at `on_chat_model_start` (the adapter exposes it; checked
  2026-10-04). A draft and its rethink are two rows, as an empty rerun is today.
- `benchmark.py --loop` gains `--think <level or policy>`, `--tier <t>` and `--runs N` (all in
  memory; `config.yaml` is never written). Each task records its passes' `think` dicts and the
  call's prompt-eval seconds; the summary adds thinking passes, drafts discarded, empties, cuts
  and seconds spent thinking. Reports are named `loop_<tier>_<think>_<ts>.json`.

### 7. The measurement, and the rule that picks the default

Candidates for `auto`: `recover`, `decide`, `decide-draft`. Baselines: `fast`, `deep`. Tiers:
4b and 9b. On mains power, unattended.

1. **Screen** — one run of each of the five on both tiers (10 runs, about 70–90 minutes).
2. **Confirm** — three runs each of `recover` and of the best one or two candidates per tier
   (`engine.md` item 13: a delta of 1 is noise).

A candidate becomes the default when, **on both tiers**, over its three runs:

- mean tasks passed ≥ `recover` + 2;
- chat-shape mean latency ≤ `recover` + 10%; and
- suite mean latency ≤ `recover` + 40%.

Between two that qualify, the faster suite wins. If none qualifies on both tiers, the default
stays `recover`, the numbers go into `docs/engine.md`, and a per-tier default becomes Logan's
call. The thresholds are his to change before the runs, not after.

The same runs answer one side question: whether a thinking pass that follows a think-off pass
keeps the daemon's cached prefix (the prompts differ only at the tail). The prompt-eval seconds
in the report say so.

## Phases

Each phase ships alone and leaves the suite green.

0. **Fixes and the record — the policy itself unchanged.** The normaliser; the
   unsupported-model path; `ASK_ALONE_TEXT` out of the evidence; `AgentState.think`; the `think`
   flag in `llm_calls`; the three benchmark flags.
1. **`core/think.py`** — step kinds and the policy table behind `runtime.think_policy`
   (default `recover`), and draft-first in the node.
2. **Measure** (§7) and set the default. Numbers into `docs/engine.md` and
   `docs/OPTIMIZATIONS.md`.
3. **Bounded and visible** — the in-stream budget, Esc stops a thought, the status bar, the
   rail, `/trace why`.
4. **`/think`** and the level names; `config.default.yaml`'s comment rewritten.
5. **Sampling, as an experiment.** Thinking passes run at temperature 0; the Qwen3.5 card
   recommends 0.6–1.0 in thinking mode and the Qwen3 card warned against greedy decoding there.
   Three runs per tier of the shipped policy at 0 against 0.6, graded on the empty rate and
   tasks passed. It ships only if it wins.

Docs when built: `CLAUDE.md` (life of a turn, models and config), `docs/ARCHITECTURE.md`,
`docs/engine.md` (step 4; item 11), `CHANGELOG.md`.

## Tests (offline, through the `_generate` seam)

- `tests/test_think.py` (new): `step_kind` and `decide` as tables — every kind against every
  level and policy; a steer behind a round; a mixed batch; an unsupported model; the normaliser
  over `on`, `True`, `off`, `False`, the old names and a typo.
- `tests/test_agent_loop.py`: draft-first — a text draft is the answer in one call; a drafted
  call is retracted and rethought; an empty rethink keeps the draft; never a third call. A
  thought cut by budget and by a pending pause reruns think-off once. `ASK_ALONE_TEXT` does not
  wake a thought. The existing think tests move to the new names.
- `tests/test_core.py`: `llm_calls.output` carries `think`.
- `tests/test_think_command.py` (new): the readout, set and persist, the one-turn request,
  `--help`, the level-word rule (`/think deep dive into X` is a request).
- The stream bound: a fake stream of reasoning chunks past the budget is closed and reported
  `cut: "budget"`.

## Out of scope

- A learned or model-based router, and any reading of the request text.
- Effort levels inside a thought (low / medium / high): a user report says Ollama ignores them
  for a Qwen3.8 GGUF (ollama/ollama#18766).
- Keeping thoughts in history (`preserve_thinking`, Qwen 3.6+): it changes the cached prefix
  and belongs to its own measurement.
- Continuing from a truncated thought (Qwen's two-call budget recipe): a cut thought is dropped.
- Per-tier policies, and a headless `--think` flag (`-p "/think …"` covers one turn).

## Risks

- **The 4b result may not transfer.** It is one run per mode on one tier; §7 exists for this.
- **Draft-first is unproven.** It assumes rethinking a drafted call recovers what `deep` gains
  on the first move. If it does not, `decide` pays on chat and fails the chat rule, and the
  default stays `recover`.
- **A wrong `wrap-up` prediction** costs a pass its thought, never its tools.
- **Two calls on the first tool pass** under draft-first: a discarded draft is about 40 tokens
  (the think-off median), roughly a second on the 4b. The suite-latency rule accounts for it.

## As built (2026-10-04)

What differs from the design above, and what a short live check on the 4b showed.

- **`state["think"]` is a per-pass list, not one dict.** `/think`'s "last turn, pass by pass"
  needs every pass, so the key is an append-only accumulator reset each turn, like
  `tool_events`. Each entry also carries `pass`, `why` and `prompt_s` (the call's prompt-eval
  seconds). The rail reads the delta's last entry.
- **A sixth outcome, `malformed`:** a thinking call whose output could not be parsed twice.
  It is not an empty thought and nothing is rerun; the turn answers with the malformed-output
  text as before.
- **The cut lands AT the budget** (1024 reasoning tokens read, then the stream is closed).
- **`think_budget: 0` turns the in-stream cut off** rather than cutting every thought at once;
  `num_predict` still bounds the pass.
- **A model that rejects the think flag is recorded `unsupported`**, learned on the first
  rejection as today; the spec's "capabilities lack `thinking`" check was dropped (it needs a
  daemon read at decision time).
- **The rail's row says `thought dropped`** for a thought that was cut or empty; the leaf gives
  the reason. The thought leaf is drawn only on the live rail, not in `/trace` replay —
  `/trace why` has a "when it thought" section instead, shown for a turn where thinking came up.
- **`/think` is in `/help`'s daily list** and in the "trust & control" group.
- **The startup warning** for a bad `runtime.think` / `runtime.think_policy` is printed by the
  REPL and by headless (stderr), not through `check_models` (that list is model health).
- **`engine.md` item 12** (grade the recorded reasoning) is not closed by this work; item 11's
  flag is, and its measurement is what remains.
- **Not tested end to end:** the interactive status bar and Esc during a live thought, and the
  REPL's `/think <request>` wiring (the headless one is). Their parts are unit-tested (the
  events, the bar, the stream cut with a pending pause, `think_for_line`); nobody has pressed
  Esc in a terminal against a thinking model yet.
- **An independent review of the diff (2026-10-04) found seven things; all fixed:**
  `config.persist` opened `config.yaml` for writing before computing the edit, so persisting a
  key the file lacks emptied it — and `/think <level>` persists by default (now the edit is
  computed first, and `/think` says which line to add); `/trace why` showed a discarded draft
  as a call the agent chose and numbered model calls, not passes (now grouped by pass, the
  draft named as a draft); Esc that stopped a thought was then reported as "arrived after the
  turn had finished", and the legend promised the stop while text was typed (Esc + text is a
  steer and does not stop a thought); `--runs N` aborted after a run that left an unexpected
  `bench_*` file; a malformed thinking call was recorded `empty`; the cut was one token past
  the budget; and a claim here that a think-off pass's reasoning is measured (langchain-ollama
  surfaces reasoning only on a call sent with the flag on).

Live check (4b, 2026-10-04, five short turns): a `deep` turn recorded a 6.3 s first-move
thought and a 1.6 s one after the tool round; `think_budget: 8` cut both thoughts at 9 tokens
and the turn still answered; under `decide-draft` "Hi there!" was one call (1.7 s) and
"What is 17 times 23?" drafted a `calculate` call, rethought it in 1.1 s with 0.03 s of prompt
eval (the cached prefix held) and answered. `thinking` start/end events reached
`run_turn(on_thinking=)`, and `llm_calls` recorded `think: true` with the reasoning.

## Measured and decided (2026-10-04)

The loop benchmark, 34 tasks, both tiers, on mains power. Two candidates were added after the
first screening pass showed where the thinking was wasted: `first` (think on the first move
only, drafted) and `act` (draft every deciding pass; think only when the draft calls a tool).

| mode | 4b passed · suite | 9b passed · suite | 9b empty thoughts |
|---|---|---|---|
| `fast` | 24 · 147 s | 29 · 367 s | 0 |
| `recover` | 23.5 · 160 s | 30 · 385 s | 0 |
| `first` | 26 · 227 s | 31 · 425 s | 0 |
| **`act`** | **29.5 · 247 s** | **31 · 455 s** | 0 |
| `decide-draft` | 29.5 · 250 s | 28 · 455 s | 12 |
| `decide` | 29 · 240 s | 29 · 467 s | 12 |
| `deep` | 30 · 234 s | 29 · 472 s | 22 |

4b figures for `recover`, `first`, `act`, `decide-draft` are the mean of two confirmation runs
(the screening runs agreed within one task); the rest, and all of the 9b, are single runs.
Reports: `logging/benchmarks/*_20261004_screen.json` and `*_20261004_confirm<n>.json`.

- **Thinking is worth about six tasks on the 4b and nothing measurable on the 9b.**
- **The waste is thinking before the final answer.** All 12 empty thoughts of a 9b
  `decide-draft` run were the turn's last pass after an information round: the model writes
  the answer inside the thought and returns nothing, and the pass is rerun.
- **`act` never thinks before a text answer**, so it has no empties, matches the best 4b score
  with fewer thoughts (48 against 64), and does not cost the 9b tasks.
- **The price** is suite time +54% on the 4b and +18% on the 9b. Chat is barely touched
  (3.4 → 3.7 s on the 4b); a lookup goes 3.5 → 6.3 s.
- **Two tasks fail under every policy that thinks on the first move** (`file_read`,
  `multi_compare`): the thought adds a cautious extra lookup and the turn exceeds its pass
  limit with the right answer.

**The decision.** By §7's rule `act` does not qualify: the 9b gain is under two tasks and the
4b suite time is over +40%. §7 says the default is then Logan's call. He chose `act`
(2026-10-04) and cut the three-run confirmation short, so the 9b side rests on one run per
mode. He also asked for three knobs and nothing else: `fast`, `auto`, `deep`. So:

- `auto` = `act`, hard-wired. `runtime.think_policy` is gone; `decide`, `decide-draft` and
  `first` are deleted.
- `recover` stays in `core/think.py` only as the benchmark's baseline
  (`benchmark.py --loop --think recover`, `think.set_policy`).
- `/think` and `config.default.yaml` describe one rule: think before acting.

**Later (Logan, 2026-10-04): other ways to decide.** `act` is hand-written. To explore, none
built: a lightweight neural classifier that marks a request complex or simple, trained on
Saturn's own runs (the per-pass think records plus the benchmark's grades are the data); a
learned per-step router (ARES); a `think` tool the model calls itself. Tracked as
`docs/engine.md` item 11a. Any of them has to beat `act` on both tiers.
