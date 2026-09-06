# Latency optimizations for a local agent

A working list of techniques for making Saturn faster on a local model, with what is already
in place, what was measured, and what to try next. Companion to `core/serving.py` ("the prefix
cache") and `tests/test_prefix_cache.py`. Numbers are from the 9b tier on an Apple M4 Pro under
Ollama 0.33 (prefill ~400 tokens/s, decode ~37 tokens/s) unless stated otherwise.

Status markers: **[have]** shipped · **[next]** a concrete candidate · **[measure]** plausible,
needs a number before it is worth the complexity.

## 1. Prompt-cache stability (done 2026-09-04)

The daemon (llama-server) restores a prompt only from a saved context checkpoint — 1024 and 4
tokens before an earlier prompt's end, or the point that prompt itself restored from — and
qwen3.5 (hybrid/recurrent) cannot reuse a partial prefix. So a prompt is cheap exactly when
everything that changed sits after such a checkpoint.

- **[have] Stable/dynamic grounding split.** `context_stable` (SATURDAY.md, manifests, the
  always-loaded memory layers) rides every node's prompt as its own message right after the
  system prompt; `context_dynamic` (matched memory, recap, attachments) follows.
- **[have] Idle primes** (`core/prime.py`, `runtime.prime`). After each turn and once after the
  startup warm-up, each lineage's `[system][stable grounding]` prefix is re-sent with one
  predicted token, think ON, so the next turn's calls resume at that boundary. Plan-call
  prefill went from ~2,000 tokens (5 s) to 75–242 tokens per turn.
- **[have] Landing-order result caps** (`plan_context.landing_caps`). A result's cap is fixed
  when it lands; earlier results are never re-truncated, so the results block only ever grows
  at its end. The previous-step callout is bounded at 2,000 chars so a step's changing tail
  fits one prefill batch.
- **[have] Tool arguments under a `format` grammar, not `bind_tools`.** The chat template
  renders bound tools into the system message; every tool step re-prefilled whole (8k tokens,
  20 s). The grammar leaves the prompt untouched.
- **[have] Same load options on every request** (`num_ctx`, `draft_num_predict`, `keep_alive`).
  A mismatch reloads the model and drops the whole cache.
- Measured A/B, same daemon: four-turn wall 65.3 s → 51.4 s; trust benchmark (13 queries)
  581 s → 397 s with verdicts unchanged.

## 2. Fewer tokens through the model

- **[have] Deterministic paths before model calls.** Concrete-step argument fill, rectify's
  deterministic branch order, the write gate's arming rules, the stall detector.
- **[next] Skip the answer rewrite on a single reasoning-step turn.** The step writes the
  answer and synthesize rewrites it (run 45: 452 then 414 tokens, ~10 s of decode). Route such
  plans straight to synthesize with the step's text as the draft, or synthesize from it
  without regenerating.
- **[next] Exact-prompt memoization.** At temperature 0 the same prompt yields the same
  output; a small on-disk cache keyed by the prompt hash skips the daemon for repeated judge
  and argument calls (common in replan loops). Invalidate on model change.
- **[have/next] Tighter output schemas.** Every grammar field is decode time. The judge now
  asks for a one-or-two-sentence rationale (it averaged 113 tokens, ~3 s). Review the other
  shapes for free-text fields a boolean would cover.
- **[next] Structural observation shaping.** Clamp tool output by shape, not just length:
  first N rows of a CSV, matched lines with a little context for search, a diff after an edit
  instead of the whole file.
- **[have] Curated per-step context** (`core/plan_context.py`) instead of raw history; think
  ON only for the planner, steered to think briefly on simple requests.

## 3. Cheaper tokens

- **[measure] Speculative decoding.** Ollama supports a draft model; `draft_num_predict` is
  pinned to 0 because logprob grading (the confidence marking) conflicts with drafting.
  Structured JSON is where a small draft shines — measure with confidence off, or enable
  drafting only for the judge and argument tasks that are never graded.
- **[measure] KV cache quantization and flash attention** (`OLLAMA_KV_CACHE_TYPE=q8_0`,
  `OLLAMA_FLASH_ATTENTION=1`). Cuts cache memory and usually speeds decode on Metal; the gain
  is smaller for a hybrid model than for a pure transformer.
- **[next] Right-size `num_ctx`.** 65k is far more than any turn uses. A smaller window trims
  per-token overhead and leaves room for the embedder to co-reside (see the hazard below).
- **[measure] Model cascade by role.** The role indirection (`core/llms.get_model(role)`)
  makes it cheap to bind a 2b–4b model to `judge`/`tool_caller` and escalate to the 9b when the
  small model's output fails validation. Cost: two resident models, which again argues for a
  smaller window.

## 4. Fewer round trips and overlapped time

- **[next] Prime during slow tools.** During a web search or any slow tool the daemon is idle;
  priming the next execute prefix then (the results ledger up to the new result) is the
  in-turn version of the end-of-turn prime. Note: a prime that will be extended only once
  saves nothing on its own — the win is when the same boundary serves several later calls, or
  when the prime's prefill would otherwise sit on the turn's critical path.
- **[next] Batch the mechanical steps.** When the next several plan steps are concrete reads,
  run them all before returning to the model; only the first step needing generated
  arguments costs a call.
- **[next] Judge less often.** A cheap token-match pre-check ("did the result contain what
  the label asked for") could retire more rectify judge calls before the LLM branch.
- **[have] Streaming and early rendering.** The answer streams; the plan and each result
  render as they land, so perceived latency stays low even when the total does not move.

## 5. Daemon and environment

- **Hazard (seen 2026-09-04):** with `keep_alive: 30m`, a resident qwen3.5:9b at 65k context
  can leave `/api/embed` (loading qwen3-embedding:8b) hanging indefinitely with no scheduler
  log line, while `/api/tags` and chat still answer. `ollama stop qwen3.5:9b` unblocks it.
  `tests/test_cli.py::test_trust_benchmark_searchless_bait_grades_ungrounded` plants documents
  through the live embedder and hangs the suite in that state. Mitigations: a smaller chat
  window, a smaller embedder, or a bounded wait around the ingest.
- Any model reload (a load-option mismatch, an eviction by the embedder) drops the entire
  prompt cache; every lineage then pays a cold prefill once.
- Never run two agents or benchmarks against one daemon at once: with `-np 1` they serialize
  and their prompts evict each other's cache entries.

## 6. How to measure

- The daemon's own log (`~/.ollama/logs/server.log`, verbosity 4) is the ground truth:
  `restored context checkpoint … n_past = N` vs `forcing full prompt re-processing`, and
  `prompt eval time = … / N tokens`. Wall time per node lives in `llm_calls.dur` in
  `database/db.sqlite` for traced runs.
- Replay one call live before hypothesizing (`llm_calls.input` is capped; rebuild the prompt
  with `core.messages.planner_sys_msg()` and the node's own builders). Send test requests with
  the app's exact options (`num_ctx`, `draft_num_predict`, `keep_alive`).
- Validate behavior with `python benchmark.py` on both trees, not timing alone: the one
  regression found on 2026-09-04 (a memory correction stored as a duplicate) came from a tool
  description truncated in the new argument brief, not from the cache work itself.

## 7. Assessment (2026-09-05) — for later consideration

Ranked against the 46 traced turns since 2026-09-02 (`llm_calls` in `database/db.sqlite`), the
daemon log, and the code each idea touches. The traffic shape decides most of it: 41 of 46 plans
are single-step, 17 of 46 are a lone reasoning (`none`) step, the plan call costs 5.5–19 s per
turn for 21–30 output tokens (thinking, not prefill), and the rectify judge said `rectify=true`
13 times in 53 — never once on a clean, fully-done plan.

**Do first**

- **Skip the reasoning-step generation on single-`none` plans** (37 % of turns). Run 45: execute
  wrote the story in 13 s, synthesize rewrote it in 15 s. Route such plans plan_gate → synthesize
  and record the streamed answer as the step's result (synthesize already streams, cites and
  grades confidence — keep that generator). Keep the judge on this path but run it BEFORE
  synthesize on request + plan alone: its only job here is the groundedness rule, and the
  benchmark's `caught_by_rectify` (2 of 3 baits on 2026-09-04) depends on it.
- **Prime the judge lineage.** The judge prompt is `RECTIFY_SYS` + request + plan with no
  grounding message and `core/prime.py` has no lineage for it, so every judge call logs
  `forcing full prompt re-processing` (run 47: 1180 tokens, 2.8 s). A `("judge",
  [RECTIFY_SYS])` lineage is a few lines and changes no behavior.
- **Retire the judge deterministically where its rules cannot fire.** New branch before the LLM
  verdict: plan clean, every step done, and a `web_search` ran ⇒ neither "all steps ran" rule
  applies. In the sample this retires the judge on every clean web turn with zero verdict
  changes.
- **Structural shaping, starting with `web_extract`.** Run 39: the judge itself reported the
  extraction returned "generic navigation text", the turn replanned three times, and synthesize
  prefilled 4.4k tokens for 25 s. A quality fix that also cuts prefill. Files/search are already
  shaped by the caps in `tools/files.py`.

**Worth doing, smaller**

- Tighter schemas: the judge rationale is 30–60 tokens (~1–1.5 s per call); keep reasoning
  before the boolean.
- Prime during slow tools: only multi-step web turns benefit; do it after the judge lineage
  exists.
- Right-size `num_ctx`: on a 48 GB machine the 9b at 65k occupies 8.1 GB, so no speed to gain;
  the only payoff is the embedder hang hazard (§5), which may not be memory at all.

**Close or drop**

- Speculative decoding: already measured negative in `core/confidence.py` (drafting ran SLOWER
  on this Mac for the 27b; the 9b has no drafter). Measured, not pending.
- Flash attention: the runner log already shows `Flash Attention enabled` (`flash_attn = auto`).
- KV cache q8_0: memory is not the constraint, only one in four layers of a hybrid model has a
  KV cache, and quantizing it shifts the logits the per-model confidence calibration was
  measured against.
- Exact-prompt memoization: prompts do not repeat within a session (the results block changes);
  they DO repeat across benchmark runs, which would stop the benchmark measuring the model.
- Batch mechanical steps: concrete-step fill already removed the model call from planned reads
  and rectify's concrete-pending branch is free (run 26's slow reads predate that commit).
- Model cascade: after the judge changes above the judge barely runs, and the planner is the one
  role where the 9b already struggles with think off.

**The lever not on the list:** after the prefix cache, the plan call's thinking is the largest
fixed cost on every turn. Think must stay ON for the planner (off ⇒ a lone `ask_user` stub on
the 9b), so the play is a tighter think budget, not a toggle. First measurement: think-token
count per plan call.

## 8. Decision (2026-09-05) — route around the engine, then trim it

Supersedes the ordering in §7. The full statement of the principle is `PLAN.md` → "The
common-case contract"; this section records only what it changes about the latency work.

**Why §7's ranking was wrong.** Every item in §7 makes the plan engine cheaper for turns that
still go through it. The traffic (41 of 46 plans single-step, 17 of 46 a lone reasoning step)
says the common turn should not go through it at all. The largest fixed cost, the plan call's
thinking (12.4 s avg for 63 output tokens), is not tunable away — think must stay ON for the
planner — so the only way to remove it from a simple turn is to not make the call.

**What replaces §7's "do first".**

- A request-side regex complexity check (`core/request_intent.py` style, zero tokens) routes
  simple turns to a bounded read-only ReAct loop: one call under a union grammar with a handful
  of read-only tool descriptions in the prompt and nothing bound (the bind_tools prefill cost in
  §1 is the reason a naive ReAct loop is slower, not faster, on the 9b). Chat: one streamed call.
  One read-only tool: two calls. Think off on both. Expected: ~8 s and ~13 s against ~20 s and
  ~25 s today, and zero thinking tokens.
- Escalation, not a judge: three tool calls, an out-of-set tool, or an error hands the
  observations to the plan engine. `/plan` and `/quick` override the check in either direction.
- The measurement before anything ships: run the regex over the traced requests and report the
  misroute rate; after it ships, model seconds and thinking tokens per turn by shape, and an
  unchanged trust benchmark.

**What survives from §7, in the plan engine, for the turns that still need it.** Prime the
judge lineage (a few lines, no behavior change). Retire the judge deterministically where its
rules cannot fire (a clean, fully-done plan that ran a web_search). Structural shaping of
`web_extract`. The single-`none` fast path in §7 is subsumed: those turns never reach the
planner now.

**Rule going forward (from the contract).** A new node, branch, or safeguard states the turn
shape it runs on and its cost there; a safeguard that cannot fire on a shape costs that shape
nothing. Rectify's branch count is held flat.
