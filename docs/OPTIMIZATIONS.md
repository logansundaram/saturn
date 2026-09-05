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
