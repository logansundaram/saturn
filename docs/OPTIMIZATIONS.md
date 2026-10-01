# Latency optimizations for a local agent

A working list of techniques for making Saturn faster on a local model, with what is already
in place, what was measured, and what to try next. Companion to `core/prime.py` and
`tests/test_prefix_cache.py`. Numbers are from the 9b tier on an Apple M4 Pro (48 GB) under
Ollama 0.33 (prefill ~400 tokens/s, decode ~37 tokens/s) unless stated otherwise. Most were
measured on the plan engine before the 2026-09-27 one-loop rewrite; the mechanisms they measure
carry over, and each number says where it came from.

Status markers: **[have]** shipped · **[next]** a concrete candidate · **[measure]** plausible,
needs a number before it is worth the complexity · **[closed]** decided against, with the reason.
The loop's own ranked improvements live in `docs/engine.md`; this file is the latency reference.

## 1. The prefix cache (done 2026-09-04)

The daemon (llama-server) restores a prompt only from a saved context checkpoint — 1024 and 4
tokens before an earlier prompt's end, or the point that prompt itself restored from — and
qwen3.5 (hybrid/recurrent) cannot reuse a partial prefix. So a prompt is cheap exactly when
everything that changed sits after such a checkpoint. The agent's prompt is built in that order
(`nodes/agent._llm_input`):

```
[system + bound tool schemas][user: stable grounding][history…][user: dynamic grounding + request][this turn…]
```

- **[have] Stable/dynamic grounding split** (`nodes/ground.py`). `context_stable` (the
  SATURN.md files, the knowledge-base manifest, the always-loaded memory layers) rides as its
  own message right after the system prompt; `context_dynamic` (the date line, matched memory,
  attachments) goes with the request, after the history.
- **[have] The idle prime** (`core/prime.py`, `runtime.prime`). After the startup warm-up and
  after each turn the agent's `[system][stable grounding]` prefix is re-sent through the same
  bound model with one predicted token, think ON (think off appends the empty think block after
  the boundary and pushes the N-4 checkpoint past it), so the next turn's first call resumes at
  that boundary. Measured on the plan engine: 5,083 tokens / 12.7 s → 20 tokens / 0.2 s for the
  first call; ~0.2 s of idle daemon time once warm. One lineage since the v2 loop. Headless
  (`-p`/`-q`, one turn per process) never primes: on the 2026-09-08 quick path a first call
  there prefilled its whole ~1.4k-token prompt in 2.3–2.6 s against 0.5 s primed.
- **[have] Pass-over-pass extension.** Within a turn each pass's prompt is the previous pass's
  prompt plus its call and the tool round's observations, appended at a message boundary, so it
  restores the N-4 checkpoint and prefills only what is new. Observations are clamped once when
  they land (`nodes/tools._clamp_observation`, 12,000 chars) and never re-truncated, so the
  prompt only grows at its end. The same shape measured on the quick path's two-call loop
  (2026-09-08): first call 0.5–0.8 s (≈50 tokens past the primed boundary), second 1.4–2.4 s
  (≈600 tokens of observation prefill). Not yet read off the daemon log for the loop itself.
- **[have] The tool catalog inside the cached prefix.** The plan engine put tool arguments
  under a `format` grammar because the chat template renders bound tools into the system
  message and a per-step tool set re-prefilled every step whole (8k tokens, 20 s). The loop
  binds the whole registry on every pass, byte-identical, and the prime binds it too, so the
  catalog is paid once. The cost moved: anything that changes the bound catalog moves the first
  bytes of the prompt and forces one cold prefill of everything. The cap used to be such a
  change (tools unbound: 10 s of prefill at 7.5k tokens on the 4b, measured 2026-10-01); it now
  keeps the tools bound and refuses the call instead (0.3–0.5 s), and only a model that calls
  again past the refusal pays the cold prefill. Catalog experiments (`docs/engine.md`
  item 7) fight the cache for the same reason.
- **[have] Same `num_ctx` on every request** (`core/llms.invoke_kwargs`, the warm-up, the
  prime). Ollama keys the loaded runner on the context size; a mismatch reloads the model and
  drops the whole cache. `keep_alive` is not a load option and never reloads.
- Measured A/B on the plan engine, same daemon: four-turn wall 65.3 s → 51.4 s; trust benchmark
  (13 queries) 581 s → 397 s with verdicts unchanged.

## 2. Fewer tokens through the model

- **[have] Deterministic checks, not model calls.** The loop's hygiene (unknown tool, missing
  arguments, a repeat of a declined call, a third identical call, a call past the cap) answers with an error
  ToolMessage — no gate, no model call — and the Sources and incidents trailers are appended
  mechanically. There is no judge and no answer rewrite: the model's last message is the answer.
  The plan engine's rewrite had cost ~10 s of decode per single-step turn (run 45: the step
  wrote 452 tokens, synthesize rewrote them in 414).
- **[have] Adaptive thinking** (`runtime.think`, since 2026-09-29). Every pass runs think-off
  except the one right after a tool round with an error, bounded by `runtime.think_budget`
  (4096); a thinking pass that returns nothing is rerun think-off. On the plan engine the
  planner's thinking was the largest fixed cost on every turn (12.4 s average for 63 output
  tokens). Whether thinking earns its latency per tier is `docs/engine.md` item 11.
- **[next] Structural observation shaping.** Clamp tool output by shape, not just length:
  first N rows of a CSV, matched lines with a little context for search, a diff after an edit
  instead of the whole file. The loop's versions are `docs/engine.md` item 4 (a
  relevance-aware clamp) and item 5 (a token budget for the prompt).
- **[closed] Exact-prompt memoization.** Prompts do not repeat within a session; they DO repeat
  across benchmark runs, which would stop the benchmark measuring the model.

## 3. Cheaper tokens

- **[closed] Speculative decoding.** Measured slower on this Mac for the 27b; the 9b has no
  drafter.
- **[closed] Flash attention and KV cache q8_0.** The runner log already shows `Flash Attention
  enabled` (`flash_attn = auto`); memory is not the constraint, and only one in four layers of a
  hybrid model has a KV cache.
- **[closed] The MLX runner** (measured 2026-09-08: `qwen3.5:9b-mlx`, nvfp4, Ollama 0.33.2's MLX
  engine, against the GGUF Q4_K_M with the app's exact options). Decode 38–39 tok/s on BOTH
  runners (37.9 GGUF), cold prefill ~450 vs ~417 tok/s, a primed message-boundary extension
  0.12–0.18 s on both. The 9b reads 6.6 GB per token and is memory-bandwidth-bound on this chip
  (273 GB/s × 0.9 / 6.6 GB ≈ 37 tok/s) whichever runner serves it; the published 1.4–3x MLX
  gains come from MoE models (few active parameters) and the M5's accelerators. The MLX prompt
  cache honors the prime boundary, so §1 would survive a future switch. `core/hardware.py` fits
  its decode and prefill speed model to these numbers.
- **[closed] Right-size `num_ctx`.** On a 48 GB machine the 9b at 65k occupies 8.1 GB: no speed
  to gain. `runtime.num_ctx: null` takes each model's declared window.
- **[closed] Model cascade by role.** The `utility` role was collapsed on 2026-09-30; each tier
  binds one `model`, so there is one resident model and one prefix cache to keep warm.

## 4. Fewer round trips and overlapped time

- **[have] Streaming and early rendering.** The answer streams under `── response` as it
  generates; the checklist and each tool result render on the rail as they land, so perceived
  latency stays low even when the total does not move.
- **[have] Warm-up at startup** (`app/startup.start_warm_up`). One minimal request loads the
  weights on a thread while the REPL starts, then the prime; `runtime.keep_alive: 30m` keeps
  them resident (the daemon's default unloads after five idle minutes). Measured 2026-09-02: a
  cold "hello" 37–50 s against 15 s warm.
- **[next] Concurrent tool batches.** `docs/engine.md` item 9: matters for two web fetches or
  two AppleScript readers in one pass.
- **[closed] Prime during slow tools.** The next pass already extends the previous prompt at a
  message boundary; there is nothing to prime while the tool runs.

## 5. Daemon and environment

- **Hazard (seen 2026-09-04):** with `keep_alive: 30m`, a resident qwen3.5:9b at 65k context
  can leave `/api/embed` (loading qwen3-embedding:8b) hanging indefinitely with no scheduler
  log line, while `/api/tags` and chat still answer. `ollama stop qwen3.5:9b` unblocks it.
  Anything that embeds through the live daemon (the knowledge-base ingest, a benchmark that
  plants documents) hangs in that state. Mitigations: a smaller chat window, a smaller
  embedder, or a bounded wait around the ingest.
- Any model reload (a `num_ctx` mismatch, an eviction by the embedder) drops the entire
  prompt cache; the next call then pays a cold prefill once.
- Never run two agents or benchmarks against one daemon at once: with `-np 1` they serialize
  and their prompts evict each other's cache entries.

## 6. How to measure

- The daemon's own log (`~/.ollama/logs/server.log`, verbosity 4) is the ground truth:
  `restored context checkpoint … n_past = N` vs `forcing full prompt re-processing`, and
  `prompt eval time = … / N tokens`. Wall time per call lives in `llm_calls.dur` in
  `database/db.sqlite` for traced runs.
- Replay one call live before hypothesizing (`llm_calls.input` is capped; rebuild the prompt
  with `core.messages.agent_sys_msg()` and `nodes/agent._llm_input`). Send test requests with
  the app's exact options (`num_ctx` and `num_predict` from `core/llms.invoke_kwargs`, and
  `keep_alive`, which `core/llms` sets on the model).
- Validate behavior with `python benchmark.py` and `python benchmark.py --loop` on both trees,
  not timing alone: the one regression found on 2026-09-04 (a memory correction stored as a
  duplicate) came from a tool description truncated in the new argument brief, not from the
  cache work itself.
