# Tweaks — small changes to consider later

Not a plan, not a spec. A parking lot for ideas that came out of a conversation and are worth
weighing before they are worth doing. Each entry says what, why, and what would have to be true
for it to be worth the change. Promote an entry to `specs/` when it gets picked up; delete it when
it is rejected (and say why in the commit message).

---

## Sampling parameters (2026-09-05)

**Where things stand.** Temperature is already task-shaped, not global:

| Call | Temperature | Where |
|---|---|---|
| plan, judge, update_plan (structured JSON) | 0.0 → 0.3 → 0.3 on parse failure | `core/structured.py::_ATTEMPT_TEMPS` |
| tool argument generation | 0.0 → 0.5 → 0.7 on parse failure | `nodes/execute.py::_ATTEMPT_TEMPS` |
| reasoning step | same ladder as tool args | `nodes/execute.py` |
| final answer stream | 0.7 fixed | `nodes/synthesize.py` (hardcoded literal) |

Verdict from the discussion: keep 0 on the agentic/structured paths (determinism feeds replay,
`/trace why`, exact-prompt memoization, calibration; greedy stays on the JSON rails). Keep the
answer sampled (greedy on a small Qwen degenerates into loops; Qwen's cards recommend ~0.6–0.7
for open-ended text). Do NOT expose raw `temperature`/`top_p`/`top_k`/`repeat_penalty` to users.

### 1. Move temperature into the task table
- **What:** add a `temps: tuple[float, ...]` field to `core/serving.Task` and read it from
  `structured()` / `execute` / `synthesize` instead of the two module-level `_ATTEMPT_TEMPS`
  tuples and the synthesizer's literal `0.7`.
- **Why:** `serving.py` already owns `num_predict` and `think` per task; temperature is the third
  per-task decision and currently lives in three places. One table, one place to read the policy.
- **Worth it when:** touching any of those three call sites anyway. Pure refactor; tests in
  `tests/test_engine.py` that pin the ladder would move to `tests/test_serving.py`.

### 2. One intent-level knob on the synthesizer only
- **What:** `runtime.answer_style: precise | balanced | creative` in `config.default.yaml`,
  mapping to answer-task temperature 0.2 / 0.7 / 1.0. Surface via `/config`. Nothing else
  becomes tunable.
- **Why:** the only call where a user has a legitimate preference. A single raw temperature would
  either break the planner or flatten the answer; five per-tier numbers nobody tunes correctly.
  Temperature is a hidden input to the trust guarantees (see the 2026-09-03 planner think-off
  incident for how sensitive the 9b is to generation settings) — keep it out of user hands on
  the agentic paths. Power users who really want raw knobs can bake them into an Ollama Modelfile
  and bind a tier to that tag.
- **Worth it when:** someone actually asks for it. Depends on #1 for a clean implementation.

### 3. Measure a cooler answer temperature
- **What:** try 0.3–0.5 on the answer task for factual/summarising requests and compare against
  the benchmark's answer-quality tasks.
- **Why:** 0.7 is defensible but unmeasured. Not 0 — degenerate repetition is the failure mode.
- **Worth it when:** running `benchmark.py` for another reason; it is a one-line change to A/B.

### 4. Pin `seed` alongside temperature 0 if replay equality is a requirement
- **What:** add `"seed": <fixed>` to the options dict built in `structured._invoke_kwargs` for
  the 0.0 rung (and optionally the retry rungs, so a retry is itself reproducible).
- **Why:** Ollama randomises the seed per request; temperature 0 is greedy but not guaranteed
  bit-identical across runs when batching or kernels differ. Exact-prompt memoization
  (`docs/OPTIMIZATIONS.md`, [next]) and any "replay should match" claim quietly assume it.
- **Worth it when:** the memoization item ships, or a replay-divergence report comes in.
  Check first that llama-server honours `seed` with the multi-prompt cache enabled.
