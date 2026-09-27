"""
The serving layer's per-TASK decoding decisions (transplanted from the engine isolate, 2026-08-15
— the measured slice: think-per-task, output bounds, the repetition retry penalty. The per-family
temperature ladders, token budget and prompt packing stayed in the isolate).

Every LLM call names what it is generating — a plan, a judgment, tool arguments, reasoning prose,
the answer, a corrective — and this module decides two things the call site used to leave to the
daemon's defaults:

  - `think`: EXPLICIT per task, never the model's default (every local model Saturn targets
    defaults to thinking ON) — OFF for every task since 2026-09-08. The planner was the one
    task that thought: its rationale is where it decides "answer this directly or ask", and
    without any rationale the 9b turned every open creative request into a lone `ask_user` or
    `write_file` stub (measured 2026-09-03 by replaying run 40: 4/4 draws `[ask_user]` with
    think off, 4/4 `[none: write a story]` with it on). Free thinking cost 5–19 s per plan call
    and could not be budgeted under Ollama. The rationale now lives INSIDE the plan grammar
    (structured chain-of-thought: `_PlanOut.rationale`, first field, maxLength-bounded), so the
    model still reasons before it commits — measured 2026-09-08 on 15 requests: the story
    request draws `none` 4/4, 12/15 plans identical to the thinking planner's, the rest
    defensible, at 1.7–3 s warm against 9–17 s. Judges, tool arguments, reasoning prose and the
    answer were already without (measured: think-off fixed `absence` 66 → 100 %,
    `no_capability` 2/5 → 5/5). The flag only changes the assistant opener at the prompt's
    tail, so it costs nothing at the prefix cache (see below) — but a PRIME is still sent think
    ON (core/prime.py), or the empty think block lands past the boundary.
  - `num_predict`: a circuit breaker, not a budget — every cap is well above what a healthy
    generation of that task uses; it exists so a whitespace loop under a JSON grammar or a small
    model that starts repeating lands as a truncated generation instead of a full context window.
    The one exception to "well above healthy" was the tool-argument cap for write_file/edit_file,
    whose healthy generation IS the file — those run under the `tool_payload` task (the execute
    node picks it per tool), and a draw the daemon cuts at the bound (`done_reason=length`) is
    refused there with the limit named rather than re-rolled.

Also home of the repetition RETRY penalty: applied to the next rung only after a degenerate draw
(`textutil.looks_repetitive`), never globally — a repeat penalty on every generation would corrupt
the outputs that legitimately repeat (a JSON schema's punctuation, an `old_string` that must
reproduce a file's text verbatim, a path named twice).

The prefix cache (measured 2026-09-03/04, Ollama 0.33 / qwen3.5, Apple M4 Pro at ~400 prompt
tokens/s; the earlier reading of this paragraph was wrong on three counts): llama-server keeps
MANY past prompts in host RAM (`--cache-ram`, ~200 MiB each) and picks the one sharing the
longest prefix with the new request; qwen35 is a hybrid/recurrent architecture, so it cannot
reuse a partial prefix — it can only RESTORE a saved context checkpoint that lies inside the
common prefix and reprocess from there. Checkpoints land at N-1024 (`n_batch` is 1024, not
512) and at N-4 of every processed prompt, and at the point a prompt restored from. So a
prompt whose predecessor differs only in its last 1024 tokens reprocesses ~1024 tokens
(2.7 s); one that EXTENDS a cached prompt at a message boundary restores its N-4 and
reprocesses only the extension (0.2 s); anything else reprocesses whole. Hence: every node's
prompt is [system][user: the STABLE grounding half][per-turn messages…] with the changing
material last and append-only (core/plan_context: results are capped in landing order and
never re-truncated); an idle prime re-sends each lineage's first two messages between turns
(core/prime.py) so the boundary checkpoint exists; the think flag only changes the assistant
opener at the prompt's tail (NOT a separate lineage — but a prime must be sent think ON, or
the empty think block lands after the boundary); and tool arguments are generated under a
`format` grammar, never a native bind, because the chat template renders bound tools INTO the
system message and every step then re-prefilled whole (measured: 8k tokens, 20 s). The same
`num_ctx` and runner options ride every request (load options — a mismatch reloads the model
and drops the whole cache).

Deliberately NOT here: per-task `num_ctx` — Ollama keys the loaded runner on the context size, so
alternating it between tasks would reload the model between nodes of one turn (config.num_ctx_for
stays THE one window per model). Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One shape of generation: `strict` output is externally constrained (JSON under a grammar,
    tool arguments against a schema); `num_predict` is its output bound; `think` whether a
    rationale is requested."""
    name: str
    strict: bool
    num_predict: int
    think: bool


TASKS: dict = {
    # think OFF since 2026-09-08 — the rationale moved INTO the grammar (core/structured
    # `_PlanOut.rationale`, first field, maxLength-bounded): the model still states what the
    # request needs before it commits to steps, which is what stopped the lone `ask_user` /
    # `write_file` stub for "write me a story" (4/4 stubs with think off and no rationale; 4/4
    # `none` steps with the bounded rationale; 4/4 with free thinking at 3–17 s per call). See the
    # module docstring and docs/OPTIMIZATIONS.md §9.
    "plan": Task("plan", strict=True, num_predict=1536, think=False),
    # 1024, not 512: a verbose-but-healthy verdict's `reasoning` field hit 512 exactly and the
    # JSON never closed (run 16, 2026-09-02); the retry only parsed because it came out shorter.
    "judge": Task("judge", strict=True, num_predict=1024, think=False),
    # The answer task survives for core/calibration.py's confidence sampling.
    "answer": Task("answer", strict=False, num_predict=1536, think=False),
    # The loop's one call (nodes/agent.py, 2026-09-27): prose OR a tool call, so the bound must
    # fit a write_file payload (4096 tokens is ~12-16 KB of text — a circuit breaker, not a
    # budget); think off — the cheap rationale is the model's own pre-call text.
    "agent": Task("agent", strict=False, num_predict=4096, think=False),
}

# The default task per model ROLE, for call sites that don't name one (the structured layer's
# planner/judge calls; the tool caller's argument generation; the synthesizer's stream).
_ROLE_TASK = {
    "planner": "plan",
    "judge": "judge",
    "tool_caller": "agent",
    "synthesizer": "answer",
}

# The retry-only repeat penalty (see the module docstring).
REPEAT_PENALTY = 1.15
REPEAT_LAST_N = 128


def task_of(name: str) -> Task:
    """The task record for `name`, falling back to the strictest safe shape for an unknown one."""
    return TASKS.get(name) or Task(str(name), strict=True, num_predict=512, think=False)


def task_for_role(role: str) -> "str | None":
    """The default task a role generates, or None (the utility role's calls name no task and keep
    the daemon's defaults)."""
    return _ROLE_TASK.get(role)


def thinks(task: str) -> bool:
    """Whether this task asks the daemon for a rationale — always an explicit boolean."""
    return task_of(task).think


def num_predict(task: str) -> int:
    """The output-token bound for a task."""
    return task_of(task).num_predict


def repetition_options() -> dict:
    """The options added to a retry rung after a degenerate draw."""
    return {"repeat_penalty": REPEAT_PENALTY, "repeat_last_n": REPEAT_LAST_N}
