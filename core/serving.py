"""
The serving layer's per-TASK decoding decisions (transplanted from the engine isolate, 2026-08-15
— the measured slice: think-per-task, output bounds, the repetition retry penalty. The per-family
temperature ladders, token budget and prompt packing stayed in the isolate).

Every LLM call names what it is generating — a plan, a judgment, tool arguments, reasoning prose,
the answer, a corrective — and this module decides two things the call site used to leave to the
daemon's defaults:

  - `think`: EXPLICIT per task, never the model's default (every local model Saturn targets
    defaults to thinking ON) — ON for the planner, OFF for everything else. The planner's
    rationale is where it decides "answer this directly or ask": without it the 9b turned every
    open creative request into a lone `ask_user` step (measured 2026-09-03 by replaying run 40:
    4/4 draws `[ask_user]` with think off, 4/4 `[none: write a story]` with it on), which the ask
    gate skips and rectify then cancels. The flag was off for a few hours that day for latency
    (~5 s per plan call on the 9b); the cost is now steered instead — the plan prompt tells the
    model to think briefly on a simple request (measured: a one-tool request's rationale
    108 → 49 words, 5.9 s → 3.1 s warm; a five-step request thinks longer, the plans unchanged).
    Judges, tool arguments, reasoning prose and the answer stay without (measured: think-off
    fixed `absence` 66 → 100 %, `no_capability` 2/5 → 5/5). The planner's think template differs
    from the other nodes', so its prompt prefix is its own cache entry (see below).
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

The prefix cache (measured 2026-09-03, Ollama 0.33 / qwen3.5, Apple M4 Pro at ~400 prompt
tokens/s): the qwen35 architecture runs ONE runner slot ("does not currently support parallel
requests" — OLLAMA_NUM_PARALLEL is ignored), and llama.cpp keeps a pool of context checkpoints,
two per processed prompt: one `num_batch` (512) tokens before its end and one at the end. A
later prompt skips prefill up to a checkpoint only when it is byte-identical up to that position,
so a node's call reuses its predecessor's cache exactly when everything that changed sits in
the prompt's LAST 512 tokens. Hence: every node keeps a byte-stable system prompt first and its
per-call material last (the plan prompt: catalog, then grounding — stable sections first, the
recap and request at the end); one `think` setting per node's prompt family (the flag changes
the template's tokens, so the planner's think-on prompts and the other nodes' think-off prompts
are two cache lineages); and the same `num_batch` on every request (a load option — it would reload
the model, and a larger batch moves the checkpoint earlier, which only helps a node whose
per-call tail is longer than 512). Measured on the 9b: the plan call 9 s cold → 1.4 s on a hit.

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
    # think ON — the one task that keeps its rationale. Turned off for latency on 2026-09-03 and
    # turned back on the same day: without it the 9b plans a lone `ask_user` for "write me a
    # story" (4/4 draws; 4/4 correct with it on). The latency is steered in the plan prompt
    # instead ("think only as much as the request needs"), see the module docstring. A rationale
    # that runs to the cap returns EMPTY content, which structured() retries at the next rung.
    "plan": Task("plan", strict=True, num_predict=1536, think=True),
    # 1024, not 512: a verbose-but-healthy verdict's `reasoning` field hit 512 exactly and the
    # JSON never closed (run 16, 2026-09-02); the retry only parsed because it came out shorter.
    "judge": Task("judge", strict=True, num_predict=1024, think=False),
    "tool_args": Task("tool_args", strict=True, num_predict=512, think=False),
    # The payload-carrying tools (write_file, edit_file): the file's CONTENT rides inside the
    # arguments, so the bound is the size of a file, not of an argument list. Still a circuit
    # breaker — 4096 tokens is ~12-16 KB of text; a longer write is refused honestly by the
    # execute node's truncation branch instead of looping (measured 2026-09-02: every write of a
    # story/idea list cut at exactly 512 with done_reason=length and no call parsed).
    "tool_payload": Task("tool_payload", strict=True, num_predict=4096, think=False),
    "reasoning": Task("reasoning", strict=False, num_predict=1024, think=False),
    "answer": Task("answer", strict=False, num_predict=1536, think=False),
    "correction": Task("correction", strict=False, num_predict=1536, think=False),
}

# The default task per model ROLE, for call sites that don't name one (the structured layer's
# planner/judge calls; the tool caller's argument generation; the synthesizer's stream).
_ROLE_TASK = {
    "planner": "plan",
    "judge": "judge",
    "tool_caller": "tool_args",
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
