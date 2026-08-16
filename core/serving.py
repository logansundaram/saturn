"""
The serving layer's per-TASK decoding decisions (transplanted from the engine isolate, 2026-08-15
— the measured slice: think-per-task, output bounds, the repetition retry penalty. The per-family
temperature ladders, token budget and prompt packing stayed in the isolate).

Every LLM call names what it is generating — a plan, a judgment, tool arguments, reasoning prose,
the answer, a corrective — and this module decides two things the call site used to leave to the
daemon's defaults:

  - `think`: EXPLICIT per task, never the model's default (every local model Saturn targets
    defaults to thinking ON). The planner is the ONE task that keeps its rationale — decomposing
    a request is the job a rationale helps, and on the floor tier the plan collapses to a
    one-step stub without it (measured); judges, tool arguments, reasoning prose and the answer
    run without (measured: think-off fixed `absence` 66 → 100 %, `no_capability` 2/5 → 5/5, and
    a rationale generated FIRST can eat a `num_predict` cap and return EMPTY content).
  - `num_predict`: a circuit breaker, not a budget — every cap is well above what a healthy
    generation of that task uses; it exists so a whitespace loop under a JSON grammar or a small
    model that starts repeating lands as a truncated generation instead of a full context window.

Also home of the repetition RETRY penalty: applied to the next rung only after a degenerate draw
(`textutil.looks_repetitive`), never globally — a repeat penalty on every generation would corrupt
the outputs that legitimately repeat (a JSON schema's punctuation, an `old_string` that must
reproduce a file's text verbatim, a path named twice).

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
    "plan": Task("plan", strict=True, num_predict=1536, think=True),
    "judge": Task("judge", strict=True, num_predict=512, think=False),
    "tool_args": Task("tool_args", strict=True, num_predict=512, think=False),
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
