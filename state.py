import operator
from typing import List, Any, Optional, Literal
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field


# --- Living plan ------------------------------------------------------------
# The plan is a first-class, mutable object in state: drafted up front by the
# `plan` node and revised in-loop by `update_plan`. It is ADVISORY, not a rigid
# DAG — the agent may deviate, but each loop step it must reconcile against it.
# It is also the transparency surface: each step's `label` is streamed to the UI
# as a PlanEvent and persisted to the trace. (See SATURDAY_MVP_PLAN.md.)

StepStatus = Literal["pending", "active", "done", "skipped"]


class PlanStep(BaseModel):
    step_id: int = Field(description="1-based position of the step in the plan")
    label: str = Field(
        description="short, human-readable description shown live to the user"
    )
    status: StepStatus = Field(
        default="pending", description="current execution status of this step"
    )
    intended_tool: Optional[str] = Field(
        default=None, description="tool this step expects to call, if any"
    )


class Plan(BaseModel):
    """Structured-output wrapper so the planner LLM can emit a full plan at once."""

    steps: List[PlanStep] = Field(default_factory=list)


def unrun_planned_tools(plan: List[dict], called) -> List[dict]:
    """Planned gathering steps the agent has NOT yet executed: non-terminal steps
    (not done/skipped) carrying an `intended_tool` that isn't in `called`.

    This is the plan/execution gap — work the planner expected but the agent skipped (the
    `gemma4:e4b` "answers without firing the planned tool" failure). Read by
    `route_after_agent` (nudge the agent back to act on it) and `synthesize_node` (when we give
    up with such a step still open, be honest that it wasn't completed rather than claiming the
    information doesn't exist)."""
    done = set(called or [])
    pending = []
    for step in plan or []:
        if step.get("status") in ("done", "skipped"):
            continue
        tool = step.get("intended_tool")
        if tool and tool not in done:
            pending.append(step)
    return pending


def active_step(plan: List[dict]) -> Optional[dict]:
    """The current step to work: the first non-terminal step (not done/skipped). Drives lockstep
    execution (`agent_node` focuses the model on this one) and is surfaced in the plan-review
    interrupt so the user can see where execution is. None when the plan is complete/empty."""
    for step in plan or []:
        if step.get("status") not in ("done", "skipped"):
            return step
    return None


def steps_to_dicts(steps: List[PlanStep]) -> List[dict]:
    """Convert planner structured-output PlanSteps into the plain dicts stored in state.

    The plan lives in state as a list of dicts (not Pydantic objects) so the SqliteSaver
    checkpointer's serializer never has to round-trip a custom type. Pydantic PlanStep/Plan
    are used only at the planner boundary."""
    return [
        {
            "step_id": i,
            "label": s.label,
            "status": "pending",
            "intended_tool": s.intended_tool,
        }
        for i, s in enumerate(steps, start=1)
    ]


# --- Agent state ------------------------------------------------------------
class AgentState(TypedDict):
    # ReAct scratchpad. Human/AI/Tool messages. Tool calls AND their ToolMessage
    # observations live here, so the model sees results and decides the next
    # action. This is the loop's working memory.
    messages: Annotated[List[Any], add_messages]

    # Convenience handles for the current turn.
    current_query: str
    current_response: str

    # Grounding string built by the `ground` node (document/workspace manifests
    # + persistent memory/profiles). Sole writer: grounding_node; downstream
    # nodes read but never mutate it.
    context: str

    # Living plan (see above), stored as plain dicts: {step_id, label, status, intended_tool}.
    # Overwritten wholesale by `plan` and `update_plan`. Kept as dicts (not PlanStep objects)
    # so the checkpointer serializes it without custom-type warnings.
    plan: List[dict]

    # Loop control / guardrails for the ReAct loop. Incremented each agent pass;
    # bounded by a max-iteration cap in config to prevent runaway loops.
    iteration: int

    # Plan-aware nudge counter: how many times this turn the agent finished (no tool calls)
    # while the plan still had an un-run gathering step, so we sent it back to act. Bounded by a
    # small budget in route_after_agent so a model that stubbornly refuses can't loop. Reset to
    # 0 per turn. See state.unrun_planned_tools + node_registry/agent.py.
    agent_nudges: int

    # Outer verify/repair loop.
    verified: bool            # verifier's verdict on the synthesized response
    verifier_feedback: str    # actionable critique fed back to the agent on repair

    # Plan-review interrupt (see node_registry/plan_gate.py). `pause_requested` is the IN-GRAPH
    # trigger seam: any node/tool (today none; later an LLM-initiated "review the plan" step) can
    # set it True to make the next plan_gate pause. External/async pauses (keyboard, /plan
    # pause|review) come through the interrupts.PauseController instead — the gate checks both.
    # `pause_reason` is the human-readable why shown at the prompt. `aborted` is set by the gate
    # when the user abandons the turn at the review prompt, routing the loop to synthesize. All
    # three reset per turn.
    pause_requested: bool
    pause_reason: str
    aborted: bool

    # Trace / transparency accumulators. The model consumes observations via
    # ToolMessages in `messages`; these mirror them as a flat, append-only record
    # for the trace store, citations, and the benchmark harness. The `operator.add`
    # reducer appends across loop iterations (reset to [] per turn before invoke).
    tools_called: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Any], operator.add]
    documents_retrieved: Annotated[List[Any], operator.add]

    # Per-call structured trace records emitted by tool_node, one dict per executed call:
    # {name, args, result (one-line preview), dur (seconds), ok}. Drives the UI's tool-I/O
    # tree (args + result preview + per-tool timing). Same append-reducer as the accumulators
    # above; reset to [] per turn.
    tool_events: Annotated[List[dict], operator.add]

    # Tokens/second from the most recent LLM call (agent or synthesizer). Overwritten
    # each LLM step; reset to 0.0 at the start of each turn. Only populated for Ollama
    # models (response_metadata carries eval_count + eval_duration); other providers
    # leave it 0.0.
    tok_per_sec: float

    # Prompt tokens ingested by the most recent LLM call — how full the context window is right
    # now. Overwritten each LLM step (agent/synthesize); the UI gauges it against the model's
    # context window. Persists across turns (the context only grows) rather than resetting.
    context_tokens: int
