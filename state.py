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

    # Outer verify/repair loop.
    verified: bool            # verifier's verdict on the synthesized response
    verifier_feedback: str    # actionable critique fed back to the agent on repair

    # Trace / transparency accumulators. The model consumes observations via
    # ToolMessages in `messages`; these mirror them as a flat, append-only record
    # for the trace store, citations, and the benchmark harness. The `operator.add`
    # reducer appends across loop iterations (reset to [] per turn before invoke).
    tools_called: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Any], operator.add]
    documents_retrieved: Annotated[List[Any], operator.add]

    # Tokens/second from the most recent LLM call (agent or synthesizer). Overwritten
    # each LLM step; reset to 0.0 at the start of each turn. Only populated for Ollama
    # models (response_metadata carries eval_count + eval_duration); other providers
    # leave it 0.0.
    tok_per_sec: float

    # Prompt tokens ingested by the most recent LLM call — how full the context window is right
    # now. Overwritten each LLM step (agent/synthesize); the TUI gauges it against the model's
    # num_ctx. Persists across turns (the context only grows) rather than resetting per turn.
    context_tokens: int
