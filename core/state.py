import operator
from collections import Counter
from typing import List, Any, Optional, Literal
from langchain.messages import HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field


# --- mid-turn steering tag ----------------------------------------------------------------
# plan_gate injects a mid-turn steering correction as a HumanMessage. When it can't merge the
# note into the trailing message it appends a STANDALONE HumanMessage carrying this prefix —
# which is NOT a turn boundary. Everything that slices the conversation by "last HumanMessage"
# (/rewind's drop_last_turn, /retry full's query lookup, agent._compact_history, the grounding
# recap) must skip steer messages via is_steer_message, or a steered turn gets mis-sliced: the
# steer note mistaken for the question, the real question compacted away.
STEER_PREFIX = "[Steering correction from the user, mid-task — adjust your approach accordingly]:"


def is_steer_message(m) -> bool:
    """True if `m` is a standalone mid-turn steering note injected by plan_gate. The merged form
    (note appended onto an existing HumanMessage's content) deliberately does NOT match — there
    the underlying message is still the real turn boundary."""
    return isinstance(m, HumanMessage) and str(m.content).startswith(STEER_PREFIX)


# --- Living plan ------------------------------------------------------------
# The plan is a first-class, mutable object in state: drafted up front by the
# `plan` node and revised in-loop by `update_plan`. It is ADVISORY, not a rigid
# DAG — the agent may deviate, but each loop step it must reconcile against it.
# It is also the transparency surface: each step's `label` is streamed to the UI
# as a PlanEvent and persisted to the trace.

StepStatus = Literal["pending", "active", "done", "skipped"]

# A step in one of these statuses is retired — the multiset walkers (unrun_planned_tools here,
# update_plan_node, approval._skip_rejected_steps) and active_step all key off the same pair.
TERMINAL_STATUSES = ("done", "skipped")


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


class ReplanVerdict(BaseModel):
    """Structured verdict from the in-loop replan/judge node (nodes/replan.py).

    When the agent finishes with no tool calls and the mechanical nudge has nothing left to
    escalate to, the judge inspects the draft answer: is it adequately grounded in what was
    actually gathered (or a legitimate general-knowledge answer), or does it assert
    current/external/specific facts that were never looked up? If the latter, it proposes a web
    search to gather the missing information so the loop can insert that step and try again."""

    grounded: bool = Field(
        description=(
            "True if the draft answer is adequately supported by the gathered tool results, or "
            "is a legitimate general-knowledge/conceptual answer that needs no external lookup. "
            "False only if it asserts current, external, or specific verifiable facts (rankings, "
            "prices, news, versions, people/products, 'best X' claims) that were stated WITHOUT "
            "being looked up this turn."
        )
    )
    search_query: Optional[str] = Field(
        default=None,
        description=(
            "When grounded is False, the web search query that would gather the missing "
            "information. Null when grounded is True."
        ),
    )
    reason: str = Field(default="", description="One short sentence explaining the verdict.")


def unrun_planned_tools(plan: List[dict], called) -> List[dict]:
    """Planned gathering steps the agent has NOT yet executed: non-terminal steps
    (not done/skipped) carrying an `intended_tool` for which no un-credited call exists.

    This is the plan/execution gap — work the planner expected but the agent skipped (the
    `gemma4:e4b` "answers without firing the planned tool" failure). Read by
    `route_after_agent` (nudge the agent back to act on it) and `synthesize_node` (when we give
    up with such a step still open, be honest that it wasn't completed rather than claiming the
    information doesn't exist).

    Matching is POSITIONAL, not set-membership: `called` is consumed as a multiset, in plan order,
    so two steps that share a tool (e.g. two web_search steps) each require their OWN call. With
    set membership a single web_search would mark both "run", silently collapsing a multi-search
    plan — this is the bug that fix keys off the *count* of calls, not just their presence.
    Already-done steps consume their tool first so a later same-tool step lines up against the
    correct remaining call (mirrors the consumption order in update_plan_node)."""
    remaining = Counter(called or [])
    pending = []
    for step in plan or []:
        tool = step.get("intended_tool")
        if step.get("status") in TERMINAL_STATUSES:
            # Already ran — account for its call so it doesn't mask a later same-tool step.
            if tool and remaining.get(tool, 0) > 0:
                remaining[tool] -= 1
            continue
        if not tool:
            continue
        if remaining.get(tool, 0) > 0:
            remaining[tool] -= 1  # an un-credited call covers this step; it effectively ran
        else:
            pending.append(step)
    return pending


def active_step(plan: List[dict]) -> Optional[dict]:
    """The current step to work: the first non-terminal step (not done/skipped). Drives lockstep
    execution (`agent_node` focuses the model on this one) and is surfaced in the plan-review
    interrupt so the user can see where execution is. None when the plan is complete/empty."""
    for step in plan or []:
        if step.get("status") not in TERMINAL_STATUSES:
            return step
    return None


def summarize_gates(gate_events) -> dict:
    """The headless --json "gates" field, derived from the `gate_events` accumulator:
    {"prompted": <total calls that faced the human>, "denied": [tool names with approved=False]}.
    Pure over plain dicts so the CLI contract is testable without driving a graph. Tolerates
    None/garbage rows (a record surface must degrade to zero, never crash the result dump)."""
    prompted = 0
    denied: List[str] = []
    for ev in gate_events or []:
        if not isinstance(ev, dict):
            continue
        for call in ev.get("calls") or []:
            if not isinstance(call, dict):
                continue
            prompted += 1
            if not call.get("approved"):
                denied.append(str(call.get("name") or "?"))
    return {"prompted": prompted, "denied": denied}


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

    # Convenience handle for the current turn's user query.
    current_query: str

    # Grounding string built by the `ground` node (document/workspace manifests
    # + persistent memory/profiles). Sole writer: grounding_node; downstream
    # nodes read but never mutate it.
    context: str

    # Per-turn @file attachments: the contents of files the user referenced with `@path` in their
    # message, pre-formatted as a context section by `mentions.expand` and appended to `context` by
    # the grounding node — so the planner/agent/synthesize (which read `context`, not raw `messages`)
    # all see the file inline. Empty when the message had no resolvable @mentions. Reset per turn.
    attachments: str

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
    # 0 per turn. See state.unrun_planned_tools + nodes/agent.py.
    agent_nudges: int

    # In-loop replan counter: how many times this turn the judge/replan node inserted a gathering
    # step because the agent's draft answer was ungrounded (asserted external facts it never looked
    # up). Bounded by REPLAN_BUDGET in route_after_agent so a stubborn model can't loop. Reset to 0
    # per turn. See nodes/replan.py.
    replans: int

    # Plan-review interrupt (see nodes/plan_gate.py). `pause_requested` is the IN-GRAPH
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

    # Structured human-gate record: exactly ONE dict appended per approval prompt that actually
    # FACED the human (auto-approved batches record nothing):
    #   {"calls": [{"id", "name", "approved"}], "decision": "approved"|"rejected"|"partial",
    #    "quarantine": bool (was this the injection-escalation gate?),
    #    "taint": [{"call_id", "source_tool"}], "step": active-step label or None}
    # Plain dicts only (gotcha #4). A human decision is the ONE run fact that can never be
    # recomputed after the fact, so this same record feeds the headless --json "gates" field, the
    # Glass Box's gate_summary, and the signed export's answer attestation — keep the shape
    # minimal (see nodes/approval.gate_event). Same append-reducer; reset per turn.
    gate_events: Annotated[List[dict], operator.add]

    # Tokens/second from the most recent LLM call (agent or synthesizer). Overwritten
    # each LLM step; reset to 0.0 at the start of each turn. Only populated for Ollama
    # models (response_metadata carries eval_count + eval_duration); other providers
    # leave it 0.0.
    tok_per_sec: float

    # Prompt tokens ingested by the most recent LLM call — how full the context window is right
    # now. Overwritten each LLM step (agent/synthesize); the UI gauges it against the model's
    # context window. Persists across turns (the context only grows) rather than resetting.
    context_tokens: int
