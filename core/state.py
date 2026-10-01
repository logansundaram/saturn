import operator
from typing import List, Any, Optional
from langchain.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# --- mid-turn steering tag ----------------------------------------------------------------
# The agent node injects a mid-turn steering correction as a HumanMessage. When it can't merge the
# note into the trailing message it appends a STANDALONE HumanMessage carrying this prefix —
# which is NOT a turn boundary. Everything that slices the conversation by "last HumanMessage"
# (app/session._compact_history, compaction.summarize_messages, nodes/agent._turn_start, the
# approval node's provenance read) must test boundaries via is_turn_start below — never a
# hand-rolled isinstance check — or a steered turn gets mis-sliced: the steer note mistaken for
# the question, the real question compacted away.
STEER_PREFIX = "[Steering correction from the user, mid-task — adjust your approach accordingly]:"


def is_steer_message(m) -> bool:
    """True if `m` is a standalone mid-turn steering note injected by the agent node. The merged form
    (note appended onto an existing HumanMessage's content) deliberately does NOT match — there
    the underlying message is still the real turn boundary."""
    return isinstance(m, HumanMessage) and str(m.content).startswith(STEER_PREFIX)


def is_turn_start(m) -> bool:
    """True when `m` is a HumanMessage that STARTS a real turn — i.e. not a standalone mid-turn
    steer note (that belongs to the turn it corrected) and not a compaction summary (carried
    history, not a question).

    THE turn-boundary predicate. Every conversation slicer (app/session._compact_history,
    compaction.summarize_messages, nodes/agent._turn_start) keys off this one function — the
    three-clause filter used to be re-spelled at each site, and the fifth copy drifted
    (summarize_messages missed the steer check, compacting a steered turn's real question
    away)."""
    # Lazy: keeps core.state import-light. No cycle — compaction imports this module at top,
    # but state itself only reaches for compaction when the predicate is actually called.
    from core.compaction import is_summary

    return isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m)


# --- The plan: the model's checklist -------------------------------------------------------
# The `plan` tool (tools/planning.py) writes it; nodes/tools.py maps a successful call onto
# `state["plan"]`. It is intent, not record — the tool rounds that actually ran are the
# ToolMessages. A step with `result is None` is not done yet; `current_step` (the first such
# step) is the gate's step context.
#
# Step shape (plain dicts — gotcha #4: the checkpointer serializer never round-trips a custom
# type):
#   {step_id, label, status, result}
#
#   status   "pending" / "done" — all the `plan` tool writes (tools/planning.py). The v1
#            engine's incident statuses left with it (2026-09-30).
#   result   "done" for a completed item, None otherwise — `current_step` (the first item with
#            `result is None`) is the gate's step context.
#
# A record written before v2 also carries `intended_tool` / `needs_resolution` and may carry a
# `skipped` status; the renderers read those with .get() so `/trace` and `--replay` still draw
# an old run, and nothing writes them any more.

# A step in one of these statuses is retired for DISPLAY purposes; execution-wise the pointer
# is `result is None` (a retired step always carries a result). `skipped` is only ever read off
# a pre-v2 record.
TERMINAL_STATUSES = ("done", "skipped")



def issuing_message(messages) -> "tuple[Any, set]":
    """The message that issued the pending tool batch, and the ids of its calls already answered:
    walk back over the trailing ToolMessages (the agent's hygiene answers, the gate's declines,
    executed results) to the first message that is not one. `(None, answered)` for an empty or
    all-ToolMessage list. The one walk-back the agent's router, the gate and the tools node share."""
    answered: set = set()
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            answered.add(m.tool_call_id)
            continue
        return m, answered
    return None, answered


def current_step(plan: List[dict]) -> Optional[dict]:
    """The first step whose `result` is None (not done yet); None when the plan is complete or
    empty. The approval gate's payload and its gate_events `step` label read this."""
    for step in plan or []:
        if step.get("result") is None:
            return step
    return None


def grounding_parts(state) -> "tuple[str, str]":
    """The grounding context as (stable, per-turn) halves — the grounding node's split
    (`context_stable` / `context_dynamic`). A state carrying only the joined `context` (an
    older checkpoint, a test fixture) is all-stable."""
    stable = state.get("context_stable")
    if stable is None and state.get("context_dynamic") is None:
        return str(state.get("context") or "").strip(), ""
    return str(stable or "").strip(), str(state.get("context_dynamic") or "").strip()


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


# --- Agent state ------------------------------------------------------------
class AgentState(TypedDict):
    # Conversation record. Human/AI/Tool messages. The agent's prompt is built from it (earlier
    # turns as history, this turn's tool rounds verbatim); tool calls AND their ToolMessage
    # observations land here so cross-turn follow-ups ("open the second result") keep working
    # through _compact_history's retained scratchpad, and so the approval/tools nodes can hand a
    # call across the interrupt boundary.
    messages: Annotated[List[Any], add_messages]

    # Convenience handle for the current turn's user query.
    current_query: str

    # Grounding string built by the `ground` node (document/workspace manifests
    # + the persistent-memory selection). Sole writer: grounding_node; downstream
    # nodes read but never mutate it.
    context: str
    # The two halves of `context` (nodes/ground.py): `context_stable` is byte-identical across
    # turns while nothing on disk changed and rides every prompt as its own message right after
    # the system prompt (the daemon's prompt cache restores past it); `context_dynamic` is the
    # per-turn remainder. `context` is their join. Absent on an older checkpoint — readers go
    # through `grounding_parts` below, which treats the whole `context` as stable then.
    context_stable: str
    context_dynamic: str

    # Per-turn @file attachments: the contents of files the user referenced with `@path` in their
    # message, pre-formatted as a context section by `mentions.expand` and appended to `context` by
    # the grounding node — so the agent sees the file inline. Empty when the message had no
    # resolvable @mentions.
    attachments: str

    # The model's checklist (tools/planning.py — the `plan` tool, mapped here by nodes/tools.py),
    # stored as plain dicts {step_id, label, status, result}.
    # Rendered by the rail and read by the gate's step context; intent, not record.
    plan: List[dict]

    # Agent passes this turn, bounded by runtime.max_iterations (nodes/agent.py): from that
    # pass on no tool call runs and the turn ends in an answer. One increment per pass.
    iteration: int

    # Trace / transparency accumulators. The agent reads observations from the ToolMessages;
    # these mirror them as a flat, append-only record: `tools_called` names every executed
    # call; `tool_results` / `documents_retrieved` hold what the answer could draw on — the
    # calls that completed and gathered something (the Sources receipt, /trace source). The `operator.add` reducer appends across loop iterations
    # (reset to [] per turn before invoke).
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
    #    "step": active-step label or None}
    # Plain dicts only (gotcha #4). A human decision is the ONE run fact that can never be
    # recomputed after the fact, so this same record feeds the headless --json "gates" field and
    # the run export — keep the shape minimal (see nodes/approval.gate_event). Same
    # append-reducer; reset per turn.
    gate_events: Annotated[List[dict], operator.add]

    # Tokens/second from the most recent LLM call. Overwritten each agent pass; reset to 0.0
    # at the start of each turn. Read from Ollama's response_metadata (eval_count +
    # eval_duration); a response without them leaves it 0.0.
    tok_per_sec: float

    # Prompt tokens ingested by the most recent LLM call — how full the context window is right
    # now. Overwritten each agent pass; the UI gauges it against the model's context window.
    # Persists across turns (the context only grows) rather than resetting.
    context_tokens: int
