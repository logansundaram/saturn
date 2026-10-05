import operator
from typing import List, Any, Optional
from langchain.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# --- mid-turn steering tag ----------------------------------------------------------------
# A mid-turn steering correction that can't merge into the trailing message is appended as a
# STANDALONE HumanMessage carrying this prefix — which is NOT a turn boundary. Everything that
# slices the conversation by "last HumanMessage" must test boundaries via is_turn_start below —
# never a hand-rolled isinstance check — or a steered turn gets mis-sliced: the steer note
# mistaken for the question, the real question compacted away.
STEER_PREFIX = "[Steering correction from the user, mid-task — adjust your approach accordingly]:"


def is_steer_message(m) -> bool:
    """True if `m` is a standalone mid-turn steering note injected by the agent node. The merged form
    (note appended onto an existing HumanMessage's content) deliberately does NOT match — there
    the underlying message is still the real turn boundary."""
    return isinstance(m, HumanMessage) and str(m.content).startswith(STEER_PREFIX)


def is_turn_start(m) -> bool:
    """True when `m` is a HumanMessage that STARTS a real turn — i.e. not a standalone mid-turn
    steer note (that belongs to the turn it corrected) and not a compaction summary (carried
    history, not a question). THE turn-boundary predicate every conversation slicer keys off."""
    # Lazy: keeps core.state import-light. No cycle — compaction imports this module at top,
    # but state itself only reaches for compaction when the predicate is actually called.
    from core.compaction import is_summary

    return isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m)


def turn_start(messages: list) -> int:
    """Index of the current turn's request; len(messages) when there is none."""
    for i in range(len(messages) - 1, -1, -1):
        if is_turn_start(messages[i]):
            return i
    return len(messages)


def this_turn(messages: list) -> list:
    """The current turn's messages, from its request onward ([] when there is none)."""
    return list(messages[turn_start(messages):])


# --- The plan: the model's checklist -------------------------------------------------------
# The `plan` tool (tools/planning.py) writes it; nodes/tools.py maps a successful call onto
# `state["plan"]`. It is intent, not record — the tool rounds that actually ran are the
# ToolMessages.
#
# Step shape (plain dicts — gotcha #4: the checkpointer serializer never round-trips a custom
# type):
#   {step_id, label, status, result}
#
#   status   "pending" / "done" — all the `plan` tool writes.
#   result   "done" for a completed item, None otherwise — `current_step` (the first item with
#            `result is None`) is the gate's step context.
#
# A pre-v2 record also carries `intended_tool` / `needs_resolution` and may carry a `skipped`
# status; the renderers read those with .get() so `/trace` and `--replay` still draw it.

# A step in one of these statuses is retired for DISPLAY purposes; execution-wise the pointer
# is `result is None` (a retired step always carries a result).
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
    # Human/AI/Tool messages. Tool calls AND their ToolMessage observations land here so
    # cross-turn follow-ups ("open the second result") survive _compact_history's retained
    # scratchpad, and so the approval/tools nodes can hand a call across the interrupt boundary.
    messages: Annotated[List[Any], add_messages]

    current_query: str

    # Grounding built by the `ground` node (its sole writer). `context_stable` is byte-identical
    # across turns while nothing on disk changed and rides every prompt right after the system
    # prompt (the daemon's prompt cache restores past it); `context_dynamic` is the per-turn
    # remainder; `context` is their join. An older checkpoint lacks the halves — read them
    # through `grounding_parts`.
    context: str
    context_stable: str
    context_dynamic: str

    # This turn's @file attachments, pre-formatted by `mentions.expand`; empty without any.
    attachments: str

    # Whether content from outside the trust boundary has EVER entered this conversation: an
    # attachment (set by the grounding node) or an untrusted tool that ran (set by the tools
    # node). Carried across turns (app/session._CARRY_ACROSS_TURNS) — the messages that held it
    # are compacted away after a turn or two, but an answer that restated a page is still in
    # history, so core/provenance.of reads this and auto-learn stays off for the conversation.
    # /clear resets it; a resumed session starts with it set.
    outside_seen: bool

    # The ids of this batch's `remember` calls the approval node let through because the user
    # typed every word of them (core/auto_memory.qualifies). Written by the approval node on
    # every route to `tools`, read by the tools node — which must take the gate's word, not
    # work it out again from a state the gate's own decline messages have changed.
    user_stated: List[str]

    # The skill the user ran this turn by typing /<name> (core/skills.block); empty otherwise.
    # The ground node folds it into the DYNAMIC half, so it never touches the cached prefix.
    skill: str

    # The model's checklist (see "The plan" above); intent, not record.
    plan: List[dict]

    # Agent passes this turn, bounded by runtime.max_iterations (nodes/agent.py).
    iteration: int

    # One record per agent pass this turn (core/think.entry): the kind of step, whether the
    # pass thought, and what came of the thought. Read by the rail, /think, /trace why and the
    # loop benchmark; never part of the prompt.
    think: Annotated[List[dict], operator.add]

    # This turn's own think level (`/think <request>` sets "deep"); empty = runtime.think.
    think_level: str

    # Flat, append-only mirrors of the ToolMessages, reset to [] per turn: `tools_called` names
    # every executed call; `tool_results` / `documents_retrieved` hold what the answer could
    # draw on (the Sources receipt, /trace source).
    tools_called: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Any], operator.add]
    documents_retrieved: Annotated[List[Any], operator.add]

    # One {name, args, result (preview), dur, ok} per executed call — the UI's tool-I/O tree.
    tool_events: Annotated[List[dict], operator.add]

    # Exactly ONE dict per approval prompt that actually FACED the human (auto-approved batches
    # record nothing):
    #   {"calls": [{"id", "name", "approved"}], "decision": "approved"|"rejected"|"partial",
    #    "quarantine": bool (was this the injection-escalation gate?),
    #    "step": active-step label or None}
    # A human decision is the ONE run fact that can never be recomputed, so this record feeds
    # the headless --json "gates" field and the run export — keep the shape minimal (see
    # nodes/approval.gate_event).
    gate_events: Annotated[List[dict], operator.add]

    # Tokens/second of the most recent model call (0.0 when the response doesn't report it).
    tok_per_sec: float

    # Prompt tokens of the most recent model call — the context gauge's numerator. Persists
    # across turns (the context only grows) rather than resetting.
    context_tokens: int
