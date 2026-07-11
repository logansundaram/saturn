"""Cross-turn conversation state.

The per-turn state shape (`_initial_state`), the fresh-turn reset (`_fresh_turn` — append the
new query, re-arm per-turn machinery, zero the accumulators), and the two history compactions:
the mechanical `_compact_history` that runs every turn, and the heavier LLM-summarizing
`_maybe_autocompact` that fires only past `runtime.compact_threshold`.
"""

from langchain.messages import HumanMessage, AIMessage

from config import get_config
from core.state import AgentState
from tui import ui


def _compact_history(messages: list, keep_recent_turns: int = 1) -> list:
    """Collapse OLDER completed turns to their conversational essence (user questions + final
    answers), but keep the ReAct scratchpad — tool-call AIMessages and their ToolMessages — of
    the most recent `keep_recent_turns` turns verbatim.

    Why a window instead of stripping everything: the scratchpad of the turn that just finished
    is exactly what the user's *next* message refers back to — "open the second result", "what
    did that file say", "multiply that by two". Dropping it on every boundary (the old
    behaviour) is what made real multi-turn use brittle: the follow-up's referent had silently
    vanished, so the model re-ran a search (getting different results) or fabricated. One turn
    of live scratchpad covers the overwhelming majority of those references.

    The original concerns still hold for OLD turns, which is why they're still compacted:
    carrying many turns of scratchpad makes the model treat a long-finished tool call as "already
    done" (reusing stale results instead of re-running a planned gather), bloats context with
    heavy tool outputs, and desyncs the model's view (`messages`) from the per-turn trace
    accumulators (`tools_called`/`tool_results`/`documents_retrieved`, reset each turn — their
    live consumers are the benchmark's grounding/gate-coverage grading, headless `--json`'s
    `tools` field, `/trace state`, and synthesize's per-turn source numbering).

    A turn starts at a REAL user HumanMessage — not a standalone mid-turn steer note (that
    belongs to the turn it corrected; treating it as a boundary would compact away the very
    scratchpad this function promises to keep) and not a compaction summary (carried history).
    Everything from the boundary onward is kept as-is (the scratchpad is intact, so no orphaned
    tool calls); everything before it is reduced to Human + non-empty final-AI messages (also
    orphan-free). Run only at the turn boundary.

    `keep_recent_turns=0` reproduces the old strip-everything behaviour."""
    from core.state import is_turn_start

    human_idxs = [i for i, m in enumerate(messages) if is_turn_start(m)]
    if keep_recent_turns > 0 and human_idxs:
        # Boundary = start of the Nth-from-last turn (clamped to the first turn).
        boundary = human_idxs[-min(keep_recent_turns, len(human_idxs))]
    else:
        boundary = len(messages)

    kept = []
    for m in messages[:boundary]:
        if isinstance(m, HumanMessage):
            kept.append(m)
        elif (
            isinstance(m, AIMessage)
            and not getattr(m, "tool_calls", None)
            and str(m.content).strip()
        ):
            kept.append(m)
        # else: ToolMessage or tool-call/empty AIMessage from an OLD turn — drop it.
    return kept + messages[boundary:]


def _maybe_autocompact(state: AgentState) -> AgentState:
    """If the turn that just finished left the context filled past `runtime.compact_threshold`, fold
    the older turns into an LLM summary (compaction.summarize_messages) so the NEXT turn doesn't
    re-send — and overflow — the window. This is the heavier LLM compaction; the mechanical
    `_compact_history` still runs every turn regardless.

    Best-effort and non-fatal: disabled via `runtime.auto_compact`, skipped when the fill is unknown,
    and any summary failure leaves the history untouched (summarize_messages swallows it). Mutates +
    returns `state` so the caller can keep its handle current."""
    cfg = get_config()
    if not cfg.get("runtime.auto_compact", True):
        return state
    used = int(state.get("context_tokens", 0) or 0)
    from core.llms import active_context_window

    window = active_context_window()
    if not window or used <= 0:
        return state
    threshold = float(cfg.get("runtime.compact_threshold", 0.85) or 0.85)
    if used / window < threshold:
        return state

    from core.compaction import summarize_messages

    new_msgs, stats = summarize_messages(state["messages"])
    if stats["summarized_turns"] > 0 and stats["after"] < stats["before"]:
        state["messages"] = new_msgs
        ui.note(
            f"auto-compacted {stats['summarized_turns']} earlier turn(s) "
            f"({stats['before']}→{stats['after']} messages) — context was "
            f"{used / window * 100:.0f}% full ({_human_int(used)}/{_human_int(window)} tok)."
        )
    return state


def _human_int(n: int) -> str:
    """Compact integer for the auto-compaction notice (1800 -> 1.8k)."""
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


# The only fields that survive a turn boundary: the conversation itself (compacted, appended
# to below) and the context-fill gauge (the window only grows; the next LLM call overwrites it).
_CARRY_ACROSS_TURNS = ("messages", "context_tokens")


def _fresh_turn(state: AgentState, user_input: str) -> AgentState:
    """Append the new query and reset per-turn fields (accumulators + loop counter).
    `messages` persists across turns to keep in-process conversation memory, but is first
    compacted (see _compact_history): older turns collapse to a clean Q&A transcript while the
    most recent turn's tool scratchpad is retained so a follow-up can refer back to it."""
    state["messages"] = _compact_history(state["messages"])
    state["messages"].append(HumanMessage(content=user_input))
    # Arm a fresh snapshot batch for this turn (lazy — created only if a file tool mutates
    # something), so /undo can reverse exactly the writes the turn that just ran made.
    from stores.snapshots import begin_turn

    begin_turn(user_input)
    # Clear the prompt-injection quarantine's per-turn flags (a flag raised last turn must not
    # escalate this turn's first tool batch).
    from trust import quarantine

    quarantine.reset_turn()
    # The reset is DERIVED from _initial_state — one canonical field list, so a new AgentState
    # field resets across turns automatically instead of silently leaking until someone
    # remembers to extend a second hand-maintained list (attachments is set by the loop after
    # mentions.expand; current_query is set to the new input below).
    fresh = _initial_state()
    for key in _CARRY_ACROSS_TURNS:
        fresh.pop(key, None)
    state.update(fresh)
    state["current_query"] = user_input
    return state


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "current_query": "",
        "context": "",
        "attachments": "",
        "plan": [],
        "iteration": 0,
        "rectify": False,
        "reasoning": "",
        "replans": 0,
        "aborted": False,
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
        "tool_events": [],
        "gate_events": [],
        "plan_vetoes": [],
        "answer_buffer": None,
        "tok_per_sec": 0.0,
        "context_tokens": 0,
    }
