"""Cross-turn conversation state.

The per-turn state shape (`_initial_state`), the fresh-turn reset (`_fresh_turn` — append the
new query, re-arm per-turn machinery, zero the accumulators), and the two history compactions:
the mechanical `_compact_history` that runs every turn, and the heavier LLM-summarizing
`_maybe_autocompact` that fires only past `runtime.compact_threshold`.
"""

from langchain.messages import HumanMessage, AIMessage

import diag
from config import get_config
from core.state import AgentState
from tui import ui
from tui.ui._base import _human_tokens


def _compact_history(messages: list, keep_recent_turns: int = 1) -> list:
    """Collapse OLDER completed turns to their conversational essence (user questions + final
    answers), but keep the ReAct scratchpad — tool-call AIMessages and their ToolMessages — of
    the most recent `keep_recent_turns` turns verbatim.

    The scratchpad of the turn that just finished is exactly what the user's *next* message
    refers back to ("open the second result", "what did that file say") — without it the model
    re-runs a search or fabricates. Older turns are still compacted: many turns of scratchpad
    make the model treat a long-finished tool call as "already done", bloat context with heavy
    tool outputs, and desync `messages` from the per-turn trace accumulators
    (`tools_called`/`tool_results`/`documents_retrieved`, reset each turn).

    A turn starts at a REAL user HumanMessage (core.state.is_turn_start) — not a mid-turn steer
    note and not a compaction summary. Everything from the boundary onward is kept as-is (no
    orphaned tool calls); everything before it is reduced to Human + non-empty final-AI messages
    (also orphan-free). Run only at the turn boundary. `keep_recent_turns=0` strips every turn."""
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


def _chars(messages: list) -> int:
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages)


def _maybe_autocompact(state: AgentState, run_id=None) -> AgentState:
    """If the turn that just finished left the context filled past `runtime.compact_threshold`, fold
    the older turns into an LLM summary (compaction.summarize_messages) so the NEXT turn doesn't
    re-send — and overflow — the window. This is the heavier LLM compaction; the mechanical
    `_compact_history` still runs every turn regardless.

    The summary keeps the most recent turn verbatim, and that turn is usually what filled the
    window (one research turn of ten reads). So when folding the older turns leaves the estimated
    fill still past the threshold — or there was nothing older to fold — the tool results that
    remain are trimmed to a head and a tail (compaction.trim_observations). Without that the next
    request starts over the threshold and its first tool result overflows num_ctx, where Ollama
    drops the system prompt and tool catalog from the front with no error.

    Best-effort and non-fatal: disabled via `runtime.auto_compact`, skipped when the fill is unknown,
    and any summary failure leaves the history untouched (summarize_messages swallows it). Mutates +
    returns `state` so the caller can keep its handle current. `run_id` is the run that just
    ended — the compaction fires after end_run, so the memory candidates the summary queues
    carry it explicitly (stores.trace.current_run_id is already None here)."""
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

    from core.compaction import summarize_messages, trim_observations

    before = _chars(state["messages"])
    new_msgs, stats = summarize_messages(state["messages"])
    folded = stats["summarized_turns"] > 0 and stats["after"] < stats["before"]
    kept = new_msgs if folded else state["messages"]
    # The fill after folding, estimated from what was removed at ~4 characters a token (tool
    # output runs denser, so this errs toward trimming).
    trimmed = 0
    if (used - (before - _chars(kept)) / 4) / window >= threshold:
        kept, trimmed = trim_observations(kept)
    if folded or trimmed:
        state["messages"] = kept
    fill = (f"{used / window * 100:.0f}% full "
            f"({_human_tokens(used)}/{_human_tokens(window)} tok).")
    if trimmed:
        ui.note(f"auto-compacted: trimmed {trimmed} tool result(s) from the last turn — context "
                f"was {fill}")
    if folded:
        ui.note(f"auto-compacted {stats['summarized_turns']} earlier turn(s) "
                f"({stats['before']}→{stats['after']} messages) — context was {fill}")
        # Persist the summary beside the memory file (the last session's brief) and queue its
        # bullets as memory candidates — proposals for /memory review, never facts written on
        # their own (core/memory_review).
        try:
            from core.compaction import is_summary
            from core.memory_review import note_compaction

            head = new_msgs[0] if new_msgs else None
            if head is not None and is_summary(head):
                queued = note_compaction(str(head.content).split(":", 1)[-1], run_id)
                if queued:
                    ui.note(f"{queued} memory candidate(s) from the summary queued — "
                            "/memory review to keep or drop them.")
        except Exception as exc:
            diag.log(f"compaction: memory candidate queue failed: {exc}")
    return state


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
    # Open the grant-lifecycle task: a task-scoped always-allow grant left standing by an aborted
    # turn (Ctrl-C never reached end_task) expires here instead of leaking into this turn.
    from trust import policy

    policy.begin_task()
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
        "skill": "",
        "plan": [],
        "iteration": 0,
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
        "tool_events": [],
        "gate_events": [],
        "tok_per_sec": 0.0,
        "context_tokens": 0,
    }
