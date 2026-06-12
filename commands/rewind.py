"""
/rewind — drop the last exchange from the carried conversation.

The conversational complement of /undo: /undo reverts what the last turn did to FILES, /rewind
reverts what it did to the CONVERSATION (a derailed answer polluting context, a question you wish
you hadn't asked). Repeatable — each /rewind walks one more turn back. Files are deliberately NOT
touched (point /undo at those), and the trace keeps its record: rewinding edits what the model
sees next turn, never the audit trail.
"""

from commands._framework import command, _print
from commands._session import clear_autosave, write_autosave
from textutil import clip

# What one user message gets echoed back as in the confirmation line.
_PREVIEW_CHARS = 70


def drop_last_turn(ctx) -> "str | None":
    """Remove the most recent turn (the last HumanMessage and everything after it) from the
    carried messages, clear the per-turn scratch that referred to it, and re-autosave so a crash
    can't resurrect what was just rewound. Returns the dropped user query, or None when there was
    no turn to drop. Shared with /retry full (rewind + re-run)."""
    from langchain.messages import HumanMessage

    from core.compaction import is_summary
    from core.state import is_steer_message

    msgs = ctx.state.get("messages", [])
    # A turn starts at a REAL user message: a standalone mid-turn steer note belongs to the turn
    # it corrected (slicing there would leave the question + half a scratchpad behind), and a
    # compaction summary is carried history, not a turn.
    human_idxs = [
        i
        for i, m in enumerate(msgs)
        if isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m)
    ]
    if not human_idxs:
        return None
    boundary = human_idxs[-1]
    dropped_query = str(msgs[boundary].content)
    ctx.state["messages"] = msgs[:boundary]

    # The per-turn scratch (plan, accumulators, current_query) described the turn that just got
    # dropped — clear it so nothing (notably /retry's re-synthesize) can act on a rewound turn.
    fresh = ctx.state
    fresh["current_query"] = ""
    fresh["plan"] = []
    fresh["iteration"] = 0
    fresh["agent_nudges"] = 0
    fresh["replans"] = 0
    fresh["tools_called"] = []
    fresh["tool_results"] = []
    fresh["documents_retrieved"] = []
    fresh["tool_events"] = []
    # Gate decisions belong to the dropped turn too — a lingering record would violate the
    # "gate_events empty means the human was never asked" invariant /glass and exports rely on.
    fresh["gate_events"] = []

    # write_autosave skips an empty conversation by design (a fresh launch + quit must not wipe
    # the previous session) — but rewinding the ONLY turn empties it deliberately, so clear the
    # slot explicitly or a crash/quit + /resume would resurrect the turn that was just rewound.
    if ctx.state["messages"]:
        write_autosave(ctx.state)
    else:
        clear_autosave()
    return dropped_query


@command(
    "rewind",
    "Drop the last exchange from the conversation (files untouched).",
    usage="/rewind",
    details="""
Removes the most recent turn — your last message and everything the agent said and gathered in
response — from the conversation the model carries forward, so a derailed or unwanted exchange
stops polluting the context. Repeat to walk further back, one turn per /rewind.

Scope (deliberate):
  - conversation only — file changes are NOT reverted; /undo does that (the two compose:
    /rewind + /undo fully unwinds a bad turn).
  - the trace is NOT rewritten — /trace still shows the full record of what actually ran.
    Rewinding changes what the model sees next turn, never the audit trail.
  - the autosave slot is updated immediately, so quitting after a /rewind stays rewound.

See also: /retry (regenerate the last answer), /clear (drop the whole conversation).
""",
)
def _rewind(ctx, args):
    n_before = len(ctx.state.get("messages", []))
    dropped = drop_last_turn(ctx)
    if dropped is None:
        _print("  nothing to rewind — the conversation is empty.")
        return
    n_dropped = n_before - len(ctx.state.get("messages", []))
    _print(f'  rewound the last turn — dropped {n_dropped} message(s) ("{clip(dropped, _PREVIEW_CHARS)}").')
    remaining = sum(
        1 for m in ctx.state.get("messages", []) if m.__class__.__name__ == "HumanMessage"
    )
    if remaining:
        _print(f"  {remaining} earlier turn(s) remain; /rewind again to keep walking back.")
    else:
        _print("  the conversation is now empty.")
    _print("  (files were not touched — /undo reverts those.)")
