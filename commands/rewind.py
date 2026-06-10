"""
/rewind — drop the last exchange from the carried conversation.

The conversational complement of /undo: /undo reverts what the last turn did to FILES, /rewind
reverts what it did to the CONVERSATION (a derailed answer polluting context, a question you wish
you hadn't asked). Repeatable — each /rewind walks one more turn back. Files are deliberately NOT
touched (point /undo at those), and the trace keeps its record: rewinding edits what the model
sees next turn, never the audit trail.
"""

from commands._framework import command, _print
from commands._session import write_autosave

# What one user message gets echoed back as in the confirmation line.
_PREVIEW_CHARS = 70


def _preview(text) -> str:
    one_line = " ".join(str(text).split())
    if len(one_line) > _PREVIEW_CHARS:
        one_line = one_line[: _PREVIEW_CHARS - 1] + "…"
    return one_line


def drop_last_turn(ctx) -> "str | None":
    """Remove the most recent turn (the last HumanMessage and everything after it) from the
    carried messages, clear the per-turn scratch that referred to it, and re-autosave so a crash
    can't resurrect what was just rewound. Returns the dropped user query, or None when there was
    no turn to drop. Shared with /retry full (rewind + re-run)."""
    from langchain.messages import HumanMessage

    msgs = ctx.state.get("messages", [])
    human_idxs = [i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)]
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

    write_autosave(ctx.state)
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
    _print(f'  rewound the last turn — dropped {n_dropped} message(s) ("{_preview(dropped)}").')
    remaining = sum(
        1 for m in ctx.state.get("messages", []) if m.__class__.__name__ == "HumanMessage"
    )
    if remaining:
        _print(f"  {remaining} earlier turn(s) remain; /rewind again to keep walking back.")
    else:
        _print("  the conversation is now empty.")
    _print("  (files were not touched — /undo reverts those.)")
