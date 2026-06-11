"""
/retry — regenerate the last answer.

Two depths, matching the two ways a turn disappoints:

  bare /retry    the gathering was fine but the ANSWER was weak — re-run only the synthesize
                 step over the same gathered results (tool results, retrieved documents, draft),
                 replacing the last answer. No tools re-run, no gate, one LLM call.
  /retry full    the whole TURN went wrong — rewind it (commands/rewind.py) and re-run the same
                 question as a fresh agent turn, plan and tools and all.

The bare form leans on the fact that the loop's per-turn scratch (tool_results,
documents_retrieved, current_query, plan) persists on state until the NEXT turn starts — exactly
the inputs synthesize_node needs. A /clear, /rewind, or /resume in between clears that scratch,
so /retry then points the user at /retry full.
"""

from commands._framework import command, _print
from commands._session import write_autosave


@command(
    "retry",
    "Regenerate the last answer (or re-run the whole last turn).",
    aliases=("regenerate",),
    usage="/retry [full]",
    details="""
  /retry        regenerate just the final answer from what the last turn already gathered —
                the same tool results, retrieved documents, and draft are synthesized again
                (one LLM call, no tools re-run, nothing re-gated) and the new answer replaces
                the old one in the conversation. Use when the research was right but the
                write-up wasn't.
  /retry full   rewind the last turn (like /rewind) and re-run its question from scratch as a
                fresh agent turn — new plan, new tool calls, the usual gates. Use when the turn
                itself went sideways.

Notes:
  - the bare form needs the last turn's working state, which survives until the next query;
    after /clear, /rewind, or /resume only /retry full is available.
  - the regenerated answer replaces the previous one in the conversation and the autosave; the
    trace keeps the original turn's record (the regeneration is not a traced run).
  - file changes are never re-run or reverted by either form (/undo owns files).
""",
)
def _retry(ctx, args):
    if args and args[0].lower() in ("full", "--full", "-f"):
        return _retry_full(ctx)
    if args:
        _print(f"  unknown argument {args[0]!r} — usage: /retry [full]")
        return
    return _retry_synthesize(ctx)


def _last_query(ctx) -> "str | None":
    from langchain.messages import HumanMessage

    from compaction import is_summary
    from state import is_steer_message

    # Last REAL question: a standalone mid-turn steer note is a correction to a turn, not the
    # turn's query (requeueing it would re-run the correction without the question), and a
    # compaction summary is carried history.
    for m in reversed(ctx.state.get("messages", [])):
        if isinstance(m, HumanMessage) and not is_steer_message(m) and not is_summary(m):
            return str(m.content)
    return None


def _retry_full(ctx):
    from commands.rewind import drop_last_turn

    query = _last_query(ctx)
    if not query:
        _print("  nothing to retry — the conversation is empty.")
        return
    drop_last_turn(ctx)
    # The REPL loop consumes `requeue` right after this handler returns and runs it as an
    # ordinary agent turn (agent.main) — fresh plan, fresh gates, fresh trace run.
    ctx.requeue = query
    _print("  re-running the last turn from scratch…")


def _retry_synthesize(ctx):
    from langchain.messages import AIMessage

    state = ctx.state
    if not state.get("current_query"):
        _print("  no working state from a previous turn to regenerate from "
               "(it clears on /clear, /rewind, and /resume) — try /retry full.")
        return
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
        _print("  the last turn left no final answer to regenerate — try /retry full.")
        return

    # Pop the old answer so synthesize sees the conversation exactly as it did the first time
    # (its draft-answer input is whatever now trails the messages); restore it on any failure so
    # a model hiccup can't eat the only answer the user has.
    popped = msgs.pop()
    _print("  regenerating the answer from the last turn's gathered results…")
    try:
        from node_registry.synthesize import synthesize_node

        out = synthesize_node(state)
    except Exception:
        msgs.append(popped)
        raise

    new_msg = out["messages"][0]
    msgs.append(new_msg)
    state["tok_per_sec"] = out.get("tok_per_sec", 0.0)
    state["context_tokens"] = out.get("context_tokens", state.get("context_tokens", 0))
    write_autosave(state)

    from tui import ui

    ui.response(str(new_msg.content))
