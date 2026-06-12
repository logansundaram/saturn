from commands._framework import command, _print


@command(
    "compact",
    "Summarize older turns into one brief to free up the context window.",
    aliases=("summarize",),
    usage="/compact",
    details="""
Folds the older turns of this conversation into a single dense, LLM-written summary (via the
`utility` model), keeping the most recent turn verbatim so follow-ups still resolve. Use it when a
long session is filling the context window and you'd rather keep going than /reset.

This is the heavier sibling of the automatic per-turn compaction: every turn already collapses old
turns to their Q&A mechanically (no LLM), and the agent ALSO auto-compacts on its own when the
context fills past runtime.compact_threshold (default 85%). /compact triggers the LLM summary on
demand, now.

Lossy by nature — it trades transcript detail for space. Facts, decisions, your stated
preferences, and open threads are preserved; verbatim phrasing and tool-output detail are not. The
durable trace (/trace, /trace calls) is untouched. To clear the conversation entirely instead,
use /reset.

Example:
  /compact
""",
)
def _compact(ctx, args):
    from core.compaction import summarize_messages

    msgs = ctx.state.get("messages", [])
    if len(msgs) < 2:
        _print("  nothing to compact yet — have a few turns first.")
        return
    _print("  compacting older turns (one LLM summary call)…")
    new_msgs, stats = summarize_messages(msgs)
    if stats["summarized_turns"] <= 0:
        _print("  only the most recent turn is present — nothing older to summarize.")
        return
    if stats["after"] >= stats["before"]:
        _print("  summary did not shrink the history — left it unchanged (see logging/diag.log).")
        return
    ctx.state["messages"] = new_msgs
    _print(
        f"  compacted {stats['summarized_turns']} earlier turn(s): "
        f"{stats['before']} → {stats['after']} messages. Most recent turn kept verbatim."
    )
