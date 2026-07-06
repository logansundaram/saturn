"""
LLM conversation compaction — fold older turns into a dense summary so a long session stops
re-sending full transcripts and outgrowing the context window.

Two tiers of compaction exist, by cost:
  - MECHANICAL (`agent._compact_history`): runs every turn, collapses OLDER turns to their Q&A and
    keeps the most recent turn's ReAct scratchpad. Fast, structural, no LLM — always on.
  - LLM SUMMARY (here): folds the turns older than the most recent into a SINGLE dense summary via
    the `utility` model. Heavier (an LLM call), so it is NOT per-turn — it fires manually (`/compact`)
    or automatically when the context fills past `runtime.compact_threshold` (`agent._maybe_autocompact`).

The summary is carried as a `HumanMessage` tagged with `_SUMMARY_PREFIX` so it (a) survives the
mechanical per-turn compaction that follows — which keeps HumanMessages — and (b) reads to the model
as established context. A later compaction folds the prior summary back into the next one (it's part
of the `older` slice that gets re-summarized), so history stays bounded instead of accreting summaries.
"""

from __future__ import annotations

import diag
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from core.state import is_turn_start
from textutil import truncate

# Tag marking the synthetic summary message. Kept on a HumanMessage so it rides through the
# mechanical compaction (which keeps Human + final-AI) instead of being dropped.
_SUMMARY_PREFIX = "[Earlier conversation, summarized]"


def is_summary(m) -> bool:
    """True if `m` is a compaction summary message (so it's excluded from turn counting and folded,
    not double-counted, on the next compaction)."""
    return isinstance(m, HumanMessage) and str(m.content).startswith(_SUMMARY_PREFIX)


def _role(m) -> str:
    if is_summary(m):
        return "Context"
    if isinstance(m, HumanMessage):
        return "User"
    if isinstance(m, AIMessage):
        return "Assistant"
    if isinstance(m, ToolMessage):
        return "Tool"
    return str(getattr(m, "type", "msg")).capitalize()


def _text(m, cap: int = 1000) -> str:
    """One-line, length-capped rendering of a message for the summary transcript. Tool-call
    AIMessages (empty content) surface the tool names so the summary knows what was attempted."""
    content = m.content
    if isinstance(content, list):  # multimodal / structured blocks
        content = " ".join(str(p) for p in content)
    content = " ".join(str(content).split())
    if not content:
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            content = "(calls: " + ", ".join(tc.get("name", "?") for tc in tcs) + ")"
    return truncate(content, cap)


def _transcript(messages) -> str:
    return "\n".join(f"{_role(m)}: {t}" for m in messages if (t := _text(m)))


def summarize_messages(messages: list, keep_recent_turns: int = 1):
    """Fold the turns older than the most recent `keep_recent_turns` into one summary HumanMessage;
    keep the recent turn(s) verbatim. Returns `(new_messages, stats)` where stats =
    `{before, after, summarized_turns}`.

    No-op (returns the input unchanged, summarized_turns=0) when there's nothing old enough to fold
    or the LLM summary fails — compaction is an optimization and must never lose or corrupt the
    conversation."""
    stats = {"before": len(messages), "after": len(messages), "summarized_turns": 0}

    # Turn boundaries = real user messages (a prior summary isn't a turn, and neither is a
    # standalone mid-turn steer note — slicing at one would fold the turn's REAL question into
    # the lossy summary while the correction survives as the apparent current question). The
    # shared is_turn_start predicate owns that rule. Need more than the retained window, or
    # there's nothing older to fold.
    human_idxs = [i for i, m in enumerate(messages) if is_turn_start(m)]
    if len(human_idxs) <= keep_recent_turns:
        return messages, stats

    boundary = human_idxs[-keep_recent_turns]
    older, recent = messages[:boundary], messages[boundary:]
    if not older:
        return messages, stats

    try:
        summary = _llm_summary(older)
    except Exception as exc:
        diag.log(f"compaction: LLM summary failed, leaving history intact: {exc}")
        return messages, stats
    if not summary:
        return messages, stats

    new = [HumanMessage(content=f"{_SUMMARY_PREFIX}:\n{summary}")] + recent
    stats["after"] = len(new)
    # Same predicate as the boundary scan: a steer note in the older slice is part of a turn,
    # not a turn — counting it would inflate the user-facing "compacted N earlier turn(s)" line.
    # (Steer notes still render in the summary transcript as "User:" lines via _role.)
    stats["summarized_turns"] = sum(1 for m in older if is_turn_start(m))
    return new, stats


def _llm_summary(older: list) -> str:
    """Summarize the `older` slice into a dense continuation brief via the `utility` model. Raises
    on an LLM failure (caller treats that as a no-op)."""
    import time

    from core.llms import get_model
    from core.messages import COMPACTION_PROMPT  # lazy: messages pulls the live tool registry

    prompt = HumanMessage(content=COMPACTION_PROMPT + _transcript(older))
    start = time.perf_counter()
    out = get_model("utility").invoke([prompt]).content
    diag.log(f"compaction: summarized {len(older)} msg(s) in {time.perf_counter() - start:.2f}s")
    return str(out).strip()
