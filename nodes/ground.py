import time
import diag

from langchain.messages import HumanMessage, AIMessage

from core.state import AgentState
from config import get_config
from textutil import clip
from stores.memory_registry import memory_context_split, mark_used
from stores.document_registry import (
    read_workspace_manifest,
    read_documents_manifest,
    sync_workspace_manifest,
)

"""
Grounding node (re-scoped from the old context_builder).

Its ONLY job is to load the things that are NOT already available to the model:
  - the document + workspace manifests (so the planner knows what docs/files exist),
  - the per-workspace SATURDAY.md instructions, and
  - persistent memory (stores/memory_registry): the user layer + open commitments + the recent
    memo digest always, agent/entities/negative facts by match against the request, all under
    one cap. (The old user_profile.md / agent_profile.md files were folded into the user and
    agent layers 2026-09-02 — nothing ever wrote them.)

It deliberately does NOT include:
  - the tool inventory  -> the planner's system prompt carries the catalog and the execute
                           step names its one tool; duplicating them here hurts small models.
  - the chat history    -> `messages` is already passed to the model directly.

Built once per turn (manifests/memory are static within a turn). Dynamic information —
tool results — flows through `messages`, never this frozen grounding string.

The block is built in TWO halves (2026-09-04): `context_stable` — instructions, manifests, the
query-independent memory layers — is byte-identical across turns while nothing on disk
changed, and `context_dynamic` — memory's by-match selection, the recent-conversation recap,
attachments — changes every turn. Every node's prompt sends the stable half as its own message
right after the system prompt and the dynamic half after it, and the idle prime (core/prime.py)
re-sends exactly `stable_grounding()` between turns so the daemon holds a checkpoint at that
message boundary: the next turn's plan/execute/synthesize calls then prefill only what is new
(core/serving.py, "the prefix cache"). `context` stays the joined block for every reader that
wants the whole thing (/trace context, older checkpoints).
"""

# Per-workspace instructions (the CLAUDE.md/AGENTS.md equivalent): a SATURDAY.md at the workspace
# root is loaded into context EVERY turn, so the user can durably steer how the agent treats this
# workspace (conventions, goals, what matters) without re-typing it. Drafted by /init, hand-edited
# freely. Capped so a runaway instructions file can't eat the context window.
_INSTRUCTIONS_FILE = "SATURDAY.md"
_INSTRUCTIONS_CAP = 6000


def _read_instructions() -> str:
    path = get_config().path("workspace") / _INSTRUCTIONS_FILE
    if not path.exists():
        return ""
    # errors="replace": a hand-edited file with a stray non-UTF-8 byte must not fail every turn
    # at the first node.
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > _INSTRUCTIONS_CAP:
        text = text[:_INSTRUCTIONS_CAP] + "\n… (SATURDAY.md truncated — keep it concise)"
    return text

# How many prior Q&A exchanges to recap into context, and how much of each to keep. Small on
# purpose: enough for the planner/synthesizer to resolve a follow-up ("do that for the other
# file") without re-bloating context — the full prior turn already rides `messages`.
_RECAP_EXCHANGES = 2
_RECAP_CHARS = 240


def _recent_exchanges(messages: list) -> str:
    """A compact recap of the last few completed Q&A exchanges, for the planner/synthesizer —
    which read `context` but are NOT given the raw `messages` the agent sees. Without this they
    are blind to the conversation, so a follow-up turn gets planned/synthesized as if it arrived
    cold. Pairs each user question with the assistant's final (non-tool-call) answer; skips the
    current in-flight query (the trailing HumanMessage with no answer yet)."""
    from core.state import is_turn_start

    pairs = []
    pending_q = None
    for m in messages:
        if isinstance(m, HumanMessage):
            # Not every HumanMessage is a question: a compaction summary is carried history and a
            # standalone mid-turn steer note is a correction — pairing either with the next answer
            # corrupts the recap (is_turn_start owns that rule). With those skipped, the LATEST
            # question wins, so a question left unanswered by a failed turn is superseded instead
            # of mis-pairing with the next turn's answer.
            if not is_turn_start(m):
                continue
            text = str(m.content).strip()
            if text:
                pending_q = text
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            answer = str(m.content).strip()
            if pending_q and answer:
                pairs.append((pending_q, answer))
                pending_q = None
            elif pairs and answer:
                # A LATER no-tool AIMessage in the same turn supersedes the pair's answer: a
                # normal turn carries TWO consecutive no-tool AIMessages — the agent's draft
                # (the finish that routed to replan/synthesize) then synthesize's final answer —
                # and the recap must show the answer the user actually saw, never the draft.
                # Worst case otherwise is the replan-repair path: the recap would carry the very
                # ungrounded draft the judge rejected. The `and answer` guard keeps an empty
                # trailing AIMessage from blanking a real final answer.
                pairs[-1] = (pairs[-1][0], answer)
    if not pairs:
        return ""

    lines = []
    for q, a in pairs[-_RECAP_EXCHANGES:]:
        lines.append(f"- User: {clip(q, _RECAP_CHARS)}\n  You: {clip(a, _RECAP_CHARS)}")
    return "\n".join(lines)


def stable_grounding() -> str:
    """The query-independent half of the grounding block — what the idle prime re-sends between
    turns. Byte-identical to the `context_stable` the next turn's grounding_node builds unless
    the workspace, the knowledge base, SATURDAY.md or the always-loaded memory layers changed
    in between (in which case the prime simply misses and the turn prefills it, as before)."""
    sections = ["## Grounding context"]

    instructions = _read_instructions()
    if instructions:
        sections.append(
            "### Workspace instructions (SATURDAY.md — the user's standing guidance "
            "for this workspace; follow it)\n" + instructions
        )

    docs_manifest = read_documents_manifest().strip()
    sections.append(
        "### Knowledge base (searchable via `search_knowledge_base`)\n"
        + (docs_manifest or "No ingested documents yet.")
    )

    # Reconcile the manifest with the workspace on disk FIRST: a file deleted or dropped in
    # outside the agent would otherwise leave this block naming a phantom (which the planner
    # then reads, fails, and replans around) or missing a real file. Best-effort — a sync
    # failure must never fail the first node of every turn.
    try:
        removed, added = sync_workspace_manifest()
        if removed or added:
            diag.log(f"grounding_node : workspace manifest synced "
                     f"(-{len(removed)} phantom, +{len(added)} unregistered)")
    except Exception as exc:
        diag.log(f"grounding_node : workspace manifest sync failed: {exc}")
    ws_manifest = read_workspace_manifest().strip()
    sections.append(
        "### Workspace files (accessible via read_file / write_file / list_directory)\n"
        + (ws_manifest or "No workspace files yet.")
    )

    # The always-loaded memory layers (user, commitments, the memo digest) are query-independent
    # — the stable half. The by-match facts land in the dynamic half below.
    always, _matched, _ids = memory_context_split("")
    if always:
        sections.append(
            "### Persistent memory (what the user asked me to remember and what I learned; "
            "#id lets `remember(..., replaces=<id>)` correct a fact)\n" + always
        )
    return "\n\n".join(sections)


def grounding_node(state: AgentState) -> dict:
    start = time.perf_counter()

    stable = stable_grounding()
    sections = []

    # Selected against THIS request (memory_registry.select_for_context): agent/entities/
    # negative facts only when they share tokens with the query, plus the trailer naming what
    # didn't load — under one cap with the always half above. /trace context shows the exact
    # block, so selection stays auditable. The by-match facts and the memo digest that loaded
    # get their last-used stamped (the expiry signal /memory flags stale on) — the one
    # read-path write, and it touches no fact text; best-effort — a stamp failure must never
    # fail the first node of every turn.
    _always, matched, matched_ids = memory_context_split(state.get("current_query", ""))
    if matched:
        sections.append(
            "### Memory facts matched to this request (same store; #id as above)\n" + matched
        )
    if matched_ids:
        try:
            mark_used(matched_ids)
        except Exception as exc:
            diag.log(f"grounding_node : memory last-used stamp failed: {exc}")

    # Per turn: the recap changes every turn and the request follows it.
    recap = _recent_exchanges(state.get("messages", []))
    if recap:
        sections.append(
            "### Recent conversation (this session — for resolving follow-up references)\n"
            + recap
        )

    # Files the user attached to THIS message with `@path` (resolved + read by mentions.expand in the
    # REPL loop, stashed on state). Folded in here so the planner/agent/synthesize — which read this
    # context, not the raw messages — all see the file contents inline. Empty on a turn with no
    # resolvable mentions.
    attachments = state.get("attachments", "")
    if attachments:
        sections.append(attachments)

    dynamic = "\n\n".join(sections)
    context = stable + ("\n\n" + dynamic if dynamic else "")
    diag.log(f"grounding_node : {time.perf_counter() - start:.4f}s")
    return {"context": context, "context_stable": stable, "context_dynamic": dynamic}
