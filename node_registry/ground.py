import time
import diag

from langchain.messages import HumanMessage, AIMessage

from state import AgentState
from config import get_config
from stores.memory_registry import read_memory_block
from stores.document_registry import (
    read_workspace_manifest,
    read_documents_manifest,
)

"""
Grounding node (re-scoped from the old context_builder; see SATURDAY_MVP_PLAN.md §8).

Its ONLY job is to load the things that are NOT already available to the model:
  - the document + workspace manifests (so the planner knows what docs/files exist),
  - persistent user/agent profiles (the persistent layer of memory), if present, and
  - durable facts saved via the `remember` tool (Phase 3 working memory).

It deliberately does NOT include:
  - the tool inventory  -> tools are bound natively via bind_tools; duplicating them as text
                           hurts tool-calling on small local models.
  - the chat history    -> `messages` is already passed to the model directly.

Built once per turn (manifests/profiles are static within a turn). Dynamic information —
tool results — flows through `messages`, never this frozen grounding string.
"""

# Persistent profiles live alongside the workspace. Optional — missing files are fine.
_PROFILE_FILES = ("user_profile.md", "agent_profile.md")

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
    pairs = []
    pending_q = None
    for m in messages:
        if isinstance(m, HumanMessage):
            pending_q = str(m.content).strip()
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            answer = str(m.content).strip()
            if pending_q and answer:
                pairs.append((pending_q, answer))
                pending_q = None
    if not pairs:
        return ""

    def _clip(s: str) -> str:
        s = " ".join(s.split())
        return s if len(s) <= _RECAP_CHARS else s[: _RECAP_CHARS - 1] + "…"

    lines = []
    for q, a in pairs[-_RECAP_EXCHANGES:]:
        lines.append(f"- User: {_clip(q)}\n  You: {_clip(a)}")
    return "\n".join(lines)


def _read_profiles() -> str:
    workspace = get_config().path("workspace")
    chunks = []
    for name in _PROFILE_FILES:
        path = workspace / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                chunks.append(text)
    return "\n\n".join(chunks)


def grounding_node(state: AgentState) -> dict:
    start = time.perf_counter()

    sections = ["## Grounding context"]

    profiles = _read_profiles()
    if profiles:
        sections.append("### Profiles\n" + profiles)

    memory = read_memory_block()
    if memory:
        sections.append(
            "### Persistent memory (facts the user asked me to remember)\n" + memory
        )

    recap = _recent_exchanges(state.get("messages", []))
    if recap:
        sections.append(
            "### Recent conversation (this session — for resolving follow-up references)\n"
            + recap
        )

    docs_manifest = read_documents_manifest().strip()
    sections.append(
        "### Knowledge base (searchable via `search_knowledge_base`)\n"
        + (docs_manifest or "No ingested documents yet.")
    )

    ws_manifest = read_workspace_manifest().strip()
    sections.append(
        "### Workspace files (accessible via read_file / write_file / list_directory)\n"
        + (ws_manifest or "No workspace files yet.")
    )

    # Files the user attached to THIS message with `@path` (resolved + read by mentions.expand in the
    # REPL loop, stashed on state). Folded in here so the planner/agent/synthesize — which read this
    # context, not the raw messages — all see the file contents inline. Empty on a turn with no
    # resolvable mentions.
    attachments = state.get("attachments", "")
    if attachments:
        sections.append(attachments)

    context = "\n\n".join(sections)
    diag.log(f"grounding_node : {time.perf_counter() - start:.4f}s")
    return {"context": context}
