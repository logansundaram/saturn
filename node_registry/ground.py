import time

from state import AgentState
from config import get_config
from memory_registry import read_memory_block
from document_registry import (
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

    context = "\n\n".join(sections)
    print(f"grounding_node : {time.perf_counter() - start:.4f}s")
    return {"context": context}
