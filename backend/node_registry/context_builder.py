import time
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import RemoveMessage

from state import AgentState
from registry import tool as tool_list
from messages import context_builder_system_msg_template
from document_registry import read_workspace_manifest, read_documents_manifest


"""
vibe coded, need to review
"""
# Maximum conversational (Human/AI) messages to keep.
# Older turns beyond this window are deleted via RemoveMessage so state doesn't grow unbounded.
_MAX_HISTORY_TURNS = 10

# Stable ID so add_messages updates this message in place each turn rather than appending a duplicate.
_ENV_MSG_ID = "__context_builder_env__"


def _build_tool_inventory() -> str:
    lines = []
    for t in tool_list:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


def _format_manifest(raw: str, empty_label: str) -> str:
    """Strip the manifest header line and return body, or a fallback label."""
    if not raw.strip():
        return empty_label
    lines = raw.splitlines()
    # Drop the "# Document manifest" header line
    body_lines = [l for l in lines if not l.startswith("# ")]
    body = "\n".join(body_lines).strip()
    return body if body else empty_label


def context_builder_node(state: AgentState) -> dict:
    start = time.perf_counter()

    workspace_docs = _format_manifest(
        read_workspace_manifest(), "No workspace files yet."
    )
    rag_docs = _format_manifest(read_documents_manifest(), "No ingested documents yet.")

    env_msg = SystemMessage(
        id=_ENV_MSG_ID,
        content=context_builder_system_msg_template.format(
            tool_inventory=_build_tool_inventory(),
            workspace_docs=workspace_docs,
            rag_docs=rag_docs,
        ),
    )

    # Prune old conversational turns beyond the history window.
    convo_msgs = [
        m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))
    ]
    to_remove = []
    if len(convo_msgs) > _MAX_HISTORY_TURNS:
        for old_msg in convo_msgs[: len(convo_msgs) - _MAX_HISTORY_TURNS]:
            to_remove.append(RemoveMessage(id=old_msg.id))

    print(f"context_builder_node : {time.perf_counter() - start:.4f}s")
    return {"messages": to_remove + [env_msg]}
