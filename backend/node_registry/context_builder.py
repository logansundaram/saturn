import time
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import RemoveMessage

from state import AgentState
from registry import tool as tool_list
from messages import context_builder_system_msg_template
from document_registry import read_workspace_manifest, read_documents_manifest


"""
vibe coded, need to review

python -m node_registry.context_builder to run from root backend folder
"""
# # Maximum conversational (Human/AI) messages to keep.
# # Older turns beyond this window are deleted via RemoveMessage so state doesn't grow unbounded.
# _MAX_HISTORY_TURNS = 10

# # Stable ID so add_messages updates this message in place each turn rather than appending a duplicate.
# _ENV_MSG_ID = "__context_builder_env__"


# build tool context
def _build_tool_inventory() -> str:
    lines = []
    for t in tool_list:
        lines.append(f"- {t.name}: {t.description}")
    tool_content = "Available tools: \n" + "\n".join(lines)
    return tool_content


def _build_document_inventory() -> str:
    # TODO: implement this
    return "\nAvailable Documents: No documents yet."  # placeholder


def _build_chat_history(state: AgentState) -> str:
    messages = state.get("messages", [])

    lines = []

    for msg in messages:
        if not hasattr(msg, "content"):
            continue

        role = msg.__class__.__name__.replace("Message", "")
        lines.append(f"{role}: {msg.content}")

    if not lines:
        return "No previous messages."

    return "Previous messages:\n" + "\n".join(lines)


# might need a better system message here
def _build_context(state: AgentState) -> str:
    """Build the context for the agent."""
    context = (
        "This is the avaible context. Use this information to determine if RAG is necessary, tools are necessary, and if the past chat history is relevant"
        + _build_tool_inventory()
        + _build_document_inventory()
        + _build_chat_history(state)
    )
    return context


# def _format_manifest(raw: str, empty_label: str) -> str:
#     """Strip the manifest header line and return body, or a fallback label."""
#     if not raw.strip():
#         return empty_label
#     lines = raw.splitlines()
#     # Drop the "# Document manifest" header line
#     body_lines = [l for l in lines if not l.startswith("# ")]
#     body = "\n".join(body_lines).strip()
#     return body if body else empty_label


def context_builder_node(state: AgentState) -> dict:
    start = time.perf_counter()

    print("building context")
    context_sys_msg = _build_context(state)

    print(f"context_builder_node : {time.perf_counter() - start:.4f}s")
    return {"context": context_sys_msg}


# if __name__ == "__main__":
#     _build_tool_inventory()
