"""
Tool-execution node for the living-plan ReAct loop (Phase 1).

tool_node executes the tool calls on the last AI message, appends the results as ToolMessages
back into `messages` (so the model sees them next iteration), and mirrors each
`name(args) -> result` into the trace accumulators — paired so synthesis can't divorce a value
from the call that produced it.
"""

from langchain.messages import ToolMessage

from registry import tools_by_name
from state import AgentState

# Tools whose results are documents worth recording as retrieved (for citations / trace).
_RETRIEVAL_TOOLS = {"search_knowledge_base"}

# Cap each argument's length so a big write_file payload doesn't bloat the trace/synthesis input.
_MAX_ARG_REPR = 200


def _fmt_call(name: str, args: dict) -> str:
    """Render a tool call like  calculate(expression='847 * 293 + 12450')  for the trace and
    for synthesis, so results stay linked to the call that produced them."""
    parts = []
    for k, v in (args or {}).items():
        r = repr(v)
        if len(r) > _MAX_ARG_REPR:
            r = r[:_MAX_ARG_REPR] + "…"
        parts.append(f"{k}={r}")
    return f"{name}({', '.join(parts)})"


def tool_node(state: AgentState):
    """Execute the tool calls on the last AI message and feed results back as ToolMessages."""
    last = state["messages"][-1]
    tool_messages = []
    tools_called = []
    tool_results = []
    documents_retrieved = []

    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        selected = tools_by_name.get(name)
        if selected is None:
            observation = f"Error: unknown tool '{name}'."
        else:
            try:
                observation = selected.invoke(args)
            except Exception as exc:  # surface tool errors to the model instead of crashing
                observation = f"Error calling {name}: {exc}"

        observation = str(observation)
        tool_messages.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"], name=name)
        )
        tools_called.append(name)
        # Pair the result with its call so synthesis can't divorce the value from what it
        # answers (this is what stops the model recomputing and contradicting the tool).
        tool_results.append(f"{_fmt_call(name, args)} -> {observation}")
        if name in _RETRIEVAL_TOOLS:
            documents_retrieved.append(observation)

    return {
        "messages": tool_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "documents_retrieved": documents_retrieved,
    }
