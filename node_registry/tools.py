"""
Tool-execution node for the living-plan ReAct loop (Phase 1).

tool_node executes the tool calls on the last AI message, appends the results as ToolMessages
back into `messages` (so the model sees them next iteration), and mirrors each
`name(args) -> result` into the trace accumulators — paired so synthesis can't divorce a value
from the call that produced it.
"""

import time

from langchain.messages import ToolMessage

from registry import tools_by_name
from state import AgentState

# Tools whose results are documents worth recording as retrieved (for citations / trace).
_RETRIEVAL_TOOLS = {"search_knowledge_base"}

# Cap each argument's length so a big write_file payload doesn't bloat the trace/synthesis input.
_MAX_ARG_REPR = 200

# Cap the one-line result preview carried in tool_events (UI tree); the full observation still
# rides messages/tool_results untouched.
_MAX_RESULT_PREVIEW = 160


def _preview(observation: str) -> str:
    """Collapse a tool observation to a single capped line for the UI's tool-I/O tree."""
    one_line = " ".join(observation.split())
    if len(one_line) > _MAX_RESULT_PREVIEW:
        one_line = one_line[: _MAX_RESULT_PREVIEW - 1] + "…"
    return one_line


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
    tool_events = []

    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        ok = True
        start = time.perf_counter()
        selected = tools_by_name.get(name)
        if selected is None:
            observation = f"Error: unknown tool '{name}'."
            ok = False
        else:
            try:
                observation = selected.invoke(args)
            except Exception as exc:  # surface tool errors to the model instead of crashing
                observation = f"Error calling {name}: {exc}"
                ok = False
        dur = time.perf_counter() - start

        observation = str(observation)
        tool_messages.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"], name=name)
        )
        tools_called.append(name)
        # Pair the result with its call so synthesis can't divorce the value from what it
        # answers (this is what stops the model recomputing and contradicting the tool).
        tool_results.append(f"{_fmt_call(name, args)} -> {observation}")
        # Structured per-call record for the UI's tool-I/O tree (args + result preview + timing).
        tool_events.append(
            {
                "name": name,
                "args": args,
                "result": _preview(observation),
                "dur": dur,
                "ok": ok,
            }
        )
        if name in _RETRIEVAL_TOOLS:
            documents_retrieved.append(observation)

    return {
        "messages": tool_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "documents_retrieved": documents_retrieved,
        "tool_events": tool_events,
    }
