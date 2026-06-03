"""
Core nodes of the living-plan ReAct loop (Phase 1).

  agent_node       — the reason-and-act step: model with tools bound natively. Emits tool
                     calls, or no tool calls to signal it is done gathering.
  tool_node        — executes the requested tool calls, appends ToolMessages back into
                     `messages` (so the model sees results next iteration), and mirrors them
                     into the trace accumulators.
  route_after_agent — conditional edge: loop to tools, or finish at synthesize. Also enforces
                      the max-iteration guardrail.

The loop is assembled in agent.py:
    ground -> plan -> agent -> (tools -> update_plan -> agent)*  -> synthesize -> END

approval_node (Phase 2) will wrap tool_node with an interrupt for side-effecting tools.
"""

import time
from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langgraph.types import interrupt, Command

from registry import tools_by_name, risk_of
from llms import llm_with_tools
from state import AgentState
from messages import agent_loop_system_msg

# Hard cap on loop iterations so a confused model can't spin forever. Becomes config in Phase 3.
MAX_ITERATIONS = 8

# Tools whose results are documents worth recording as retrieved (for citations / trace).
_RETRIEVAL_TOOLS = {"search_knowledge_base"}

_STATUS_GLYPH = {"pending": "○", "active": "▶", "done": "✓", "skipped": "—"}

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


def render_plan(plan: list[dict]) -> str:
    """Human-readable checklist, injected into the agent's context and streamed to the UI."""
    if not plan:
        return "(no plan)"
    lines = []
    for step in plan:
        glyph = _STATUS_GLYPH.get(step["status"], "○")
        tool = f"  [{step['intended_tool']}]" if step.get("intended_tool") else ""
        lines.append(f"{glyph} {step['step_id']}. {step['label']}{tool}")
    return "\n".join(lines)


def agent_node(state: AgentState):
    """One ReAct decision: look at the plan + conversation, then call tools or finish."""
    start = time.perf_counter()

    messages = [
        agent_loop_system_msg,
        SystemMessage(content=state.get("context", "")),
        SystemMessage(content="Current plan:\n" + render_plan(state.get("plan", []))),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    print(
        f"agent_node : {time.perf_counter() - start:.4f}s "
        f"(iter {state.get('iteration', 0)}, {len(tool_calls)} tool call(s))"
    )
    return {"messages": [response], "iteration": state.get("iteration", 0) + 1}


def tool_node(state: AgentState):
    """Execute the tool calls on the last AI message and feed results back as ToolMessages."""
    start = time.perf_counter()

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

    print(f"tool_node : {time.perf_counter() - start:.4f}s ({len(tool_messages)} executed)")
    return {
        "messages": tool_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "documents_retrieved": documents_retrieved,
    }


def route_after_agent(state: AgentState) -> str:
    """Send tool requests through the approval gate (and we're under the cap); else finish."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("iteration", 0) < MAX_ITERATIONS:
        return "approval"
    return "synthesize"


def approval_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """Human-in-the-loop safety gate. Read-only tool batches pass straight through. If any
    pending tool call is side-effecting/destructive, pause via `interrupt` and let the user
    approve or reject the whole batch.

    On reject we still emit ToolMessages for every pending call (so the message history stays
    valid — orphaned tool_calls break the next model turn) and route back to the agent to
    respond without having performed the action."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    gated = [tc for tc in tool_calls if risk_of(tc["name"]) != "read_only"]

    if not gated:
        return Command(goto="tools")

    approved = interrupt(
        {
            "type": "approval_request",
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"], "risk": risk_of(tc["name"])}
                for tc in gated
            ],
        }
    )

    if approved:
        return Command(goto="tools")

    decline = [
        ToolMessage(
            content="Execution declined by the user. Do not retry this action; tell the user you did not perform it.",
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in tool_calls
    ]
    return Command(goto="agent", update={"messages": decline})
