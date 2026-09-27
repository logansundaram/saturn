"""Graph assembly: wire nodes/ into the compiled loop.

This is the ONLY place the LangGraph is assembled (see CLAUDE.md design rules) — the node
functions stay atomic in nodes/ (one file per node, routing helpers beside their node), and
everything runtime-shaped (turn driving, CLI, REPL) lives in the sibling app/ modules.
"""

import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_config
from core.state import AgentState
from nodes.agent import agent_node, route_after_agent
from nodes.approval import approval_node
from nodes.ground import grounding_node
from nodes.tools import tool_node

DB_PATH = str(get_config().path("db_sqlite"))


def build_agent():
    """The v2 loop (2026-09-27; spec docs/superpowers/specs/2026-09-27-v2-react-loop-design.md):

        START → ground → agent ─(no tool calls)─→ END
                           ↑          │ tool calls
                           │          ▼
                           └── tools ← approval   (a fully-rejected batch → agent)

    `agent` makes one native tool-calling call per pass (nodes/agent.py) and answers its own
    malformed / repeated calls with error ToolMessages (→ straight back to agent); `approval` is
    the human gate (Command(goto=...) — "tools", or "agent" when every call was declined);
    `tools` executes, clamps, records egress and fences quarantine. Compiled with a SqliteSaver
    checkpointer, which is what lets the approval / pause / ask_user `interrupt`s resume."""
    builder = StateGraph(AgentState)
    builder.add_node("ground", grounding_node)
    builder.add_node("agent", agent_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "ground")
    builder.add_edge("ground", "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent, {"approval": "approval", "agent": "agent", "end": END}
    )
    # approval routes dynamically via Command(goto=...): "tools" or "agent".
    builder.add_edge("tools", "agent")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(conn))
