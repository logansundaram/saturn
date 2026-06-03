import sys

# Force UTF-8 console output. Node prints (plan glyphs, tool results, model output) routinely
# contain non-cp1252 characters that crash print() on the default Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import sqlite3
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import HumanMessage

from state import AgentState

# loop nodes
from node_registry.ground import grounding_node
from node_registry.plan import plan_node
from node_registry.synthesize import synthesize_node
from node_registry.update_plan import update_plan_node
from node_registry.agent import agent_node, route_after_agent
from node_registry.tools import tool_node
from node_registry.approval import approval_node

# RAG ingest (populates the in-memory vector store the search_knowledge_base tool reads)
from rag import build_ingest

# transparency + safety UI
from trace import Tracer
import ui

# REPL meta-commands (lines starting with `/`)
import commands

DB_PATH = "database/db.sqlite"


def build_agent():
    """Assemble the living-plan ReAct loop with a human-in-the-loop approval gate:

        START -> ground -> plan -> agent -> approval -> (tools -> update_plan -> agent)* -> synthesize -> END
                                     │          │
                              (no tool calls)  (reject -> back to agent)
                                     ▼
                                 synthesize

    Compiled with a SqliteSaver checkpointer, which both persists sessions and is what lets the
    approval `interrupt` pause and resume.
    """
    builder = StateGraph(AgentState)

    builder.add_node("ground", grounding_node)
    builder.add_node("plan", plan_node)
    builder.add_node("agent", agent_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", tool_node)
    builder.add_node("update_plan", update_plan_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "ground")
    builder.add_edge("ground", "plan")
    builder.add_edge("plan", "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent, {"approval": "approval", "synthesize": "synthesize"}
    )
    # approval routes dynamically via Command(goto=...) to "tools" (approved) or "agent" (rejected)
    builder.add_edge("tools", "update_plan")
    builder.add_edge("update_plan", "agent")
    builder.add_edge("synthesize", END)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


def run_turn(graph, payload, config, approver, on_update=None):
    """Drive one turn to completion, streaming node updates and pausing at the approval gate.

    `approver(interrupt_value) -> bool` decides each approval. `on_update(node, delta)` is called
    for every node update (used for the trace + live plan panel). Returns the final state."""
    pending = payload
    while True:
        for chunk in graph.stream(pending, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                continue  # detected via get_state below
            for node, delta in chunk.items():
                if on_update:
                    on_update(node, delta or {})

        snapshot = graph.get_state(config)
        if not snapshot.next:
            return snapshot.values  # turn complete

        # Paused on an interrupt — pull its payload, ask the approver, resume.
        interrupt_value = None
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
        decision = approver(interrupt_value)
        pending = Command(resume=decision)


def _make_on_update(tracer, run_id, show_ui=True):
    def on_update(node, delta):
        tracer.log_event(run_id, node, delta)
        if show_ui:
            ui.show_node(node, delta)
            if delta.get("plan"):
                ui.show_plan(delta["plan"])

    return on_update


def _fresh_turn(state: AgentState, user_input: str) -> AgentState:
    """Append the new query and reset per-turn fields (accumulators + loop counter).
    `messages` persists across turns to keep in-process conversation memory."""
    state["messages"].append(HumanMessage(content=user_input))
    state["current_query"] = user_input
    state["current_response"] = ""
    state["context"] = ""
    state["plan"] = []
    state["iteration"] = 0
    state["verified"] = False
    state["verifier_feedback"] = ""
    state["tools_called"] = []
    state["tool_results"] = []
    state["documents_retrieved"] = []
    return state


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "current_query": "",
        "current_response": "",
        "context": "",
        "plan": [],
        "iteration": 0,
        "verified": False,
        "verifier_feedback": "",
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
    }


if __name__ == "__main__":
    # Populate the knowledge base once at startup. Non-fatal if it fails (e.g. embedding
    # model not pulled) — the search_knowledge_base tool will just return "no documents".
    try:
        build_ingest().invoke({"documents": []})
    except Exception as exc:
        ui.warn(f"knowledge-base ingest failed, continuing without RAG: {exc}")

    graph = build_agent()
    tracer = Tracer(DB_PATH)
    state = _initial_state()

    # Startup header — model / tool count / corpus size, like a tool's first line.
    from llms import llm
    from registry import tool as _tools
    from document_registry import DOCUMENTS_DIR

    n_docs = sum(
        1 for p in DOCUMENTS_DIR.glob("*") if p.is_file() and p.name != ".manifest.md"
    ) if DOCUMENTS_DIR.exists() else 0
    ui.banner(getattr(llm, "model", "unknown"), len(_tools), n_docs, DB_PATH)

    # Carries the live session into slash-command handlers. `make_initial_state` lets
    # /reset rebuild state without commands.py importing back into agent.py.
    cmd_ctx = commands.CommandContext(
        state=state, make_initial_state=_initial_state, db_path=DB_PATH
    )

    while True:
        user_input = ui.prompt(commands.command_names())

        # `/`-prefixed lines are REPL meta-commands, not agent turns — intercept them here.
        if commands.is_command(user_input):
            commands.dispatch(user_input, cmd_ctx)
            if cmd_ctx.should_quit:
                break
            state = cmd_ctx.state  # a command (e.g. /reset) may have swapped state out
            continue

        if not user_input.strip():
            continue

        state = _fresh_turn(state, user_input)
        # Fresh thread per turn: gives the approval interrupt a stable thread to pause/resume on,
        # while cross-turn memory rides on the manually-carried `messages`.
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        run_id = tracer.start_run(thread_id, user_input)
        ui.reset_turn()  # reset node-timing + plan-diff state for this turn's trace

        try:
            state = run_turn(
                graph,
                state,
                config,
                approver=ui.ask_approval,
                on_update=_make_on_update(tracer, run_id, show_ui=True),
            )
            tracer.end_run(run_id, "ok", state["messages"][-1].content)
        except Exception as exc:
            tracer.end_run(run_id, "error", str(exc))
            raise

        cmd_ctx.state = state  # keep the command context pointed at the latest state
        ui.response(state["messages"][-1].content)
