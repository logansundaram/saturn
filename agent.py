import sys

# Force UTF-8 console output. Node prints (plan glyphs, tool results, model output) routinely
# contain non-cp1252 characters that crash print() on the default Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage

from state import AgentState

# loop nodes
from node_registry.context_builder import context_builder_node
from node_registry.plan import plan_node
from node_registry.synthesize import synthesize_node
from node_registry.reflect import update_plan_node
from tool import agent_node, tool_node, route_after_agent

# RAG ingest (populates the in-memory vector store the search_knowledge_base tool reads)
from rag import build_ingest


def build_agent():
    """Assemble the living-plan ReAct loop:

        START -> ground -> plan -> agent -> (tools -> update_plan -> agent)* -> synthesize -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("ground", context_builder_node)
    builder.add_node("plan", plan_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("update_plan", update_plan_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "ground")
    builder.add_edge("ground", "plan")
    builder.add_edge("plan", "agent")

    # ReAct branch: act on tools, or finish.
    builder.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "synthesize": "synthesize"}
    )
    builder.add_edge("tools", "update_plan")
    builder.add_edge("update_plan", "agent")

    builder.add_edge("synthesize", END)

    return builder.compile()


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


if __name__ == "__main__":
    # Populate the knowledge base once at startup. Non-fatal if it fails (e.g. embedding
    # model not pulled) — the search_knowledge_base tool will just return "no documents".
    try:
        build_ingest().invoke({"documents": []})
    except Exception as exc:
        print(f"[warn] knowledge-base ingest failed, continuing without RAG: {exc}")

    graph = build_agent()

    state: AgentState = {
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

    while True:
        user_input = input("User: ")

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "state":
            print(state)
            continue

        state = _fresh_turn(state, user_input)
        state = graph.invoke(state)

        print(f"Assistant: {state['messages'][-1].content}")
