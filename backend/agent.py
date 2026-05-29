from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

# import llm for llms file
from node_registry.plan import plan_node
from node_registry.synthesize import synthesize_node
from node_registry.verifier import verifier_node
from node_registry.repair import repair_node

# state
from state import AgentState

from tool import build_tool

tool_node = build_tool()

# from rag import
from rag import build_retrieval

rag_node = build_retrieval()

# libraries for structured output
from pydantic import BaseModel, Field


# build out the main graph
builder = StateGraph(AgentState)

# add nodes
builder.add_node("plan", plan_node)
builder.add_node("rag", rag_node)
builder.add_node("tool", tool_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("verifier", verifier_node)
builder.add_node("repair", repair_node)


# add edges
builder.add_edge(START, "plan")
builder.add_edge("plan", "rag")
builder.add_edge("rag", "tool")
builder.add_edge("tool", "synthesize")

# need to add conditional edges
builder.add_edge("synthesize", "verifier")
builder.add_edge("verifier", "repair")
builder.add_edge("repair", END)


graph = builder.compile()

state: AgentState = {
    "messages": [],
    "current_query": "",
    "current_response": "",
    "tools_called": [],
    "tool_results": [],
    "context": [],
}

# inf loop to allow for chat like experience
while True:
    user_input = input("User: ")

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "state":
        print(state)
        continue

    state["messages"].append(HumanMessage(content=user_input))

    state["current_query"] = user_input

    state["context"] = []
    state["tool_results"] = []

    # IMPORTANT: save returned state
    state = graph.invoke(state)

    messages = state["messages"]
    last_msg = messages[-1]

    print(f"Assistant: {last_msg.content}")
