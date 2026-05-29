from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

# import llm for llms file
from node_registry.llm_call import llm_call
from node_registry.synthesize import synthesize_node
from node_registry.plan import plan_node

# state
from state import AgentState

from tool import build_tool

tool_node = build_tool()

# from rag import

# libraries for structured output
from pydantic import BaseModel, Field


# build out the main graph
builder = StateGraph(AgentState)

# add nodes
builder.add_node("tool_node", tool_node)
builder.add_node("plane_node", plan_node)
builder.add_node("synthesize_node", synthesize_node)


# add edges
builder.add_edge(START, "plane_node")
builder.add_edge("plane_node", "tool_node")
builder.add_edge("tool_node", "synthesize_node")
builder.add_edge("synthesize_node", END)

graph = builder.compile()

state = AgentState(messages=[], initial_query=[])


# inf loop to allow for chat like experience
while True:
    user_input = input("User: ")

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "state":
        print(state)
        continue

    state["messages"].append({"role": "user", "content": user_input})
    state["initial_query"].append(user_input)

    result = graph.invoke(state)

    messages = result["messages"]

    print(f"Assistant: {messages[-1].content}")
