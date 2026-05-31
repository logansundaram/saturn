from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

# import llm for llms file
from node_registry.context_builder import context_builder_node
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


from debug import print_graph

# build out the main graph
builder = StateGraph(AgentState)

# add nodes
builder.add_node("context_builder", context_builder_node)
builder.add_node("plan", plan_node)
builder.add_node("rag", rag_node)
builder.add_node("tool", tool_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("verifier", verifier_node)
builder.add_node("repair", repair_node)


def determine_tool(state: AgentState):
    return state["tool_results"]


def determine_rag(state: AgentState):
    return state["rag_necessary"]


def determine_repair(state: AgentState):
    # some function to determine if past results, tools calls, document6s retrieved, syntheisis, etc are incomplete and wrong
    # need to flesh out this part
    return True


# add edges
builder.add_edge(START, "context_builder")
builder.add_edge("context_builder", "plan")
# parallel execution
builder.add_conditional_edges("plan", determine_rag, {True: "rag", False: "synthesize"})

builder.add_conditional_edges(
    "plan", determine_tool, {True: "tool", False: "synthesize"}
)

builder.add_edge("rag", "synthesize")
builder.add_edge("tool", "synthesize")

# loop logic

builder.add_edge("synthesize", "verifier")
builder.add_conditional_edges(
    "verifier", determine_repair, {True: "repair", False: END}
)
builder.add_edge("repair", "plan")


graph = builder.compile()

# should visualize the graph
print_graph(graph)


# class AgentState(TypedDict):
#     messages: Annotated[List[Any], add_messages]
#     current_query: str
#     current_response: str
#     tools_called: List[str]
#     tool_results: List[Any]
#     context: List[str]
# tools_necessary: bool
# rag_necessary: bool

state: AgentState = {
    "messages": [],
    "current_query": "",
    "current_response": "",
    "tools_called": [],
    "tool_results": [],
    "context": [],
    "tools_necessary": False,
    "rag_necessary": False,
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
