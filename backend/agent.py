from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# import system messages from messages file
from messages import complexity_router_msg


# import llm for llms file
from llms import llm


# libraries for structured output
from pydantic import BaseModel, Field

# todo: need to define custom state
# todo: need to create router function to route to different subgraphs based on the complexity of the request using structured output
# todo: need to create the workflow grpahs in seperate files and import them here
# todo: need to create the state checkpoint using langgraph and write to sqlite db


# import agentstate class from state file
from state import AgentState


# import subgraphs
from workflows.light import build_light
from workflows.moderate import build_moderate
from workflows.complex import build_complex

# create subgraphs
light_graph = build_light()
moderate_graph = build_moderate()
complex_graph = build_complex()


# router node to route to three different subgraphs based on the complexity of the query
# need to utilize structured output and conditional edges to route to different subgraphs
def complexity_router(state: AgentState):
    print("routing to subgraph based on complexity of query")


# need to define structured output for the router node
class RouterOutput(BaseModel):
    # need to define the output structure for the router node
    complexity: int = Field(description="0 for light, 1 for moderate, 2 for complex")


def complexity_router_function(state: AgentState):
    # lightweight llm currently, could use an ml model or a hyperspecific llm for this task to reduce overhead
    # should make it configurable as well
    llm_with_router_structure = llm.with_structured_output(RouterOutput)
    complexity = int(
        llm_with_router_structure.invoke(
            state["messages"] + [complexity_router_msg]
        ).complexity
    )
    print(complexity)
    print(type(complexity))
    # switch llm with structured output here to make sure the output is always a single digit between 0-2
    return complexity


# build out the main graph
builder = StateGraph(AgentState)

builder.add_node("complexity_router", complexity_router)
builder.add_node("light", light_graph)
builder.add_node("moderate", moderate_graph)
builder.add_node("complex", complex_graph)

builder.add_edge(START, "complexity_router")


# build out graph edges
# should be fetch docs -> call tools -> execute tools -> synthesize output
# basic proof of concept agent
builder.add_conditional_edges(
    "complexity_router",  # source node
    complexity_router_function,  # routing function
    {0: "light", 1: "moderate", 2: "complex"},  # mapping of return values to node names
)

# condiitonal edge if tools calls are needed
"""
builder.add_conditional_edges(
    "call_tools", tools_necessary, {True: "tool_node", False: "synthesize_output"}
)
builder.add_edge("tool_node", "synthesize_output")
builder.add_edge("fetch_docs", "call_tools")

"""
builder.add_edge("moderate", END)
builder.add_edge("complex", END)
builder.add_edge("light", END)

graph = builder.compile()

messages = []


# inf loop to allow for chat like experience
while True:
    user_input = input("User: ")

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "state":
        print(messages)
        continue

    messages.append({"role": "user", "content": user_input})

    result = graph.invoke({"messages": messages})

    messages = result["messages"]

    print(f"Assistant: {messages[-1].content}")
