from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# import system messages from messages file
from messages import call_tool_msg
from messages import fetch_docs_msg
from messages import synthesize_output_msg
from messages import complexity_router_msg


# import llm for llms file
from llms import llm
from llms import llm_with_tools

# tools from tools file
from tools import tool

# libraries for structured output
from pydantic import BaseModel, Field

# todo: need to define custom state
# todo: need to create router function to route to different subgraphs based on the complexity of the request using structured output
# todo: need to create the workflow grpahs in seperate files and import them here
# todo: need to create the state checkpoint using langgraph and write to sqlite db


# import agentstate class from state file
from state import AgentState


tools_by_name = {t.name: t for t in tool}


# need to define structured output for the router node


# router node to route to three different subgraphs based on the complexity of the query
# need to utilize structured output and conditional edges to route to different subgraphs
def complexity_router(state: AgentState):
    pass


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


def light(state: AgentState):
    return {"messages": "light"}


def moderate(state: AgentState):
    return {"messages": "moderate"}


def complex(state: AgentState):
    return {"messages": "complex"}


"""
def fetch_docs(state: AgentState):
    relevant_docs = "my name is logan"
    return {"messages": relevant_docs}


def call_tools(state: AgentState):
    # double check this is correct syntax for appending system_message
    llm_response = llm_with_tools.invoke(state["messages"] + [call_tool_msg])
    return {"messages": llm_response}


def tool_node(state: AgentState):
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    return {"messages": result}


def synthesize_output(state: AgentState):
    llm_response = llm.invoke(state["messages"] + [synthesize_output_msg])
    return {"messages": llm_response}


def tools_necessary(state: AgentState):
    if state["messages"][-1].tool_calls:
        return True
    return False """


builder = StateGraph(AgentState)
# langgraph nodes
builder.add_node("complexity_router", complexity_router)

"""
builder.add_node("fetch_docs", fetch_docs)
builder.add_node("call_tools", call_tools)
builder.add_node("tool_node", tool_node)
builder.add_node("synthesize_output", synthesize_output)"""

builder.add_node("light", light)
builder.add_node("moderate", moderate)
builder.add_node("complex", complex)

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
