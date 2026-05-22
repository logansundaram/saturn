from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


# import system messages from messages file
from messages import complexity_router_msg, agent_verifier_msg


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


# need to define structered output for the verifier node
class VerifierOutput(BaseModel):
    # need to define the output structure for the verifier node
    valid: bool = Field(
        description="True if the output is valid and answers the initial query, False otherwise"
    )


def complexity_router_function(state: AgentState):
    # lightweight llm currently, could use an ml model or a hyperspecific llm for this task to reduce overhead
    # should make it configurable as well
    # should mkae more complex, give a frameowkr to the llm to deduce the complexity of the query
    # can add a conifdence rating, default to moderate complexity if confidence is too low
    # can move to an external file as it grows in code and complexity
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


def agent_verfier(state: AgentState):
    # verify the output of the agent and make sure it is correct and complete
    # can add a verification step here to make sure the output is correct and complete
    # can add a feedback loop to improve the agent over time
    # can escalate to a different subgraph if the output is not correct or complete
    print("this is the agent verfier")
    llm_with_verifier_structure = llm.with_structured_output(VerifierOutput)
    # could formatted string here later
    llm_response = llm_with_verifier_structure.invoke(
        "initial query: "
        + state["initial_query"][-1]
        + "output: "
        + state["messages"][-1].content
        + agent_verifier_msg.content
    )
    if llm_response:
        print("passed")
    else:
        print("failed")
    pass


# build out the main graph
builder = StateGraph(AgentState)

builder.add_node("complexity_router", complexity_router)
builder.add_node("light", light_graph)
builder.add_node("moderate", moderate_graph)
builder.add_node("complex", complex_graph)
builder.add_node("agent_verifier", agent_verfier)


builder.add_edge(START, "complexity_router")


# build out graph edges
# should be fetch docs -> call tools -> execute tools -> synthesize output
# basic proof of concept agent
builder.add_conditional_edges(
    "complexity_router",  # source node
    complexity_router_function,  # routing function
    {0: "light", 1: "moderate", 2: "complex"},  # mapping of return values to node names
)
builder.add_edge("moderate", "agent_verifier")
builder.add_edge("complex", "agent_verifier")
builder.add_edge("light", "agent_verifier")

builder.add_edge("agent_verifier", END)
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
