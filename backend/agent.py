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

# import llm for llms file
from llms import llm
from llms import llm_with_tools

# tools from tools file
from tools import tool


# todo: need to define custom state
# todo: need to create router function to route to different subgraphs based on the complexity of the request using structured output
# todo: need to create the workflow grpahs in seperate files and import them here
# todo: need to create the state checkpoint using langgraph and write to sqlite db


# import agentstate class from state file
from state import AgentState


tools_by_name = {t.name: t for t in tool}


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
    return False


builder = StateGraph(AgentState)
# langgraph nodes
builder.add_node("fetch_docs", fetch_docs)
builder.add_node("call_tools", call_tools)
builder.add_node("tool_node", tool_node)
builder.add_node("synthesize_output", synthesize_output)

# build out graph edges
# should be fetch docs -> call tools -> execute tools -> synthesize output
# basic proof of concept agent
builder.add_edge(START, "fetch_docs")
builder.add_edge("fetch_docs", "call_tools")
# condiitonal edge if tools calls are needed
builder.add_conditional_edges(
    "call_tools", tools_necessary, {True: "tool_node", False: "synthesize_output"}
)
builder.add_edge("tool_node", "synthesize_output")
builder.add_edge("synthesize_output", END)

graph = builder.compile()

messages = []

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
