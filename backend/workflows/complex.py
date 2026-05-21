# build and compile complex subgraph to be routed to from main graph
# build and compile moderate subgraph to be routed to from main graph
from langgraph.graph import StateGraph, START, END, MessagesState

from llms import llm_with_tools, llm

from messages import (
    medium_call_tool_msg,
    medium_fetch_docs_msg,
    medium_synthesize_output_msg,
)

from state import AgentState

from langchain.messages import ToolMessage

from tools import tools_by_name


def build_complex():
    def fetch_docs(state: AgentState):
        relevant_docs = "my name is logan"
        return {"messages": relevant_docs}

    def call_tools(state: AgentState):
        # double check this is correct syntax for appending system_message
        llm_response = llm_with_tools.invoke(state["messages"] + [medium_call_tool_msg])
        return {"messages": llm_response}

    def tools_necessary(state: AgentState):
        if state["messages"][-1].tool_calls:
            return True
        return False

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
        llm_response = llm.invoke(state["messages"] + [medium_synthesize_output_msg])
        return {"messages": llm_response}

    # instantiate builder
    moderate_builder = StateGraph(MessagesState)

    # add nodes
    moderate_builder.add_node("fetch_docs", fetch_docs)
    moderate_builder.add_node("call_tools", call_tools)
    moderate_builder.add_node("tool_node", tool_node)
    moderate_builder.add_node("synthesize_output", synthesize_output)

    # add edges
    moderate_builder.add_edge(START, "fetch_docs")
    moderate_builder.add_edge("fetch_docs", "call_tools")
    moderate_builder.add_conditional_edges(
        "call_tools", tools_necessary, {True: "tool_node", False: "synthesize_output"}
    )
    moderate_builder.add_edge("tool_node", "synthesize_output")
    moderate_builder.add_edge("fetch_docs", "call_tools")
    moderate_builder.add_edge("synthesize_output", END)

    moderate_graph = moderate_builder.compile()
    return moderate_graph
