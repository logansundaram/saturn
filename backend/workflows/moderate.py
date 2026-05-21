# build and compile moderate subgraph to be routed to from main graph
from langgraph.graph import StateGraph, START, END, MessagesState

from llms import llm_with_tools, llm

from messages import light_llm_msg

from state import AgentState


def build_moderate():
    def fetch_docs(state: AgentState):
        relevant_docs = "my name is logan"
        return {"messages": relevant_docs}

    def call_tools(state: AgentState):
        # double check this is correct syntax for appending system_message
        llm_response = llm_with_tools.invoke(state["messages"] + [call_tool_msg])
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
        llm_response = llm.invoke(state["messages"] + [synthesize_output_msg])
        return {"messages": llm_response}

    moderate_builder = StateGraph(MessagesState)
    moderate_builder.add_node("llm_call", llm_call)
    moderate_builder.add_edge(START, "llm_call")
    moderate_builder.add_edge("llm_call", END)
    moderate_graph = moderate_builder.compile()
    return moderate_graph
