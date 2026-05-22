from registry import tools_by_name
from langgraph import StateGraph, START, END
from llms import llm_with_tools
from state import AgentState
from messages import medium_call_tool_msg
from langchain.messages import ToolMessage


def exectue_tool_call(state: AgentState):
    def call_tools(state: AgentState):
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

    tool_builder = StateGraph(AgentState)
    tool_builder.add_node("call_tools", call_tools)
    tool_builder.add_node("tool_node", tool_node)
    tool_builder.add_conditional_edges(
        "call_tools", tools_necessary, {True: "tool_node", False: "synthesize_output"}
    )
    tool_builder.add_edge("tool_node", "call_tools")
    tool_builder.add_edge(START, "call_tools")
    tool_builder.add_edge("call_tools", END)
    tool_graph = tool_builder.compile()
    return tool_graph
