from registry import tools_by_name
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from llms import llm_with_tools
from state import AgentState
from messages import medium_call_tool_msg
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.types import interrupt, Command

# should  have benchmark to measure the overhead of langgraph and agent architecture vs direct llm call


def build_tool():
    def call_tools(state: AgentState):
        print(state["messages"])

        llm_response = llm_with_tools.invoke(state["messages"] + [medium_call_tool_msg])

        print("calling tool")
        print(llm_response)

        # Keep this in messages because it contains .tool_calls
        return {"messages": [llm_response]}

    def approval_node(
        state: AgentState,
    ):
        pass
        # Pause execution; payload shows up in result.interrupts (v2) or result["__interrupt__"] (v1) is_approved = interrupt( { "question": "Do you want to proceed with this action?", "details": state["messages"], } ) # Route based on the response if is_approved: return Command( goto="tool_node" ) # Runs after the resume payload is provided else: print("Execution of the tool cancelled") return Command(goto=END) # Cancel the exection

    def tools_necessary(state: AgentState):
        if state["messages"][-1].tool_calls:
            print("tool is necessary")
            return True
        return False

    def tool_node(state: AgentState):
        tools_called = []
        tool_results = []

        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            selected_tool = tools_by_name[tool_name]
            observation = selected_tool.invoke(tool_args)

            tools_called.append(tool_name)
            tool_results.append(str(observation))

            print(observation)

        return {
            "tools_called": tools_called,
            "tool_results": tool_results,
        }

    tool_builder = StateGraph(AgentState)

    tool_builder.add_node("call_tools", call_tools)
    tool_builder.add_node("tool_node", tool_node)
    tool_builder.add_node("approval_node", approval_node)

    tool_builder.add_edge(START, "call_tools")

    tool_builder.add_conditional_edges(
        "call_tools",
        tools_necessary,
        {
            True: "tool_node",
            False: END,
        },
    )

    tool_builder.add_edge("tool_node", END)

    tool_graph = tool_builder.compile()
    return tool_graph
