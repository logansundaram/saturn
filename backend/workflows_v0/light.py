"""
Deprecated file, kept for reference only
"""

# build and compile light subgraph to be routed to from main graph
from langgraph.graph import StateGraph, START, END, MessagesState

from llms import llm

from messages import light_llm_msg

from state import AgentState


def build_light():
    def llm_call(state: AgentState):
        llm_response = llm.invoke(state["messages"] + [light_llm_msg])
        print(llm_response)

        # not properly handling appending llm response as well as state

        return {"messages": llm_response}

    light_builder = StateGraph(AgentState)
    light_builder.add_node("llm_call", llm_call)
    light_builder.add_edge(START, "llm_call")
    light_builder.add_edge("llm_call", END)
    light_graph = light_builder.compile()
    return light_graph
