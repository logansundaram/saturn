# build and compile moderate subgraph to be routed to from main graph
from langgraph.graph import StateGraph, START, END, MessagesState

from llms import llm_with_tools, llms

from messages import light_llm_msg


def build_moderate():
    def llm_call(state: MessagesState):
        return llm_with_tools.invoke(state["messages"] + [light_llm_msg])

    light_builder = StateGraph(MessagesState)
    light_builder.add_node("llm_call", llm_call)
    light_builder.add_edge(START, "llm_call")
    light_builder.add_edge("llm_call", END)
    light_graph = light_builder.compile()

    moderate_builder = StateGraph(MessagesState)
    moderate_builder.add_node("llm_call", llm_call)
    moderate_builder.add_edge(START, "llm_call")
    moderate_builder.add_edge("llm_call", END)
    moderate_graph = moderate_builder.compile()
    return moderate_graph
