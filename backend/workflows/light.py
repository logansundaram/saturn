# build and compile light subgraph to be routed to from main graph
from langgraph.graph import StateGraph, START, END, MessagesState


def llm_call(state: MessagesState):
    pass


builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", END)
graph = builder.compile()
