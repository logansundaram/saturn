from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama


llm = ChatOllama(model="qwen3.6:35b")


def respond(state: MessagesState):
    ai_response = llm.invoke(state["messages"])
    return {"messages": [ai_response]}


builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile()

messages = []

while True:
    user_input = input("User: ")

    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    result = graph.invoke({"messages": messages})

    messages = result["messages"]

    print(f"Assistant: {messages[-1].content}")