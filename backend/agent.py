from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional

call_tool = SystemMessage(content="Call the relevante tools based on the user request")

fetch_docs = SystemMessage(content="Fetch the relevant documents based on the user request") 

synthesize_output = SystemMessage(content="Synthesize the output based on the user request")

llm = ChatOllama(model="qwen3.6:35b")

@tool
def addition(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


llm_with_tools = llm.bind_tools([addition])


def fetch_docs(state: MessagesState):
    relevant_docs = "my name is logan"
    return {"messages": relevant_docs}

def call_tools(state: MessagesState):
    #double check this is correct syntax for appending system_message
    llm_response = llm_with_tools.invoke(state["messages"] + call_tool)
    return {"messages" : llm_response}

def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def synthesize_output(state: MessagesState):
    llm_response = llm.invoke(state["messages"].append(call_tool))
    return {"messages" : llm_response}







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