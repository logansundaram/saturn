from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import List, Dict, Any, Optional

call_tool_msg = SystemMessage(content="Call the relevante tools based on the user request")

fetch_docs_msg = SystemMessage(content="Fetch the relevant documents based on the user request") 

synthesize_output_msg = SystemMessage(content="Synthesize the output based on the user request")

llm = ChatOllama(model="qwen3.6:35b")

@tool
def addition(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b



def fetch_docs(state: MessagesState):
    relevant_docs = "my name is logan"
    return {"messages": relevant_docs}

def call_tools(state: MessagesState):
    #double check this is correct syntax for appending system_message
    llm_response = llm_with_tools.invoke(state["messages"] + [call_tool_msg])
    return {"messages" : llm_response}

tools = [addition]
tools_by_name = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

def tool_node(state: MessagesState):
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": result}

def synthesize_output(state: MessagesState):
    llm_response = llm.invoke(state["messages"] + [synthesize_output_msg])
    return {"messages" : llm_response}



builder = StateGraph(MessagesState)
#langgraph nodes
builder.add_node("fetch_docs", fetch_docs)
builder.add_node("call_tools", call_tools)
builder.add_node("tool_node", tool_node)
builder.add_node("synthesize_output", synthesize_output)

#build out graph edges
#should be fetch docs -> call tools -> execute tools -> synthesize output
#basic proof of concept agent
builder.add_edge(START, "fetch_docs")
builder.add_edge("fetch_docs", "call_tools")
builder.add_edge("call_tools", "tool_node")
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