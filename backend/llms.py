from langchain_ollama import ChatOllama
from registry import tool


llm = ChatOllama(model="gemma4:31b")


llm_with_tools = llm.bind_tools(tool)
