from langchain_ollama import ChatOllama
from tools import tool


llm = ChatOllama(model="gemma4:e4b")


llm_with_tools = llm.bind_tools(tool)
