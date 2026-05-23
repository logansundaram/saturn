# web search tool. one high level tool at first
from langchain.tools import tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient

"""
  Tier 1 — Core utility (build these first)
  - Web search (Tavily or DuckDuckGo) — covers most "I don't know" cases without RAG
  - Code execution (subprocess sandbox) — lets the agent run and verify code it writes
  - File read/write — needed for almost any real task involving documents or projects

  Tier 2 — Makes RAG actually useful
  - URL/webpage scraper — ingest arbitrary web content into your vector store
  - PDF/text loader — ingest local documents into the vector store you just built

  Tier 3 — Power tools
  - Shell/terminal — for system tasks, running builds, git operations
  - Python REPL — heavier than subprocess but stateful across calls, good for data work
"""

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
print(tavily_api_key)

# web tool suite

# docstring of the tool is automatically exposed to the llm, that is how it determines what tool to call


@tool
def web_search(query: str):
    """Searches the web for the given query."""
    tavily_client = TavilyClient(api_key=tavily_api_key)
    response = tavily_client.search(query)
    return response
