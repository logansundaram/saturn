# file to add the tools for the agents to use
from langchain.tools import tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from langchain_community.agent_toolkits import FileManagementToolkit

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


@tool
def web_search(query: str):
    """Searches the web for the given query."""
    tavily_client = TavilyClient(api_key=tavily_api_key)
    response = tavily_client.search(query)
    return response


# should make it easy for end users to generate and supply their own tavily API key


# file tool suite

_file_toolkit = FileManagementToolkit(
    root_dir="./database/documents",
    selected_tools=["read_file", "write_file", "list_directory"],
)
file_tools = _file_toolkit.get_tools()


# math tool suite


@tool
def addition(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def subtraction(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b


@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def division(a: int, b: int) -> int:
    """Divides a by b."""
    return a / b


tool = [addition, subtraction, multiplication, division, web_search, file_tools]

tools_by_name = {t.name: t for t in tool}
