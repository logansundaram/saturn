from langchain.tools import tool

from tool_registry._tavily import get_tavily_client

# Outsource the ugly web plumbing: search, crawl, scrape, extract. Keep ownership of planning,
# state, source selection, verification, caching, and final synthesis.


@tool
def web_search(query: str):
    """Execute a web search query using Tavily Search."""
    return get_tavily_client().search(query)
