import time

from langchain.tools import tool

from tool_registry._tavily import get_tavily_client

POLL_INTERVAL = 3        # seconds between status checks
MAX_WAIT_SECONDS = 600   # give up rather than hang the whole agent turn forever


@tool
def deep_research(query: str):
    """Performs a deep research on the given query. Returns a comprehensive research report. This tool is used for in-depth research on a topic. It is a more advanced version of the web search tool. It is used to get a comprehensive understanding of a topic. It is used to get a detailed analysis"""
    client = get_tavily_client()

    job = client.research(input=query, model="pro")
    request_id = job["request_id"]

    # Bound the poll loop: a job that never completes must not block the turn indefinitely.
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        status_response = client.research_get(request_id)
        if status_response["status"] == "completed":
            return status_response["response"]
        time.sleep(POLL_INTERVAL)

    return (
        f"Deep research timed out after {MAX_WAIT_SECONDS}s without completing. "
        "Try a narrower query, or fall back to a single web_search."
    )
