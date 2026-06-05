"""
Web tools — everything that reaches the live internet, grouped behind one cached Tavily client.

  web_search    — a single Tavily search query.
  web_extract   — fetch + extract the readable content behind one or more URLs.
  deep_research — heavyweight multi-source Tavily research job (slow/costly).

These all share the same Tavily account, so they share one lazily-built client (`_client()`)
instead of each re-reading the key and re-instantiating per call.
"""

import os
import time

from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

# Seconds between status checks while a deep_research job runs.
_POLL_INTERVAL = 3

_TAVILY = None


def _client() -> TavilyClient:
    """The shared Tavily client, built on first use from TAVILY_API_KEY."""
    global _TAVILY
    if _TAVILY is None:
        _TAVILY = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _TAVILY


@tool
def web_search(query: str):
    """Execute a web search query using Tavily Search."""
    start = time.perf_counter()
    try:
        return _client().search(query)
    finally:
        print(f"web_search : {time.perf_counter() - start:.4f}s")


@tool
def web_extract(url: str):
    """Extract the readable page content behind one or more URLs using Tavily Extract. Pass a
    single URL string, or a list of URLs. Use this to read a specific page web_search surfaced."""
    start = time.perf_counter()
    try:
        return _client().extract(url)
    finally:
        print(f"web_extract : {time.perf_counter() - start:.4f}s")


@tool
def deep_research(query: str):
    """Performs deep research on the given query and returns a comprehensive research report.
    A more advanced, multi-source version of web_search for getting a thorough, detailed
    analysis of a topic. Slow and costly — use only when a single web_search will not suffice."""
    start = time.perf_counter()
    client = _client()
    job = client.research(input=query, model="pro")
    request_id = job["request_id"]

    while True:
        status_response = client.research_get(request_id)
        if status_response["status"] == "completed":
            print(f"deep_research : {time.perf_counter() - start:.4f}s")
            return status_response["response"]
        time.sleep(_POLL_INTERVAL)
