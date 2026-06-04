"""
Shared Tavily client for the web tools (web_search, web_extract, deep_research).

One cached client built lazily, and one place to fail clearly when TAVILY_API_KEY is
missing — instead of each tool re-running load_dotenv() and constructing a fresh client
per call (which buried a missing-key error deep inside three separate code paths).
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


@lru_cache(maxsize=1)
def get_tavily_client() -> TavilyClient:
    """Return a process-wide cached TavilyClient, or raise a clear error if the key is unset."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set — add it to your .env to use the web tools "
            "(web_search, web_extract, deep_research)."
        )
    return TavilyClient(api_key=api_key)
