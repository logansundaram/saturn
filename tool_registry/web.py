"""
Web tools — everything that reaches the live internet.

  web_search    — a single web search query.
  web_extract   — fetch + extract the readable content behind a URL.
  deep_research — multi-source research report.

Key-optional, local-first provider strategy
--------------------------------------------
None of these *require* a Tavily API key. Tavily is treated as a quality upgrade, not a hard
dependency, so the agent stays useful when the user has no key or has run out of usage.

  web_search    resolves a backend via `_use_tavily()`. With `web.provider: auto` (the default)
                it prefers Tavily when a healthy TAVILY_API_KEY is present and transparently
                falls back to keyless DuckDuckGo (`ddgs`) on a missing key or a quota/usage
                error. Once Tavily fails that way it is disabled for the rest of the session
                (`_TAVILY_DISABLED`) so we don't re-hit a dead key every turn.
  web_extract   is local-first: it fetches + extracts readable text with `trafilatura`, no key
                and no API call. (Tavily Extract is only used if `web.provider: tavily` is
                forced and a key is present.)
  deep_research uses Tavily's research job when Tavily is available; otherwise it reimplements
                the same shape locally — web_search -> read the top hits -> synthesize with the
                local `synthesizer` model. Slower, but keyless and free.

Provider selection lives in `config.yaml` under `web:` (`provider`, `max_results`,
`deep_research_sources`); nothing is hard-coded here.
"""

import os
import time

import httpx
import trafilatura
from ddgs import DDGS
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)

from config import get_config

load_dotenv()

# Seconds between status checks while a (Tavily) deep_research job runs.
_POLL_INTERVAL = 3

# Tavily errors that mean "this key is unusable" — they trigger the keyless fallback and
# disable Tavily for the rest of the session.
_TAVILY_FALLBACK_ERRORS = (MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError)

_TAVILY = None
# Set once a Tavily call fails with a key/quota error so we stop retrying a dead key this session.
_TAVILY_DISABLED = False


# --- provider resolution ---------------------------------------------------
def _provider() -> str:
    """Configured web provider: 'auto' (default), 'tavily', or 'duckduckgo'."""
    return str(get_config().get("web.provider", "auto")).lower()


def _max_results() -> int:
    return int(get_config().get("web.max_results", 5))


def _use_tavily() -> bool:
    """Whether the current call should go through Tavily. False forces the keyless path."""
    if _TAVILY_DISABLED:
        return False
    provider = _provider()
    if provider == "duckduckgo":
        return False
    if provider == "tavily":
        return True
    # auto: use Tavily only when a key is actually present.
    return bool(os.getenv("TAVILY_API_KEY"))


def _client() -> TavilyClient:
    """The shared Tavily client, built on first use from TAVILY_API_KEY."""
    global _TAVILY
    if _TAVILY is None:
        _TAVILY = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _TAVILY


def reset_web_clients() -> None:
    """Drop the cached Tavily client and the session disable flag so a key change (via
    `/config key`) is picked up immediately, without restarting. Mirrors `llms.reset_models()`."""
    global _TAVILY, _TAVILY_DISABLED
    _TAVILY = None
    _TAVILY_DISABLED = False


def _disable_tavily(err: Exception) -> None:
    """Mark Tavily unusable for the rest of the session and explain why (once)."""
    global _TAVILY_DISABLED
    if not _TAVILY_DISABLED:
        print(f"[web] Tavily unavailable ({type(err).__name__}); falling back to keyless DuckDuckGo.")
    _TAVILY_DISABLED = True


# --- keyless backends ------------------------------------------------------
def _ddg_search(query: str, max_results: int) -> dict:
    """Keyless web search via DuckDuckGo, normalized to the Tavily-style result shape the
    synthesize node already understands ({'query', 'results': [{title, url, content}]})."""
    hits = DDGS().text(query, max_results=max_results)
    return {
        "query": query,
        "provider": "duckduckgo",
        "results": [
            {"title": h.get("title"), "url": h.get("href"), "content": h.get("body")}
            for h in hits
        ],
    }


def _local_extract(url: str) -> str:
    """Keyless page-content extraction: fetch then pull readable text with trafilatura,
    falling back to trafilatura's own fetch if the direct GET is blocked."""
    html = None
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=20.0,
                         headers={"User-Agent": "Mozilla/5.0 (Saturday.ai)"})
        resp.raise_for_status()
        html = resp.text
    except Exception:
        html = trafilatura.fetch_url(url)
    if not html:
        return f"[could not fetch {url}]"
    text = trafilatura.extract(html, include_links=False, include_comments=False)
    return text or f"[no readable content extracted from {url}]"


# --- tools -----------------------------------------------------------------
@tool
def web_search(query: str):
    """Execute a web search query. Uses Tavily when a key is configured, otherwise falls back
    to keyless DuckDuckGo automatically — no API key required."""
    start = time.perf_counter()
    try:
        if _use_tavily():
            try:
                return _client().search(query, max_results=_max_results())
            except _TAVILY_FALLBACK_ERRORS as err:
                _disable_tavily(err)  # fall through to the keyless backend below
        return _ddg_search(query, _max_results())
    finally:
        print(f"web_search : {time.perf_counter() - start:.4f}s")


@tool
def web_extract(url: str):
    """Extract the readable page content behind a URL. Use this to read a specific page that
    web_search surfaced. Runs locally (trafilatura) with no API key by default; only uses
    Tavily Extract when the web provider is explicitly forced to 'tavily'."""
    start = time.perf_counter()
    try:
        if _provider() == "tavily" and _use_tavily():
            try:
                return _client().extract(url)
            except _TAVILY_FALLBACK_ERRORS as err:
                _disable_tavily(err)
        # Local-first path: handle a single URL or a list of URLs.
        urls = url if isinstance(url, (list, tuple)) else [url]
        results = {u: _local_extract(u) for u in urls}
        return results if len(results) > 1 else next(iter(results.values()))
    finally:
        print(f"web_extract : {time.perf_counter() - start:.4f}s")


def _local_deep_research(query: str) -> str:
    """Keyless deep_research: search, read the top hits, and synthesize a report with the
    local `synthesizer` model. Mirrors the shape of Tavily's research job without a key."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from llms import get_model

    n = int(get_config().get("web.deep_research_sources", 4))
    hits = _ddg_search(query, max_results=n).get("results", [])
    sources = []
    for h in hits:
        body = _local_extract(h["url"]) if h.get("url") else (h.get("content") or "")
        sources.append(f"## {h.get('title')}\n{h.get('url')}\n\n{body[:4000]}")
    corpus = "\n\n---\n\n".join(sources) or "(no sources retrieved)"

    msgs = [
        SystemMessage(content=(
            "You are a research assistant. Write a comprehensive, well-structured report that "
            "answers the user's query using ONLY the provided sources. Cite source URLs inline. "
            "If the sources are insufficient, say so plainly."
        )),
        HumanMessage(content=f"Query: {query}\n\nSOURCES:\n{corpus}"),
    ]
    return get_model("synthesizer").invoke(msgs).content


@tool
def deep_research(query: str):
    """Performs deep research on the given query and returns a comprehensive research report.
    A more advanced, multi-source version of web_search for a thorough, detailed analysis.
    Slow and costly — use only when a single web_search will not suffice. Uses Tavily's
    research job when a key is configured, otherwise runs a keyless local research loop."""
    start = time.perf_counter()
    try:
        if _use_tavily():
            try:
                client = _client()
                job = client.research(input=query, model="pro")
                request_id = job["request_id"]
                while True:
                    status_response = client.research_get(request_id)
                    if status_response["status"] == "completed":
                        return status_response["response"]
                    time.sleep(_POLL_INTERVAL)
            except _TAVILY_FALLBACK_ERRORS as err:
                _disable_tavily(err)  # fall through to the local loop below
        return _local_deep_research(query)
    finally:
        print(f"deep_research : {time.perf_counter() - start:.4f}s")
