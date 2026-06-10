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
import diag

import httpx
import trafilatura
from ddgs import DDGS
from dotenv import load_dotenv
from tavily import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)

from config import get_config
from toolspec import register_tool

load_dotenv()

# Seconds between status checks while a (Tavily) deep_research job runs.
_POLL_INTERVAL = 3

# Hard ceiling (seconds) on how long we wait for a Tavily research job before giving up and
# falling back to the local research loop. Without it the poll below is an unbounded `while True`
# that hangs the whole turn forever if the job never reaches "completed" (stuck or failed job).
_RESEARCH_TIMEOUT_DEFAULT = 180


def _research_timeout() -> float:
    """Max seconds to wait on a Tavily research job (config `web.deep_research_timeout`)."""
    try:
        return float(get_config().get("web.deep_research_timeout", _RESEARCH_TIMEOUT_DEFAULT))
    except (TypeError, ValueError):
        return float(_RESEARCH_TIMEOUT_DEFAULT)

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
        diag.log(f"[web] Tavily unavailable ({type(err).__name__}); falling back to keyless DuckDuckGo.")
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
@register_tool("read_only")
def web_search(query: str):
    """Execute a web search query. Uses Tavily when a key is configured, otherwise falls back
    to keyless DuckDuckGo automatically — no API key required."""
    start = time.perf_counter()
    try:
        if _use_tavily():
            try:
                return _client().search(query, max_results=_max_results())
            except _TAVILY_FALLBACK_ERRORS as err:
                _disable_tavily(err)  # dead key/quota — keyless for the rest of the session
            except Exception as err:
                # Any other Tavily failure (network blip, 5xx, odd response shape) falls back to
                # the keyless backend for THIS call only — the key may be fine, so it isn't
                # disabled for the session. A flaky Tavily must never cost an answer DuckDuckGo
                # could have given (deep_research already degrades the same way).
                diag.log(f"[web] Tavily search failed ({type(err).__name__}); DuckDuckGo fallback")
        return _ddg_search(query, _max_results())
    finally:
        diag.log(f"web_search : {time.perf_counter() - start:.4f}s")


@register_tool("read_only")
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
            except Exception as err:
                # Transient Tavily failure: degrade to the local extractor for this call only
                # (mirrors web_search — the key may be fine, so don't disable it).
                diag.log(f"[web] Tavily extract failed ({type(err).__name__}); local fallback")
        # Local-first path: handle a single URL or a list of URLs.
        urls = [u for u in (url if isinstance(url, (list, tuple)) else [url]) if u]
        if not urls:
            return "No URL provided to extract."
        results = {u: _local_extract(u) for u in urls}
        return results if len(results) > 1 else next(iter(results.values()))
    finally:
        diag.log(f"web_extract : {time.perf_counter() - start:.4f}s")


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


@register_tool("side_effecting")
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
                # Bounded poll: give up at the deadline or on a terminal failure status, then fall
                # through to the keyless local loop — never spin forever on a stuck/failed job.
                deadline = time.perf_counter() + _research_timeout()
                while time.perf_counter() < deadline:
                    status_response = client.research_get(request_id)
                    status = status_response.get("status")
                    if status == "completed":
                        return status_response["response"]
                    if status in ("failed", "error", "cancelled"):
                        diag.log(f"deep_research : Tavily job {status}; falling back to local")
                        break
                    time.sleep(_POLL_INTERVAL)
                else:
                    diag.log("deep_research : Tavily research timed out; falling back to local")
            except _TAVILY_FALLBACK_ERRORS as err:
                _disable_tavily(err)  # key/quota dead — fall through to the local loop below
            except Exception as err:
                # Any other Tavily failure (unexpected response shape, network drop) must not strand
                # the turn — log and fall through to the keyless local research loop.
                diag.log(f"deep_research : Tavily research failed ({type(err).__name__}); local fallback")
        return _local_deep_research(query)
    finally:
        diag.log(f"deep_research : {time.perf_counter() - start:.4f}s")
