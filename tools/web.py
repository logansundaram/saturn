"""
Web tools — everything that reaches the live internet.

  web_search    — a single web search query.
  web_extract   — fetch + extract the readable content behind a URL.
  http_request  — one HTTP request to any URL/API; registered `destructive` so the gate shows
                  the exact method/URL/headers/body before anything is sent. The universal
                  integration: it talks to every REST API (self-hosted services especially)
                  without Saturn owning a per-service integration.

(There is deliberately no monolithic `deep_research` tool: multi-source research is the
living-plan loop's job — the planner composes web_search + web_extract steps, each visible in
the plan rail, gated, and traced. A single opaque research call would hide exactly the steps
this product exists to show; it was removed June 2026 as a scope cut.)

Key-optional, local-first provider strategy
--------------------------------------------
Neither of these *require* a Tavily API key. Tavily is treated as a quality upgrade, not a hard
dependency, so the agent stays useful when the user has no key or has run out of usage.

  web_search    resolves a backend via `_use_tavily()`. With `web.provider: auto` (the default)
                it prefers Tavily when a healthy TAVILY_API_KEY is present and transparently
                falls back to keyless DuckDuckGo (`ddgs`) on a missing key or a quota/usage
                error. Once Tavily fails that way it is disabled for the rest of the session
                (`_TAVILY_DISABLED`) so we don't re-hit a dead key every turn.
  web_extract   is local-first: it fetches + extracts readable text with `trafilatura`, no key
                and no API call. (Tavily Extract is only used if `web.provider: tavily` is
                forced and a key is present.)

Provider selection lives in `config.yaml` under `web:` (`provider`, `max_results`); nothing is
hard-coded here.
"""

import os

import diag
from trust import egress

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
from tools.toolspec import register_tool

load_dotenv()

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
    # ONE air-gap check up front — the gate keys on airgap_on(), not the host, and a second
    # check before the keyless fallback would double-record the blocked event. RECORDING, by
    # contrast, happens at each real send point below, so the ledger names the backend actually
    # contacted: a Tavily attempt that falls back to DuckDuckGo really exits twice.
    blocked = egress.check(
        "web_search", "tavily.com" if _use_tavily() else "duckduckgo.com", query
    )
    if blocked:
        return blocked
    if _use_tavily():
        # Recorded BEFORE the attempt — the ledger's deliberate fail-toward-recording property
        # (egress.record docstring; http_request does the same): a call that dies mid-flight
        # still left the machine.
        egress.record("web_search", "tavily.com", query, provider="tavily",
                      n_bytes=len(query or ""))
        try:
            return _client().search(query, max_results=_max_results())
        except _TAVILY_FALLBACK_ERRORS as err:
            _disable_tavily(err)  # dead key/quota — keyless for the rest of the session
        except Exception as err:
            # Any other Tavily failure (network blip, 5xx, odd response shape) falls back to
            # the keyless backend for THIS call only — the key may be fine, so it isn't
            # disabled for the session. A flaky Tavily must never cost an answer DuckDuckGo
            # could have given.
            diag.log(f"[web] Tavily search failed ({type(err).__name__}); DuckDuckGo fallback")
    # The keyless path really contacts DuckDuckGo — its own ledger event. After a failed Tavily
    # attempt this is deliberately a SECOND event: both exits happened.
    egress.record("web_search", "duckduckgo.com", query, provider="duckduckgo",
                  n_bytes=len(query or ""))
    return _ddg_search(query, _max_results())


@register_tool("read_only")
def web_extract(url: str):
    """Extract the readable page content behind a URL. Use this to read a specific page that
    web_search surfaced. Runs locally (trafilatura) with no API key by default; only uses
    Tavily Extract when the web provider is explicitly forced to 'tavily'."""
    # Normalize FIRST: an empty call must return before any egress accounting — host_of(str([]))
    # would otherwise put a phantom blocked event with the garbage host "[]" into the air-gap
    # ledger (and the durable hash-chained log) for a call that could never have sent anything.
    urls = [u for u in (url if isinstance(url, (list, tuple)) else [url]) if u]
    if not urls:
        return "No URL provided to extract."
    first_url = urls[0]
    # ONE air-gap check, per-send-point recording — mirrors web_search: the gate keys on
    # airgap_on(), so a single check covers the whole call (a second would double-record the
    # blocked event); RECORDING below names the host actually contacted, per send.
    blocked = egress.check("web_extract", egress.host_of(str(first_url)), str(first_url))
    if blocked:
        return blocked
    if _provider() == "tavily" and _use_tavily():
        # Forced-Tavily extraction sends every URL to Tavily's API in one call — the pages' own
        # hosts are never contacted on this branch, so ONE event names the backend (same label
        # as web_search), sized by everything sent.
        egress.record("web_extract", "tavily.com", str(first_url), provider="tavily",
                      n_bytes=sum(len(str(u)) for u in urls))
        try:
            return _client().extract(url)
        except _TAVILY_FALLBACK_ERRORS as err:
            _disable_tavily(err)
        except Exception as err:
            # Transient Tavily failure: degrade to the local extractor for this call only
            # (mirrors web_search — the key may be fine, so don't disable it).
            diag.log(f"[web] Tavily extract failed ({type(err).__name__}); local fallback")
    # Local-first path: each URL is its own fetch, so each gets its own ledger event naming ITS
    # host — a multi-URL extract to three hosts is three sends, and /privacy egress, the rail
    # leaf, and the Glass Box must say so (recorded before the send: fail-toward-recording).
    # After a failed Tavily attempt these are deliberately additional events.
    results = {}
    for u in urls:
        egress.record("web_extract", egress.host_of(str(u)), str(u), n_bytes=len(str(u)))
        results[u] = _local_extract(u)
    return results if len(results) > 1 else next(iter(results.values()))


# Response content-types returned as text; anything else is summarized, not dumped — a binary
# body would be mojibake in context (and the tool node clamps observations anyway, gotcha #5).
_TEXTUAL_TYPES = ("text", "json", "xml", "html", "javascript", "urlencoded")


@register_tool("destructive")
def http_request(url: str, method: str = "GET", headers: dict | None = None,
                 body: str | None = None):
    """Send one HTTP request to a URL or API endpoint and return the response (status code,
    content type, body). Use this to talk to APIs and self-hosted services (REST endpoints,
    home-lab apps, webhooks) — NOT for ordinary web reading (use web_search/web_extract for
    that). Every call is approved by the human first, who sees the exact method, URL, headers,
    and body before anything is sent."""
    method = (method or "GET").upper()
    host = egress.host_of(url)
    blocked = egress.check("http_request", host, f"{method} {url}")
    if blocked:
        return blocked
    egress.record("http_request", host, f"{method} {url}",
                  n_bytes=len(url or "") + len(body or ""))
    try:
        timeout = float(get_config().get("web.request_timeout", 30))
    except (TypeError, ValueError):
        timeout = 30.0
    diag.log(f"http_request : {method} {url}")  # which endpoint, beyond the wrapper's timing line
    try:
        resp = httpx.request(
            method, url,
            headers=headers or None,
            content=body if body is not None else None,
            timeout=timeout,
            follow_redirects=True,
        )
        ctype = resp.headers.get("content-type", "")
        if not ctype or any(t in ctype for t in _TEXTUAL_TYPES):
            payload = resp.text
        else:
            payload = f"(binary response: {ctype}, {len(resp.content)} bytes)"
        return {"status": resp.status_code, "content_type": ctype, "body": payload}
    except httpx.HTTPError as err:
        return f"http_request failed: {type(err).__name__}: {err}"
