"""
Web tools — everything that reaches the live internet.

  web_search    — a single web search query.
  web_extract   — fetch + extract the readable content behind a URL.

(There is deliberately no monolithic `deep_research` tool: multi-source research is the
plan/execute loop's job — the planner composes web_search + web_extract steps, each visible in
the plan rail, gated, and traced. A single opaque research call would hide exactly the steps
this product exists to show; it was removed June 2026 as a scope cut. `http_request` — the
one-call-to-any-REST-API "universal integration" — was CUT 2026-07-16: the MCP client is the
integration surface now, and it arrives with per-server trust declarations, arg redaction, and
status/reload that a generic POST-anywhere tool never had. With it gone, the only ways out of
this machine are a search query, a page fetch, and the MCP servers the user configured.)

API-less by design (2026-07-06 — the Tavily removal)
----------------------------------------------------
No web tool requires an API key or a paid provider account — a product whose pitch is "your
data stays yours" should not steer its users toward mailing every search query to a keyed
SaaS backend, and key management was the single piece of first-run friction the web tools
carried.

  web_search    keyless DuckDuckGo (`ddgs`). The query is the only thing sent, recorded in the
                egress ledger like every exit.
  web_extract   fully local extraction: fetch the page (httpx) + pull readable text with
                `trafilatura`. Only the page's own host is contacted.

(The Tavily backend — `web.provider`, TAVILY_API_KEY, the session fallback latch — was removed
2026-07-06. `trust/redaction.py` deliberately KEEPS the `tvly-` secret pattern: the redaction
scanner guards whatever secrets pass through outgoing text, not just ones Saturn uses.)

`web.max_results` lives in `config.yaml`; nothing is hard-coded here.
"""

from trust import egress

import httpx
import trafilatura
from ddgs import DDGS

from config import get_config
from tools.toolspec import register_tool


def _max_results() -> int:
    return int(get_config().get("web.max_results", 5))


def _ddg_search(query: str, max_results: int) -> dict:
    """Keyless web search via DuckDuckGo, in the result shape the synthesize node understands
    ({'query', 'results': [{title, url, content}]})."""
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
@register_tool("read_only", untrusted=True)
def web_search(query: str):
    """Execute a web search query. Keyless DuckDuckGo — no API key, no account, ever."""
    blocked = egress.check("web_search", "duckduckgo.com", query)
    if blocked:
        return blocked
    # Recorded BEFORE the attempt — the ledger's deliberate fail-toward-recording property
    # (egress.record docstring): a call that dies mid-flight still left the machine.
    egress.record("web_search", "duckduckgo.com", query, provider="duckduckgo",
                  n_bytes=len(query or ""))
    return _ddg_search(query, _max_results())


@register_tool("read_only", untrusted=True)
def web_extract(url: str):
    """Extract the readable page content behind a URL. Use this to read a specific page that
    web_search surfaced. Runs locally (trafilatura) — no API key; only the page's host is
    contacted."""
    # Normalize FIRST: an empty call must return before any egress accounting — host_of(str([]))
    # would otherwise put a phantom blocked event with the garbage host "[]" into the air-gap
    # ledger for a call that could never have sent anything.
    urls = [u for u in (url if isinstance(url, (list, tuple)) else [url]) if u]
    if not urls:
        return "No URL provided to extract."
    # ONE air-gap check up front — the gate keys on airgap_on(), not the host, so a single check
    # covers the whole call (a per-URL check would multi-record the blocked event); RECORDING
    # below names the host actually contacted, per send.
    blocked = egress.check("web_extract", egress.host_of(str(urls[0])), str(urls[0]))
    if blocked:
        return blocked
    # Each URL is its own fetch, so each gets its own ledger event naming ITS host — a multi-URL
    # extract to three hosts is three sends, and /privacy egress, the rail leaf, and the Glass
    # Box must say so (recorded before the send: fail-toward-recording).
    results = {}
    for u in urls:
        egress.record("web_extract", egress.host_of(str(u)), str(u), n_bytes=len(str(u)))
        results[u] = _local_extract(u)
    return results if len(results) > 1 else next(iter(results.values()))
