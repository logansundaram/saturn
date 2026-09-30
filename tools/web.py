"""
Web tools — everything that reaches the live internet.

  web_search    — a single web search query.
  web_extract   — fetch + extract the readable content behind a URL.

(There is deliberately no monolithic `deep_research` tool: multi-source research is the
agent loop's job — the agent composes web_search + web_extract calls, each visible in the
trace rail, gated, and traced. A single opaque research call would hide exactly the steps
this product exists to show; it was removed June 2026 as a scope cut. `http_request` — the
one-call-to-any-REST-API "universal integration" — was CUT 2026-07-16: the MCP client is the
integration surface now, and it arrives with per-server trust declarations, the egress ledger, and
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
                `trafilatura`. Only the page's own host is contacted, plus any host its
                redirects lead to — each hop followed by hand and recorded before it is sent.

(The Tavily backend — `web.provider`, TAVILY_API_KEY, the session fallback latch — was removed
2026-07-06. `trust/secret_scan.py` deliberately KEEPS the `tvly-` secret pattern: the gate's
secret warning covers whatever secrets a call carries, not just ones Saturn uses.)

`web.max_results` lives in `config.yaml`; nothing is hard-coded here.
"""

from urllib.parse import urljoin

from trust import egress

import httpx
from ddgs import DDGS

from config import get_config
from tools.toolspec import ToolError, register_tool


def _max_results() -> int:
    return int(get_config().get("web.max_results", 5))


def _ddg_search(query: str, max_results: int) -> dict:
    """Keyless web search via DuckDuckGo, shaped as the web_search observation
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


_MAX_REDIRECTS = 5


def _fetch(url: str) -> str:
    """GET `url`, following redirects ONE HOP AT A TIME: a hop to a host this fetch has not
    contacted yet is air-gap checked and recorded BEFORE it is sent, so the ledger names every
    host a page reached, not only the one asked for (web_extract records the first). Automatic
    redirect following — httpx's or trafilatura's own fetch — would contact hosts out of the
    ledger's sight."""
    contacted = {egress.host_of(url)}
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            resp = httpx.get(url, follow_redirects=False, timeout=20.0,
                             headers={"User-Agent": "Mozilla/5.0 (Saturday.ai)"})
        except httpx.HTTPError as exc:
            raise ToolError(f"could not fetch {url}: {exc}") from exc
        location = resp.headers.get("location")
        if not (resp.is_redirect and location):
            if resp.status_code >= 400:
                raise ToolError(f"could not fetch {url}: HTTP {resp.status_code}")
            return resp.text
        url = urljoin(url, location)
        host = egress.host_of(url)
        if host not in contacted:
            blocked = egress.check("web_extract", host, url)
            if blocked:
                raise ToolError(blocked)
            egress.record("web_extract", host, url, n_bytes=len(url))
            contacted.add(host)
    raise ToolError(f"too many redirects (over {_MAX_REDIRECTS}) fetching {url}")


def _local_extract(url: str) -> str:
    """Keyless page-content extraction: fetch (`_fetch`) then pull readable text with
    trafilatura."""
    # Imported here, not at module scope: trafilatura costs ~160ms to import and only this one
    # function needs it, while tools/registry pulls this module on every launch (including the
    # -p/-q one-shots, where launch latency is most of the wall clock).
    import trafilatura

    html = _fetch(url)
    if not html:
        raise ToolError(f"could not fetch {url}: empty response")
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
    web_search surfaced. Runs locally (trafilatura) — no API key; only the page's host (and
    any host its redirects lead to) is contacted."""
    # Normalize FIRST: an empty call must return before any egress accounting — host_of(str([]))
    # would otherwise put a phantom blocked event with the garbage host "[]" into the air-gap
    # ledger for a call that could never have sent anything.
    urls = [u for u in (url if isinstance(url, (list, tuple)) else [url]) if u]
    if not urls:
        raise ToolError("No URL provided to extract.")
    # ONE air-gap check up front — the gate keys on airgap_on(), not the host, so a single check
    # covers the whole call (a per-URL check would multi-record the blocked event); RECORDING
    # below names the host actually contacted, per send.
    blocked = egress.check("web_extract", egress.host_of(str(urls[0])), str(urls[0]))
    if blocked:
        return blocked
    # Each URL is its own fetch, so each gets its own ledger event naming ITS host — a multi-URL
    # extract to three hosts is three sends, and /policy egress, the rail leaf, and the Glass
    # Box must say so (recorded before the send: fail-toward-recording).
    if len(urls) == 1:
        egress.record("web_extract", egress.host_of(str(urls[0])), str(urls[0]), n_bytes=len(str(urls[0])))
        return _local_extract(urls[0])  # its ToolError is the call's failure
    results, failed = {}, 0
    for u in urls:
        egress.record("web_extract", egress.host_of(str(u)), str(u), n_bytes=len(str(u)))
        try:
            results[u] = _local_extract(u)
        except ToolError as exc:  # one dead page among several is part of the result
            results[u] = f"[{exc}]"
            failed += 1
    if failed == len(urls):
        raise ToolError("could not fetch any of: " + ", ".join(map(str, urls)))
    return results
