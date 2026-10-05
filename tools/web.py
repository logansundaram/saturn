"""
Web tools — everything that reaches the live internet.

  web_search    — a single web search query.
  web_extract   — fetch + extract the readable content behind a URL.

There is deliberately no monolithic research tool: multi-source research is the agent loop's
job, composed of web_search + web_extract calls that are each visible in the trace, gated and
recorded. Nor a generic HTTP tool: the MCP client is the integration surface, with per-server
trust declarations. The only ways out of this machine are a search query, a page fetch, and the
MCP servers the user configured.

No web tool requires an API key or a paid provider account — a product whose pitch is "your
data stays yours" should not steer its users toward mailing every search query to a keyed
SaaS backend.

  web_search    keyless DuckDuckGo (`ddgs`). The query is the only thing sent, recorded in the
                egress ledger like every exit.
  web_extract   fully local extraction: fetch the page (httpx) + pull readable text with
                `trafilatura`. Only the page's own host is contacted, plus any host its
                redirects lead to — each hop followed by hand and recorded before it is sent.

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
# How much of one response body is read. The observation clamp (nodes/tools.py) trims what
# reaches the model, but only after the whole body is in memory — an endless or enormous
# response has to be cut off at the socket. 5 MB is far past any page worth extracting.
_MAX_FETCH_BYTES = 5_000_000
_USER_AGENT = "Mozilla/5.0 (Saturn)"


def _get(url: str) -> httpx.Response:
    """ONE request, never following a redirect itself, with the body read up to
    `_MAX_FETCH_BYTES` (a redirect's body is not read at all). The seam the fetch tests
    replace."""
    with httpx.Client(follow_redirects=False, timeout=20.0,
                      headers={"User-Agent": _USER_AGENT}) as client:
        with client.stream("GET", url) as resp:
            chunks, size = [], 0
            if not resp.is_redirect:
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= _MAX_FETCH_BYTES:
                        break
            # Rebuilt from the decoded bytes actually read; only the headers the caller reads
            # survive (a kept content-encoding would decode the body a second time).
            headers = {k: v for k, v in resp.headers.items()
                       if k.lower() in ("location", "content-type")}
            return httpx.Response(resp.status_code, headers=headers, content=b"".join(chunks),
                                  request=resp.request)


def _fetch(url: str) -> str:
    """GET `url`, following redirects ONE HOP AT A TIME: a hop to a host this fetch has not
    contacted yet is air-gap checked and recorded BEFORE it is sent, so the ledger names every
    host a page reached, not only the one asked for (web_extract records the first). Automatic
    redirect following — httpx's or trafilatura's own fetch — would contact hosts out of the
    ledger's sight."""
    contacted = {egress.host_of(url)}
    private_start = egress.is_private_host(egress.host_of(url))
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            resp = _get(url)
        except httpx.HTTPError as exc:
            raise ToolError(f"could not fetch {url}: {exc}") from exc
        location = resp.headers.get("location")
        if not (resp.is_redirect and location):
            if resp.status_code >= 400:
                raise ToolError(f"could not fetch {url}: HTTP {resp.status_code}")
            return resp.text
        url = urljoin(url, location)
        host = egress.host_of(url)
        if egress.is_private_host(host) and not private_start:
            # The server chose this hop: a public page must not steer the fetch onto this
            # machine or the LAN (a local service trusts localhost).
            raise ToolError(f"redirect to a private address ({host}) refused — not followed")
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
    # Normalize FIRST: an empty call must return before any egress accounting — it could never
    # have sent anything, so it must not put an event in the ledger.
    url = str(url or "").strip()
    if not url:
        raise ToolError("No URL provided to extract.")
    host = egress.host_of(url)
    blocked = egress.check("web_extract", host, url)
    if blocked:
        return blocked
    # Recorded before the send (fail-toward-recording); _fetch records each further host a
    # redirect reaches.
    egress.record("web_extract", host, url, n_bytes=len(url))
    return _local_extract(url)  # its ToolError is the call's failure
