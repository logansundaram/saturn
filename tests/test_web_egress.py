"""
web.py egress attribution — the ledger must name the host ACTUALLY contacted.

API-less since 2026-07-06: web_search is keyless DuckDuckGo (one send, one event naming
duckduckgo.com), web_extract fetches each page itself (one event PER URL naming ITS host — a
multi-URL extract to three hosts is three sends, and /policy egress and the rail leaf must
say so). The air-gap check stays single and up-front (it keys on airgap_on(),
not the host); recording is fail-toward-recording, before the send. (The Tavily backend and its
fallback double-record contract left with the API-less pivot.)

Everything runs offline: DDGS and the local extractor are stubbed.
"""

import pytest

import tools.web as web
from config import get_config
from trust import egress
from tools.toolspec import ToolError


class _StubDDGS:
    """Offline DDGS stand-in: one canned hit, never touches the network."""

    def text(self, query, max_results=None):
        return [{"title": "t", "href": "https://example.com/a", "body": "b"}]


@pytest.fixture(autouse=True)
def fresh_web_state(isolated_paths, monkeypatch):
    """Empty ledger, air-gap off, DDGS stubbed — each test runs deterministically offline."""
    egress.clear()
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", False)
    monkeypatch.setattr(web, "DDGS", _StubDDGS)
    yield
    egress.clear()


# ── web_search ─────────────────────────────────────────────────────────────────────────────────


def test_keyless_search_records_duckduckgo_only():
    out = web.web_search.invoke({"query": "hello"})
    assert out["provider"] == "duckduckgo"
    assert [(e.host, e.provider, e.status) for e in egress.events()] == [
        ("duckduckgo.com", "duckduckgo", egress.SENT)
    ]


def test_search_recorded_before_the_send(monkeypatch):
    # Fail-toward-recording: a search that dies mid-flight still left the machine — the ledger
    # must already carry the event when the backend raises.
    class _Boom:
        def text(self, *a, **k):
            raise RuntimeError("network died")

    monkeypatch.setattr(web, "DDGS", _Boom)
    with pytest.raises(RuntimeError):
        web.web_search.invoke({"query": "hello"})
    assert [e.host for e in egress.events()] == ["duckduckgo.com"]


def test_airgap_blocks_search_before_any_send(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)

    class _Boom:
        def text(self, *a, **k):
            raise AssertionError("network touched under air-gap")

    monkeypatch.setattr(web, "DDGS", _Boom)
    out = web.web_search.invoke({"query": "hello"})
    assert "air-gap" in out.lower()
    # Exactly ONE blocked event (the single up-front check), nothing sent.
    assert [e.status for e in egress.events()] == [egress.BLOCKED]


# ── web_extract ────────────────────────────────────────────────────────────────────────────────


def test_extract_local_records_target_host(monkeypatch):
    monkeypatch.setattr(web, "_local_extract", lambda u: "page text")
    out = web.web_extract.invoke({"url": "https://example.org/page"})
    assert out == "page text"
    assert [(e.host, e.channel) for e in egress.events()] == [("example.org", "web_extract")]


def test_extract_empty_list_records_nothing():
    # No URL → nothing sent → nothing recorded (the old top-of-function record logged a phantom
    # event for an empty call). .func bypasses the str schema to reach the list-tolerant body.
    with pytest.raises(ToolError, match="No URL"):
        web.web_extract.func(url=[])
    assert egress.count() == 0


def test_extract_empty_list_records_nothing_under_airgap(monkeypatch):
    # The empty guard precedes the air-gap check: an empty call must not put a phantom BLOCKED
    # event with the garbage host "[]" into the ledger for a send that could never have happened.
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)
    with pytest.raises(ToolError, match="No URL"):
        web.web_extract.func(url=[])
    assert egress.count() == 0


def test_extract_airgap_blocks_before_any_fetch(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)
    monkeypatch.setattr(
        web, "_local_extract",
        lambda u: (_ for _ in ()).throw(AssertionError("fetched under air-gap")),
    )
    out = web.web_extract.invoke({"url": "https://example.org/page"})
    assert "air-gap" in out.lower()
    assert [e.status for e in egress.events()] == [egress.BLOCKED]


def test_extract_multi_url_records_every_host(monkeypatch):
    # Each URL in a multi-URL extract is its own fetch — each host gets its own ledger event.
    # (Previously one event named only the first host, hiding real egress to every other host
    # from /policy egress, the rail leaf, and the receipt.)
    monkeypatch.setattr(web, "_local_extract", lambda u: f"text of {u}")
    out = web.web_extract.func(url=["https://a.example/x", "https://b.example/y",
                                    "https://c.example/z"])
    assert set(out) == {"https://a.example/x", "https://b.example/y", "https://c.example/z"}
    assert [(e.host, e.channel, e.status) for e in egress.events()] == [
        ("a.example", "web_extract", egress.SENT),
        ("b.example", "web_extract", egress.SENT),
        ("c.example", "web_extract", egress.SENT),
    ]


def _redirecting_get(chain):
    """A stand-in for httpx.get over `chain` = {url: (status, location_or_body)}; records the
    URLs actually requested, and refuses to be asked to follow redirects itself."""
    import httpx

    asked = []

    def get(url, *, follow_redirects, **_kw):
        assert follow_redirects is False  # every hop is Saturn's, so every host is recorded
        asked.append(url)
        status, value = chain[url]
        req = httpx.Request("GET", url)
        if 300 <= status < 400:
            return httpx.Response(status, headers={"location": value}, request=req)
        return httpx.Response(status, text=value, request=req)

    return get, asked


def test_extract_records_every_host_a_redirect_reaches(monkeypatch):
    """A redirect to another host is a send to that host: recorded (and air-gap checked)
    BEFORE the hop, so the ledger names every host contacted — not only the one asked for."""
    get, asked = _redirecting_get({
        "https://a.example/x": (302, "https://tracker.b.example/r?id=1"),
        "https://tracker.b.example/r?id=1": (301, "/final"),
        "https://tracker.b.example/final": (200, "<html><body><p>hello</p></body></html>"),
    })
    monkeypatch.setattr(web.httpx, "get", get)
    monkeypatch.setattr("trafilatura.extract", lambda html, **kw: "hello")
    assert web.web_extract.invoke({"url": "https://a.example/x"}) == "hello"
    assert asked == ["https://a.example/x", "https://tracker.b.example/r?id=1",
                     "https://tracker.b.example/final"]
    assert [(e.host, e.status) for e in egress.events()] == [
        ("a.example", egress.SENT), ("tracker.b.example", egress.SENT)]


def test_extract_redirect_loop_is_a_failure(monkeypatch):
    get, _asked = _redirecting_get({"https://a.example/x": (302, "https://a.example/x")})
    monkeypatch.setattr(web.httpx, "get", get)
    with pytest.raises(ToolError, match="redirect"):
        web.web_extract.invoke({"url": "https://a.example/x"})


def test_extract_fetches_only_through_the_recorded_path():
    """trafilatura's own fetch follows redirects out of sight of the ledger — never called."""
    import inspect

    assert "fetch_url" not in inspect.getsource(web) and "fetch_response" not in inspect.getsource(web)


# ── http_request: CUT 2026-07-16 ───────────────────────────────────────────────────────────────


def test_http_request_is_cut():
    """The universal-integration tool is gone — MCP is the integration surface. A resurrected
    http_request here means the cut regressed (and the gate lost its renderer for it)."""
    assert not hasattr(web, "http_request")
    from tools.registry import tools_by_name

    assert "http_request" not in tools_by_name
