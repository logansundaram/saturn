"""
web.py egress attribution — the ledger must name the host ACTUALLY contacted.

web_search is keyless DuckDuckGo (one send, one event naming
duckduckgo.com), web_extract fetches one page itself (one event naming ITS host, plus one per
further host a redirect reaches). The air-gap check is up-front; recording is
fail-toward-recording, before the send.

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


def test_extract_empty_url_records_nothing():
    # No URL → nothing sent → nothing recorded.
    with pytest.raises(ToolError, match="No URL"):
        web.web_extract.invoke({"url": "  "})
    assert egress.count() == 0


def test_extract_empty_url_records_nothing_under_airgap(monkeypatch):
    # The empty guard precedes the air-gap check: an empty call must not put a phantom BLOCKED
    # event into the ledger for a send that could never have happened.
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)
    with pytest.raises(ToolError, match="No URL"):
        web.web_extract.invoke({"url": ""})
    assert egress.count() == 0


def test_extract_takes_one_url_as_its_schema_says():
    """The schema is `url: str`, so a list never reaches the body — there is no multi-URL
    path to keep in step with the ledger."""
    with pytest.raises(Exception):
        web.web_extract.invoke({"url": ["https://a.example/x", "https://b.example/y"]})
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


def _redirecting_get(chain):
    """A stand-in for web._get over `chain` = {url: (status, location_or_body)}; records the
    URLs actually requested."""
    import httpx

    asked = []

    def get(url):
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
    monkeypatch.setattr(web, "_get", get)
    monkeypatch.setattr("trafilatura.extract", lambda html, **kw: "hello")
    assert web.web_extract.invoke({"url": "https://a.example/x"}) == "hello"
    assert asked == ["https://a.example/x", "https://tracker.b.example/r?id=1",
                     "https://tracker.b.example/final"]
    assert [(e.host, e.status) for e in egress.events()] == [
        ("a.example", egress.SENT), ("tracker.b.example", egress.SENT)]


def test_extract_redirect_loop_is_a_failure(monkeypatch):
    get, _asked = _redirecting_get({"https://a.example/x": (302, "https://a.example/x")})
    monkeypatch.setattr(web, "_get", get)
    with pytest.raises(ToolError, match="redirect"):
        web.web_extract.invoke({"url": "https://a.example/x"})


def test_extract_fetches_only_through_the_recorded_path():
    """trafilatura's own fetch follows redirects out of sight of the ledger — never called."""
    import inspect

    assert "fetch_url" not in inspect.getsource(web) and "fetch_response" not in inspect.getsource(web)


# ── no generic HTTP tool ───────────────────────────────────────────────────────────────────────


def test_http_request_is_cut():
    """The universal-integration tool is gone — MCP is the integration surface. A resurrected
    http_request here means the cut regressed (and the gate lost its renderer for it)."""
    assert not hasattr(web, "http_request")
    from tools.registry import tools_by_name

    assert "http_request" not in tools_by_name


def test_a_public_page_cannot_redirect_the_fetch_into_the_private_network(monkeypatch):
    """A redirect is the server choosing the next URL. One that points a public fetch at this
    machine or the LAN is refused, not followed — and nothing is sent to the private host."""
    import httpx

    def fake_get(url):
        return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/tags"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(web, "_get", fake_get)
    with pytest.raises(ToolError, match="private"):
        web._fetch("https://example.com/start")
    assert egress.events() == []  # the hop was never recorded because it was never sent


def _serve(monkeypatch, handler):
    """Route web._get's client through an in-process transport — the real request path, no
    network."""
    import httpx

    real = httpx.Client
    monkeypatch.setattr(web.httpx, "Client",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_fetch_stops_reading_a_body_past_the_cap(monkeypatch):
    """A page is read up to the cap and no further: one endless or enormous response must not
    be pulled into memory whole (the observation clamp only trims what was already read)."""
    import httpx

    served = []

    def body():
        for _ in range(1000):
            served.append(1)
            yield b"x" * 65536

    _serve(monkeypatch, lambda request: httpx.Response(200, content=body()))
    monkeypatch.setattr(web, "_MAX_FETCH_BYTES", 200_000)
    text = web._fetch("https://example.com/huge")
    assert 200_000 <= len(text) < 200_000 + 65536
    assert len(served) < 10


def test_fetch_never_follows_redirects_itself_and_names_saturn(monkeypatch):
    import httpx

    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="<p>ok</p>")

    clients = []
    real = httpx.Client

    def client(**kw):
        clients.append(kw)
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(web.httpx, "Client", client)
    assert web._fetch("https://example.com/") == "<p>ok</p>"
    assert clients[0]["follow_redirects"] is False   # every hop is Saturn's, so every host is recorded
    assert "Saturn" in seen["ua"] and "Saturday" not in seen["ua"]
