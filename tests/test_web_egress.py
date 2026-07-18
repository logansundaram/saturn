"""
web.py egress attribution — the ledger must name the host ACTUALLY contacted.

API-less since 2026-07-06: web_search is keyless DuckDuckGo (one send, one event naming
duckduckgo.com), web_extract fetches each page itself (one event PER URL naming ITS host — a
multi-URL extract to three hosts is three sends, and /privacy egress, the rail leaf, and the
Glass Box must say so). The air-gap check stays single and up-front (it keys on airgap_on(),
not the host); recording is fail-toward-recording, before the send. (The Tavily backend and its
fallback double-record contract left with the API-less pivot.)

Everything runs offline: DDGS and the local extractor are stubbed.
"""

import pytest

import tools.web as web
from config import get_config
from trust import egress


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
    out = web.web_extract.func(url=[])
    assert "No URL" in out
    assert egress.count() == 0


def test_extract_empty_list_records_nothing_under_airgap(monkeypatch):
    # The empty guard precedes the air-gap check: an empty call must not put a phantom BLOCKED
    # event with the garbage host "[]" into the ledger for a send that could never have happened.
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)
    out = web.web_extract.func(url=[])
    assert "No URL" in out
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
    # from /privacy egress, the rail leaf, the receipt, and the Glass Box.)
    monkeypatch.setattr(web, "_local_extract", lambda u: f"text of {u}")
    out = web.web_extract.func(url=["https://a.example/x", "https://b.example/y",
                                    "https://c.example/z"])
    assert set(out) == {"https://a.example/x", "https://b.example/y", "https://c.example/z"}
    assert [(e.host, e.channel, e.status) for e in egress.events()] == [
        ("a.example", "web_extract", egress.SENT),
        ("b.example", "web_extract", egress.SENT),
        ("c.example", "web_extract", egress.SENT),
    ]


# ── http_request: CUT 2026-07-16 ───────────────────────────────────────────────────────────────


def test_http_request_is_cut():
    """The universal-integration tool is gone — MCP is the integration surface. A resurrected
    http_request here means the cut regressed (and the gate lost its renderer for it)."""
    assert not hasattr(web, "http_request")
    from tools.registry import tools_by_name

    assert "http_request" not in tools_by_name
