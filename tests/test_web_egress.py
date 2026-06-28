"""
web.py egress attribution — the ledger must name the backend ACTUALLY contacted.

The Tavily→DuckDuckGo fallback used to compute one backend host up front and record it before
branching: a Tavily attempt that failed (dead key OR transient error) fell through to DuckDuckGo
with the ledger carrying only a tavily.com event that never completed — and NOTHING for the DDG
request that really went out. Recording now happens at each send point (fail-toward-recording,
before the attempt), so a fallback turn is deliberately TWO events: both exits happened. The
air-gap check stays single and up-front (it keys on airgap_on(), not the host — a second check
on the fallback branch would double-record the blocked event).

web_extract has the mirror contract: forced provider:tavily sends the URL to Tavily's API and
must record the backend, never the target page's host; the local path records the page host.

Everything runs offline: the Tavily client, DDGS, and the local extractor are stubbed.
"""

import pytest
from tavily import UsageLimitExceededError

import tools.web as web
from config import get_config
from trust import egress


class _StubDDGS:
    """Offline DDGS stand-in: one canned hit, never touches the network."""

    def text(self, query, max_results=None):
        return [{"title": "t", "href": "https://example.com/a", "body": "b"}]


class _DeadTavily:
    """Tavily client whose every call fails with a quota error (the dead-key shape that
    disables Tavily for the session via _disable_tavily)."""

    def search(self, *a, **k):
        raise UsageLimitExceededError("quota exhausted")

    def extract(self, *a, **k):
        raise UsageLimitExceededError("quota exhausted")


class _OkTavily:
    def search(self, *a, **k):
        return {"query": "q", "provider": "tavily", "results": []}

    def extract(self, *a, **k):
        return {"results": []}


@pytest.fixture(autouse=True)
def fresh_web_state(isolated_paths, monkeypatch):
    """Empty ledger, air-gap off, Tavily session-disable flag reset, DDGS stubbed, and no real
    key in the environment — each test picks its backend deterministically and offline.
    isolated_paths keeps the durable egress log (every record() also appends to disk) out of
    the real database/."""
    egress.clear()
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", False)
    # setattr FIRST so teardown restores the pristine value even when a test trips
    # _disable_tavily (which assigns the module global directly).
    monkeypatch.setattr(web, "_TAVILY_DISABLED", False)
    monkeypatch.setattr(web, "DDGS", _StubDDGS)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    yield
    egress.clear()


def _force_tavily(monkeypatch):
    """Pin web.provider to 'tavily' so _use_tavily() is True without a key in the env."""
    monkeypatch.setitem(get_config()._data["web"], "provider", "tavily")


# ── web_search ─────────────────────────────────────────────────────────────────────────────────


def test_keyless_search_records_duckduckgo_only():
    out = web.web_search.invoke({"query": "hello"})
    assert out["provider"] == "duckduckgo"
    assert [(e.host, e.provider, e.status) for e in egress.events()] == [
        ("duckduckgo.com", "duckduckgo", egress.SENT)
    ]


def test_tavily_success_records_tavily_only(monkeypatch):
    _force_tavily(monkeypatch)
    monkeypatch.setattr(web, "_client", lambda: _OkTavily())
    web.web_search.invoke({"query": "hello"})
    assert [(e.host, e.provider) for e in egress.events()] == [("tavily.com", "tavily")]


def test_dead_key_fallback_records_both_exits(monkeypatch):
    # The headline regression: the Tavily attempt is recorded AND the DDG request that actually
    # answered is recorded — previously the ledger carried only the never-completed tavily.com
    # event and silently omitted the real DuckDuckGo egress.
    _force_tavily(monkeypatch)
    monkeypatch.setattr(web, "_client", lambda: _DeadTavily())
    out = web.web_search.invoke({"query": "hello"})
    assert out["provider"] == "duckduckgo"  # the answer really came from the fallback
    evs = egress.events()
    assert [(e.host, e.provider) for e in evs] == [
        ("tavily.com", "tavily"),
        ("duckduckgo.com", "duckduckgo"),
    ]
    assert all(e.status == egress.SENT for e in evs)


def test_transient_failure_fallback_records_both_exits(monkeypatch):
    # A non-key error (network blip / 5xx) must not disable Tavily for the session, but the
    # ledger still carries both exits for THIS call.
    _force_tavily(monkeypatch)

    class _Flaky:
        def search(self, *a, **k):
            raise RuntimeError("503")

    monkeypatch.setattr(web, "_client", lambda: _Flaky())
    web.web_search.invoke({"query": "hello"})
    assert web._TAVILY_DISABLED is False
    assert [e.host for e in egress.events()] == ["tavily.com", "duckduckgo.com"]


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


def test_extract_forced_tavily_records_backend_not_target(monkeypatch):
    # Forced provider:tavily never contacts the page's host — the event names the backend the
    # URL was actually SENT to (same label as web_search), not where the URL points.
    _force_tavily(monkeypatch)
    monkeypatch.setattr(web, "_client", lambda: _OkTavily())
    web.web_extract.invoke({"url": "https://example.org/page"})
    assert [(e.host, e.provider) for e in egress.events()] == [("tavily.com", "tavily")]


def test_extract_tavily_failure_records_both_exits(monkeypatch):
    _force_tavily(monkeypatch)
    monkeypatch.setattr(web, "_client", lambda: _DeadTavily())
    monkeypatch.setattr(web, "_local_extract", lambda u: "page text")
    out = web.web_extract.invoke({"url": "https://example.org/page"})
    assert out == "page text"
    assert [e.host for e in egress.events()] == ["tavily.com", "example.org"]


def test_extract_empty_list_records_nothing():
    # No URL → nothing sent → nothing recorded (the old top-of-function record logged a phantom
    # event for an empty call). .func bypasses the str schema to reach the list-tolerant body.
    out = web.web_extract.func(url=[])
    assert "No URL" in out
    assert egress.count() == 0


def test_extract_empty_list_records_nothing_under_airgap(monkeypatch):
    # The empty guard precedes the air-gap check: an empty call must not put a phantom BLOCKED
    # event with the garbage host "[]" into the ledger (and the durable hash-chained audit log)
    # for a send that could never have happened.
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", True)
    out = web.web_extract.func(url=[])
    assert "No URL" in out
    assert egress.count() == 0


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
