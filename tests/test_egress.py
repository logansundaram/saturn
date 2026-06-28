"""egress.py — the network-boundary ledger + air-gap gate."""

import pytest

from trust import egress
from config import get_config


@pytest.fixture(autouse=True)
def fresh_ledger(isolated_paths, monkeypatch):
    """Empty the ledger and pin air-gap off around each test. _CLEARED_AT is reset too — the
    isolation clear() must look like a FRESH SESSION, not an operator `/privacy egress clear`
    (which summary() now reports via its `cleared` marker)."""
    egress.clear()
    monkeypatch.setattr(egress, "_CLEARED_AT", 0)
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", False)
    yield
    egress.clear()


def _set_airgap(monkeypatch, on):
    monkeypatch.setitem(get_config()._data["runtime"], "airgap", on)


def test_record_and_summary():
    egress.record("web_search", "duckduckgo.com", "hello", n_bytes=5)
    egress.record("http_request", "api.example.com", "GET /x", n_bytes=10)
    s = egress.summary()
    assert s["sent"] == 2
    assert s["blocked"] == 0
    assert s["bytes"] == 15
    assert set(s["hosts"]) == {"duckduckgo.com", "api.example.com"}
    assert s["by_channel"] == {"web_search": 1, "http_request": 1}
    assert egress.count() == 2


def test_check_passes_when_airgap_off(monkeypatch):
    _set_airgap(monkeypatch, False)
    assert egress.check("web_search", "duckduckgo.com", "q") is None
    # A pass-through does NOT record a blocked event (the tool records its own SENT event).
    assert egress.count() == 0


def test_check_blocks_and_records_when_airgap_on(monkeypatch):
    _set_airgap(monkeypatch, True)
    msg = egress.check("http_request", "api.example.com", "POST /x")
    assert msg is not None
    assert "air-gap" in msg.lower()
    s = egress.summary()
    assert s["blocked"] == 1
    assert s["sent"] == 0


def test_airgap_read_live(monkeypatch):
    assert egress.airgap_on() is False
    _set_airgap(monkeypatch, True)
    assert egress.airgap_on() is True


def test_redactions_aggregated():
    egress.record("llm", "anthropic API", "claude", n_bytes=100, redactions=3)
    egress.record("llm", "anthropic API", "claude", n_bytes=100, redactions=2)
    assert egress.summary()["redactions"] == 5


def test_record_is_crash_proof():
    # Junk must never raise into the calling network op.
    egress.record("web_search", None, None, n_bytes="lots")  # type: ignore[arg-type]
    assert egress.count() == 1


def test_ledger_cap():
    for i in range(egress._MAX_EVENTS + 50):
        egress.record("web_search", "h", str(i))
    assert egress.count() == egress._MAX_EVENTS


def test_clear():
    egress.record("web_search", "h", "x")
    egress.clear()
    assert egress.count() == 0


def test_summary_carries_cleared_marker():
    """summary() must say when the counts are since-the-clear: the signed trust report embeds
    this dict verbatim as egress_session, and without the marker a post-clear report would
    attest 'sent: 0' over a session that sent (the per-turn cleared_since hazard, ledger-wide)."""
    egress.record("web_search", "h", "x")
    assert egress.summary()["cleared"] is False  # fresh session (fixture resets _CLEARED_AT)
    egress.clear()
    s = egress.summary()
    assert s["cleared"] is True
    assert s["sent"] == 0  # the understated count the marker exists to qualify


def test_host_of():
    # THE shared ledger host derivation (tools/web.py and tools/mcp_client.py both import it
    # from here — formerly two byte-identical local copies).
    assert egress.host_of("https://api.example.com/v1/x?q=1") == "api.example.com"
    assert egress.host_of("http://localhost:8000/mcp") == "localhost"
    # An unparseable/host-less value falls back to the raw input — a boundary label must never
    # be lost (unlike ollama_is_local, which deliberately fails toward NOT-local instead).
    assert egress.host_of("not a url") == "not a url"
    assert egress.host_of("") == ""
