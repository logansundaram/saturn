"""egress.py — the network-boundary ledger + air-gap gate."""

import pytest

import egress
from config import get_config


@pytest.fixture(autouse=True)
def fresh_ledger(monkeypatch):
    """Empty the ledger and pin air-gap off around each test."""
    egress.clear()
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
