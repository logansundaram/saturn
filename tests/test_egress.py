"""egress.py — the network-boundary ledger + air-gap gate."""

import pytest

from trust import egress
from config import get_config


@pytest.fixture(autouse=True)
def fresh_ledger(isolated_paths, monkeypatch):
    """Empty the ledger and pin air-gap off around each test. _CLEARED_AT is reset too — the
    isolation clear() must look like a FRESH SESSION, not an operator `/policy egress clear`
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
    """summary() must say when the counts are since-the-clear: /policy egress renders this
    dict, and without the marker a post-clear readout would claim 'sent: 0' over a session
    that sent (the per-turn cleared_since hazard, ledger-wide)."""
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


# ── check_or_raise is a view of check(), and the refusal reaches the ledger ───────────────────
# A raising twin that re-implemented check()'s body would let a future rung added inside check()
# silently bypass every LLM/embedder exit.


def test_check_or_raise_records_blocked_and_raises(isolated_paths, monkeypatch):
    import pytest

    from trust import egress

    monkeypatch.setattr(egress, "airgap_on", lambda: True)
    before = len(egress.events())
    with pytest.raises(RuntimeError, match="air-gap"):
        egress.check_or_raise("llm", "ollama @ http://10.0.0.5:11434", "tool_caller → m",
                              provider="ollama", subject="role 'tool_caller' (m)")
    evs = egress.events()
    assert len(evs) == before + 1
    assert evs[-1].status == egress.BLOCKED
    assert evs[-1].provider == "ollama"


def test_check_or_raise_delegates_to_check(monkeypatch):
    from trust import egress

    seen = {}

    def fake_check(channel, host, detail="", *, provider=""):
        seen["args"] = (channel, host, detail, provider)
        return None

    monkeypatch.setattr(egress, "check", fake_check)
    egress.check_or_raise("llm", "h", "d", provider="p")  # allowed: returns without raising
    assert seen["args"] == ("llm", "h", "d", "p")


def test_untracked_run_is_neither_sent_nor_silent():
    """A shell command or a stdio MCP server is a process Saturn cannot see inside. Its run is
    recorded as UNTRACKED: never counted as a send, and never absent — so no surface can claim
    the boundary stayed closed over a turn that ran one."""
    egress.record("shell", "?", "git pull", status=egress.UNTRACKED)
    s = egress.summary()
    assert (s["sent"], s["blocked"], s["untracked"]) == (0, 0, 1)
    assert s["hosts"] == [] and s["bytes"] == 0
    assert egress.count() == 0  # the status bar counts what crossed or was blocked
    assert [e.status for e in egress.events()] == [egress.UNTRACKED]


@pytest.mark.parametrize("host, private", [
    ("localhost", True), ("127.0.0.1", True), ("::1", True), ("0.0.0.0", True),
    ("192.168.1.1", True), ("10.0.0.7", True), ("169.254.169.254", True),
    ("printer.local", True), ("nas", True), ("app.localhost", True),
    ("example.com", False), ("8.8.8.8", False), ("127.evil.example.com", False),
])
def test_is_private_host(host, private):
    assert egress.is_private_host(host) is private


@pytest.mark.parametrize("host", ["127.1", "0x7f.0.0.1", "0177.0.0.1", "127.0.1", "2130706433"])
def test_is_private_host_sees_through_inet_aton_shorthand(host):
    """`127.1`, hex and octal octets and a bare decimal are all 127.0.0.1 to the resolver
    (inet_aton); `ipaddress` refuses them, and "has a dot, no private suffix" must not then
    read them as public — that was a straight path from a composed URL to a local service."""
    assert egress.is_private_host(host) is True
    assert egress.is_loopback_host(host) is True
    assert egress.is_private_host(f"{host}.example.com") is False
