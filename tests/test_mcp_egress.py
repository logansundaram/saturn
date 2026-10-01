"""
The MCP boundary — a remote (http/sse) tool call is recorded to the egress ledger before it is
sent; a stdio server is a local child process, not network egress.
"""

import pytest

from trust import egress
from tools.toolspec import ToolError

ANTHROPIC_KEY = "sk-ant-" + "a" * 24
BEARER = "Bearer " + "b" * 30


def _forge(monkeypatch, transport="http"):
    """A configured-but-down server: the egress boundary runs before the (stubbed)
    reconnect, so call_tool exercises the boundary without any network."""
    from tools import mcp_client as mc

    spec = mc.ServerSpec(name="srv", transport=transport, url="http://mcp.example.com/api")
    st = mc._ServerState(spec=spec, state="error", error="down")
    monkeypatch.setitem(mc._SERVERS, "srv", st)
    monkeypatch.setattr(mc, "_launch", lambda s: None)
    monkeypatch.setattr(mc, "_await_ready", lambda states, timeout: None)
    return mc


def test_http_call_is_recorded_with_its_args_unchanged(monkeypatch, isolated_paths):
    mc = _forge(monkeypatch)
    args = {"text": ANTHROPIC_KEY, "headers": [BEARER]}
    mark = egress.next_seq()
    with pytest.raises(ToolError):  # never connected — the boundary already did its job
        mc.call_tool("srv", "post", args)
    evs = egress.events_since(mark)
    assert [(e.channel, e.host, e.detail) for e in evs] == [("mcp", "mcp.example.com", "srv.post")]
    assert evs[0].n_bytes > 0
    assert args == {"text": ANTHROPIC_KEY, "headers": [BEARER]}  # nothing rewrites the args


def test_stdio_server_call_is_untracked_not_a_send(monkeypatch, isolated_paths):
    mc = _forge(monkeypatch, transport="stdio")
    mark = egress.next_seq()
    with pytest.raises(ToolError):
        mc.call_tool("srv", "post", {"text": ANTHROPIC_KEY})
    # A local child process is not a recorded SEND — but Saturn cannot see what it does with
    # the network, so the call is on the ledger as untracked.
    assert [(e.channel, e.host, e.status) for e in egress.events_since(mark)] == [
        ("mcp", "srv", egress.UNTRACKED)]


def test_map_strings_visits_exactly_what_iter_strings_yields():
    # The rewrite walker (map_strings — the trace store's clipping) and the scan walker
    # (iter_strings — the gate's secret scan) read the same leaves (dict keys and non-string
    # scalars skipped).
    from textutil import iter_strings, map_strings

    tree = {"a": "s1", "b": [1, "s2", ("s3", None)], "c": {"k": "s4"}, "n": 7}
    seen = []

    def swap(s):
        seen.append(s)
        return s.upper()

    out = map_strings(tree, swap)
    assert seen == list(iter_strings(tree))
    assert out == {"a": "S1", "b": [1, "S2", ["S3", None]], "c": {"k": "S4"}, "n": 7}
