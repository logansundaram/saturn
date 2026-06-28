"""
Redaction parity at the MCP boundary — remote tool args are scanned like cloud-LLM messages.

The cloud-LLM boundary (llms._CloudBoundaryModel) has had warn/redact for a while; MCP tool args
crossed the wire blind. call_tool now scans outgoing args for http/sse servers: `warn` counts
secret-like values into the egress event, `redact` replaces them in the args actually sent.
stdio servers are local child processes — no boundary, no scan.
"""

import pytest

from trust import egress

ANTHROPIC_KEY = "sk-ant-" + "a" * 24
BEARER = "Bearer " + "b" * 30


def _runtime():
    from config import get_config

    return get_config()._data.setdefault("runtime", {})


def _forge(monkeypatch, transport="http"):
    """A configured-but-down server: the egress/redaction boundary runs before the (stubbed)
    reconnect, so call_tool exercises the boundary without any network."""
    from tools import mcp_client as mc

    spec = mc.ServerSpec(name="srv", transport=transport, url="http://mcp.example.com/api")
    st = mc._ServerState(spec=spec, state="error", error="down")
    monkeypatch.setitem(mc._SERVERS, "srv", st)
    monkeypatch.setattr(mc, "_launch", lambda s: None)
    monkeypatch.setattr(mc, "_await_ready", lambda states, timeout: None)
    return mc


def test_redact_args_rewrites_string_leaves():
    from tools.mcp_client import _redact_args

    args = {
        "text": f"use {ANTHROPIC_KEY} please",
        "nested": {"headers": [BEARER]},
        "count": 7,
    }
    new, total = _redact_args(args)
    assert total == 2
    assert "[REDACTED:anthropic-key]" in new["text"]
    assert "[REDACTED:bearer-token]" in new["nested"]["headers"][0]
    assert new["count"] == 7
    assert ANTHROPIC_KEY in args["text"]  # the original tree is never mutated


def test_warn_mode_counts_redactions_into_egress(monkeypatch, isolated_paths):
    mc = _forge(monkeypatch)
    monkeypatch.setitem(_runtime(), "redaction", "warn")
    mark = egress.next_seq()
    out = mc.call_tool("srv", "post", {"text": ANTHROPIC_KEY})
    assert out.startswith("Error")  # never connected — the boundary already did its job
    evs = egress.events_since(mark)
    assert evs and evs[0].channel == "mcp"
    assert evs[0].redactions == 1


def test_off_mode_does_not_scan(monkeypatch, isolated_paths):
    mc = _forge(monkeypatch)
    monkeypatch.setitem(_runtime(), "redaction", "off")
    mark = egress.next_seq()
    mc.call_tool("srv", "post", {"text": ANTHROPIC_KEY})
    evs = egress.events_since(mark)
    assert evs and evs[0].redactions == 0


def test_stdio_server_not_gated_or_scanned(monkeypatch, isolated_paths):
    mc = _forge(monkeypatch, transport="stdio")
    monkeypatch.setitem(_runtime(), "redaction", "warn")
    mark = egress.next_seq()
    out = mc.call_tool("srv", "post", {"text": ANTHROPIC_KEY})
    assert out.startswith("Error")
    assert egress.events_since(mark) == []  # a local child process is not network egress


def test_map_strings_visits_exactly_what_iter_strings_yields():
    # The rewrite walker (_redact_args' map_strings) and the scan walker (scan_args'
    # iter_strings) must agree about what counts as argument content — warn-mode counts and
    # redact-mode rewrites read the same leaves (dict keys and non-string scalars skipped).
    from textutil import iter_strings, map_strings

    tree = {"a": "s1", "b": [1, "s2", ("s3", None)], "c": {"k": "s4"}, "n": 7}
    seen = []

    def swap(s):
        seen.append(s)
        return s.upper()

    out = map_strings(tree, swap)
    assert seen == list(iter_strings(tree))
    assert out == {"a": "S1", "b": [1, "S2", ["S3", None]], "c": {"k": "S4"}, "n": 7}
