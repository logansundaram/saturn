"""
The ambient-trust wave — the trust stack surfacing in the DEFAULT flow, no command required:

  - the session-start posture line (receipt.posture_spans + ui.posture_line),
  - per-call egress attribution riding tool_events (nodes/tools._egress_slice + tool_node) and
    its rail leaf (trace._egress_leaf),
  - the gate-decision echo (trace._render_trust_annotations),
  - the native Sources footer split (response._split_sources).

(The taint-warning render and the status bar's session token spend left with the audit-crypto
shelve / 2026-07-03 runtime trim; their tests went with them.)

All pure/offline; tool "calls" are fakes that record egress without touching the network.
NOTE: the tui.ui package re-exports flat, so submodules are reached via importlib.
"""

import importlib

import pytest

from trust import egress
from trust import receipt


# --- the session-start posture line --------------------------------------------------------------

def _runtime(monkeypatch) -> dict:
    from config import get_config

    return get_config()._data.setdefault("runtime", {})


def test_posture_spans_default_posture_is_silent(monkeypatch):
    # Deviation-only (2026-07-06): the safe default posture (gate read_only · local inference ·
    # quarantine gate · no airgap) renders NO spans — a stock install prints no posture line at
    # all. Silence means the defaults hold; /privacy carries the affirmative readout.
    rt = _runtime(monkeypatch)
    monkeypatch.setitem(rt, "auto_approve", "read_only")
    monkeypatch.setitem(rt, "airgap", False)
    monkeypatch.setitem(rt, "quarantine", "gate")

    monkeypatch.setattr(egress, "_inference", lambda: {"all_local": True})

    assert receipt.posture_spans() == []


def test_posture_spans_loud_states_lead_and_warn(monkeypatch):
    rt = _runtime(monkeypatch)
    monkeypatch.setitem(rt, "auto_approve", "destructive")  # the gate is OPEN, not "at a tier"
    monkeypatch.setitem(rt, "airgap", True)
    monkeypatch.setitem(rt, "quarantine", "off")

    monkeypatch.setattr(
        egress, "_inference",
        lambda: {"all_local": False, "remote_ollama": "http://10.0.0.5:11434"},
    )

    spans = receipt.posture_spans()
    assert spans[0] == ("⚠ GATE OFF", "risk")
    assert ("⛓ airgap", "accent") in spans
    assert ("inference off-machine: ollama @ http://10.0.0.5:11434", "warn") in spans
    assert ("quarantine off", "warn") in spans
    assert not any(text.startswith("redaction") for text, _ in spans)  # cut 2026-09-29


def test_posture_spans_state_the_effective_quarantine_mode(monkeypatch):
    """The posture line must state the mode IN FORCE (quarantine.mode() — invalid values run as
    'gate', case is normalized), never echo a raw config string the system ignored: 'quarantine
    none' rendered calm-dim over a system actually running 'gate' is a posture it didn't read."""
    rt = _runtime(monkeypatch)

    monkeypatch.setattr(egress, "_inference", lambda: {"all_local": True})

    monkeypatch.setitem(rt, "quarantine", "none")  # invalid → the system runs gated (= default,
    spans = receipt.posture_spans()                # so deviation-only says nothing at all)
    assert not any(t.startswith("quarantine") for t, _ in spans)

    monkeypatch.setitem(rt, "quarantine", "OFF")   # case variant → effective off, styled loud
    spans = receipt.posture_spans()
    assert ("quarantine off", "warn") in spans


def test_posture_line_prints_deviations_with_pointer(capsys, monkeypatch):
    mod = importlib.import_module("tui.ui.prompt")
    monkeypatch.setattr(receipt, "posture_spans", lambda: [("⚠ GATE OFF", "risk")])
    capsys.readouterr()  # drain
    mod.posture_line()
    out = capsys.readouterr().out
    assert "GATE OFF" in out
    assert "/privacy" in out and "/policy" in out


def test_posture_line_silent_on_default_posture(capsys, monkeypatch):
    # Deviation-only: no spans → no line, no pointers — the default install's screen carries no
    # ambient privacy chrome at all.
    mod = importlib.import_module("tui.ui.prompt")
    monkeypatch.setattr(receipt, "posture_spans", lambda: [])
    capsys.readouterr()  # drain
    mod.posture_line()
    assert capsys.readouterr().out == ""


def test_posture_line_styles_cover_every_kind():
    mod = importlib.import_module("tui.ui.prompt")
    assert {"ok", "warn", "risk", "accent", "dim"} <= set(mod._POSTURE_LINE_STYLE)


def test_posture_line_swallows_a_broken_posture(capsys, monkeypatch):
    mod = importlib.import_module("tui.ui.prompt")
    monkeypatch.setattr(receipt, "posture_spans", lambda: 1 / 0)
    mod.posture_line()  # must not raise
    assert "/privacy" not in capsys.readouterr().out  # and must not print a guessed posture


# --- per-call egress attribution (nodes/tools) ---------------------------------------------------

def test_tool_node_attaches_the_calls_egress_slice(monkeypatch, isolated_paths):
    from langchain.messages import AIMessage

    import nodes.tools as tn

    class SendingTool:
        def invoke(self, args):
            egress.record("http", "api.example.com", "GET /", n_bytes=123)
            return "ok"

    monkeypatch.setitem(tn.tools_by_name, "http_request", SendingTool())
    msg = AIMessage(content="", tool_calls=[{"name": "http_request", "args": {}, "id": "c1"}])
    delta = tn.tool_node({"messages": [msg]})

    ev = delta["tool_events"][0]
    assert ev["egress"] == [{
        "channel": "http", "host": "api.example.com",
        "n_bytes": 123, "status": "sent",
    }]


def test_tool_node_attaches_blocked_events_and_silent_calls_get_none(monkeypatch, isolated_paths):
    from langchain.messages import AIMessage

    import nodes.tools as tn

    rt = _runtime(monkeypatch)
    monkeypatch.setitem(rt, "airgap", True)

    class BlockedTool:
        def invoke(self, args):
            refusal = egress.check("web_search", "duckduckgo.com", "q")
            return refusal or "sent"

    class SilentTool:
        def invoke(self, args):
            return "42"

    monkeypatch.setitem(tn.tools_by_name, "web_search", BlockedTool())
    monkeypatch.setitem(tn.tools_by_name, "calculate", SilentTool())
    msg = AIMessage(content="", tool_calls=[
        {"name": "web_search", "args": {}, "id": "c1"},
        {"name": "calculate", "args": {}, "id": "c2"},
    ])
    delta = tn.tool_node({"messages": [msg]})

    blocked, silent = delta["tool_events"]
    assert blocked["egress"][0]["status"] == "blocked"
    assert blocked["egress"][0]["host"] == "duckduckgo.com"
    assert "egress" not in silent  # a local-only call carries no boundary annotation


def test_egress_slice_caps_a_runaway_call(isolated_paths):
    import nodes.tools as tn

    mark = egress.next_seq()
    for i in range(6):
        egress.record("http", f"h{i}.example.com", "x")
    out = tn._egress_slice(mark)
    assert len(out) == tn._MAX_EGRESS_EVENTS + 1
    assert out[-1] == {"more": 6 - tn._MAX_EGRESS_EVENTS}


# --- the rail leaves (trace) ---------------------------------------------------------------------

def test_egress_leaf_text_and_styles():
    tr = importlib.import_module("tui.ui.trace")

    text, style = tr._egress_leaf(
        {"channel": "http", "host": "api.example.com", "n_bytes": 123,
         "redactions": 1, "status": "sent"})  # an older record's field is ignored
    assert text.startswith("⇅ sent → api.example.com")
    assert "http" in text and "redaction" not in text
    assert style == "yellow"

    text, style = tr._egress_leaf(
        {"channel": "web_search", "host": "duckduckgo.com", "status": "blocked"})
    assert text.startswith("⊘ air-gap blocked web_search → duckduckgo.com")
    assert style == "bold red"

    text, style = tr._egress_leaf({"more": 2})
    assert "+2 more" in text and "/privacy egress" in text


def test_gate_decision_echo_renders_both_verdicts(capsys):
    tr = importlib.import_module("tui.ui.trace")

    tr._render_trust_annotations("approval", {"gate_events": [{
        "calls": [
            {"id": "1", "name": "write_file", "approved": True},
            {"id": "2", "name": "run_shell", "approved": False},
        ],
        "decision": "partial", "quarantine": True, "step": None,
    }]})
    out = capsys.readouterr().out
    assert "you approved write_file" in out
    assert "you rejected run_shell" in out
    assert "quarantine escalation" in out


_FOOTER_TEXT = ("The answer body cites [1].\n\n"
                "Sources:\n  [1] web_extract(url='https://e.com')\n  [2] knowledge base: a.md")


def test_split_sources_extracts_a_wellformed_footer():
    resp = importlib.import_module("tui.ui.response")

    prose, entries = resp._split_sources(_FOOTER_TEXT)
    assert prose == "The answer body cites [1]."
    assert entries == ["  [1] web_extract(url='https://e.com')", "  [2] knowledge base: a.md"]


@pytest.mark.parametrize("text", [
    "no footer at all",
    "prose\n\nSources:\n  - a bullet, not an [n] entry",
    "prose\n\nSources:",  # header with no entries
])
def test_split_sources_leaves_anything_else_alone(text):
    resp = importlib.import_module("tui.ui.response")

    assert resp._split_sources(text) == (text, None)


# --- the status bar's posture zone ---------------------------------------------------------------

def test_statusbar_unreadable_posture_is_unknown_never_calm(monkeypatch):
    # A config read failing mid-refresh must NOT render the calm `read_only` tier — that would
    # show a SAFER posture than reality on exactly the surface that exists to shout ⚠ GATE OFF
    # while the gate is open. The facet renders an explicit unknown instead (the posture-line
    # rule: a facet that can't be read is omitted/marked, never guessed).
    sb = importlib.import_module("tui.ui.statusbar")
    if not sb._RICH:
        pytest.skip("rich not available")
    import config as config_mod

    def boom():
        raise RuntimeError("config unreadable mid-refresh")

    monkeypatch.setattr(config_mod, "get_config", boom)
    plain = sb._StatusBar().__rich__().plain
    assert "read_only" not in plain
    assert "posture ?" in plain
