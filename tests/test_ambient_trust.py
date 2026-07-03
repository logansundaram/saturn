"""
The ambient-trust wave — the trust stack surfacing in the DEFAULT flow, no command required:

  - the session-start posture line (receipt.posture_spans + ui.posture_line),
  - per-call egress attribution riding tool_events (nodes/tools._egress_slice + tool_node) and
    its rail leaf (trace._egress_leaf),
  - the gate-decision echo + judge-verdict leaf (trace._render_trust_annotations),
  - native answer provenance (response._split_sources/_print_sources/_print_taint_warning over
    a live Glass Box), and the centralized live-slice guard (glassbox.build_live),
  - the status bar's session token spend.

All pure/offline; tool "calls" are fakes that record egress without touching the network.
NOTE: the tui.ui package re-exports flat, so submodules are reached via importlib.
"""

import importlib

import pytest

from trust import egress
from trust import glassbox
from trust import receipt


# --- the session-start posture line --------------------------------------------------------------

def _runtime(monkeypatch) -> dict:
    from config import get_config

    return get_config()._data.setdefault("runtime", {})


def test_posture_spans_default_posture_is_calm(monkeypatch):
    rt = _runtime(monkeypatch)
    monkeypatch.setitem(rt, "auto_approve", "read_only")
    monkeypatch.setitem(rt, "airgap", False)
    monkeypatch.setitem(rt, "quarantine", "gate")

    monkeypatch.setattr(egress, "_inference", lambda: {"all_local": True, "cloud_providers": []})

    spans = receipt.posture_spans()
    texts = [t for t, _ in spans]
    kinds = [k for _, k in spans]
    assert ("gate read_only", "ok") == spans[0]
    assert ("inference local", "ok") in spans
    assert ("quarantine gate", "dim") in spans
    assert "risk" not in kinds  # nothing loud on the safe default posture
    assert not any(t.startswith("⛓") for t in texts)


def test_posture_spans_loud_states_lead_and_warn(monkeypatch):
    rt = _runtime(monkeypatch)
    monkeypatch.setitem(rt, "auto_approve", "destructive")  # the gate is OPEN, not "at a tier"
    monkeypatch.setitem(rt, "airgap", True)
    monkeypatch.setitem(rt, "quarantine", "off")
    monkeypatch.setitem(rt, "redaction", "off")

    monkeypatch.setattr(
        egress, "_inference", lambda: {"all_local": False, "cloud_providers": ["anthropic"]}
    )

    spans = receipt.posture_spans()
    assert spans[0] == ("⚠ GATE OFF", "risk")
    assert ("⛓ airgap", "accent") in spans
    assert ("inference cloud: anthropic", "warn") in spans
    assert ("quarantine off", "warn") in spans
    # redaction off only matters when a cloud boundary exists to redact for — here it does
    assert ("redaction off", "warn") in spans


def test_posture_spans_state_the_effective_quarantine_mode(monkeypatch):
    """The posture line must state the mode IN FORCE (quarantine.mode() — invalid values run as
    'gate', case is normalized), never echo a raw config string the system ignored: 'quarantine
    none' rendered calm-dim over a system actually running 'gate' is a posture it didn't read."""
    rt = _runtime(monkeypatch)

    monkeypatch.setattr(egress, "_inference", lambda: {"all_local": True, "cloud_providers": []})

    monkeypatch.setitem(rt, "quarantine", "none")  # invalid → the system runs gated
    spans = receipt.posture_spans()
    assert ("quarantine gate", "dim") in spans
    assert not any(t.startswith("quarantine none") for t, _ in spans)

    monkeypatch.setitem(rt, "quarantine", "OFF")   # case variant → effective off, styled loud
    spans = receipt.posture_spans()
    assert ("quarantine off", "warn") in spans


def test_posture_line_prints_spans_and_pointer(capsys, monkeypatch):
    mod = importlib.import_module("tui.ui.prompt")
    out_before = capsys.readouterr()  # drain
    mod.posture_line()
    out = capsys.readouterr().out
    assert "gate" in out
    assert "/privacy" in out and "/policy" in out


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
        "n_bytes": 123, "redactions": 0, "status": "sent",
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
         "redactions": 1, "status": "sent"})
    assert text.startswith("⇅ sent → api.example.com")
    assert "http" in text and "1 redaction" in text
    assert style == "yellow"

    text, style = tr._egress_leaf(
        {"channel": "web_search", "host": "duckduckgo.com", "status": "blocked"})
    assert text.startswith("⛔ air-gap blocked web_search → duckduckgo.com")
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


def test_replan_verdict_leaf_is_honest_both_ways(capsys):
    tr = importlib.import_module("tui.ui.trace")

    tr._render_trust_annotations("replan", {"replans": 1})
    assert "ungrounded" in capsys.readouterr().out
    tr._render_trust_annotations("replan", {})
    assert "accepted" in capsys.readouterr().out
    tr._render_trust_annotations("agent", {})  # other nodes say nothing
    assert capsys.readouterr().out == ""


# --- native answer provenance (response) ---------------------------------------------------------

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


def _provenance_box():
    """A live Glass Box with one network source ([1] web_extract) and one local trusted source
    ([2] read_file) — built through the real assembler, no synthetic dict."""
    from langchain.messages import AIMessage, HumanMessage

    state = {
        "current_query": "q",
        "messages": [HumanMessage(content="q"), AIMessage(content="Answer [1][2].")],
        "tool_results": [
            "web_extract(url='u') -> some network page body of reasonable length here",
            "read_file(path='x') -> a local trusted file body of reasonable length here",
        ],
        "documents_retrieved": [],
        "tool_events": [{"name": "web_extract"}, {"name": "read_file"}],
        "replans": 0,
    }
    return glassbox.build_from_state(state, egress_events=None, gated=0)


def test_facet_annotation_vocabulary():
    resp = importlib.import_module("tui.ui.response")
    gb = _provenance_box()

    glyph, style, note = resp._facet_annotation(gb.sources[0])  # web_extract: network/untrusted
    assert (glyph, style) == ("◐", "yellow") and "web" in note
    glyph, style, note = resp._facet_annotation(gb.sources[1])  # read_file: local + trusted
    assert (glyph, style, note) == ("✓", "green", "local")


def test_print_sources_colors_by_facet_and_dims_without_provenance(capsys):
    resp = importlib.import_module("tui.ui.response")
    gb = _provenance_box()

    resp._print_sources(["  [1] web_extract(url='u')", "  [2] read_file(path='x')"], gb)
    out = capsys.readouterr().out
    assert "Sources:" in out
    assert "◐ web" in out
    assert "✓ local" in out

    resp._print_sources(["  [1] web_extract(url='u')"], None)  # no provenance: text only
    out = capsys.readouterr().out
    assert "[1] web_extract(url='u')" in out
    assert "◐" not in out and "✓" not in out


def test_set_turn_provenance_pops_on_read(isolated_paths):
    resp = importlib.import_module("tui.ui.response")

    state = {
        "current_query": "q",
        "messages": [],
        "tool_results": ["web_search(query='x') -> a result"],
        "documents_retrieved": [],
        "tool_events": [{"name": "web_search", "args": {}, "result": "r", "dur": 0.1, "ok": True}],
        "replans": 0,
    }
    resp.set_turn_provenance(state)
    gb = resp._pop_turn_provenance()
    assert gb is not None and gb.sources[0].tool == "web_search"
    assert resp._pop_turn_provenance() is None  # consumed — can never paint a later answer


# --- the centralized live-slice guard (glassbox.build_live) --------------------------------------

_EMPTY_STATE = {"current_query": "q", "messages": [], "tool_results": [],
                "documents_retrieved": [], "tool_events": [], "replans": 0}


def test_build_live_without_a_turn_mark_is_unknown(monkeypatch, isolated_paths):
    monkeypatch.setattr(receipt, "_TURN_MARK", 0)
    gb = glassbox.build_live(_EMPTY_STATE)
    assert gb.sent_known is False  # never 'local-only' over a slice that may be missing sends


def test_build_live_with_a_mark_uses_the_exact_slice(monkeypatch, isolated_paths):
    monkeypatch.setattr(receipt, "_TURN_MARK", receipt._TURN_MARK)
    receipt.reset_turn()
    egress.record("llm", "anthropic API", "model", n_bytes=10)
    gb = glassbox.build_live(_EMPTY_STATE)
    assert gb.sent_known is True
    assert gb.composed_local is False  # an llm-channel event in the slice


def test_build_live_treats_a_cleared_slice_as_unknown(monkeypatch, isolated_paths):
    monkeypatch.setattr(receipt, "_TURN_MARK", receipt._TURN_MARK)
    monkeypatch.setattr(egress, "_CLEARED_AT", egress._CLEARED_AT)
    receipt.reset_turn()
    egress.record("http", "api.example.com", "x")
    egress.clear()
    gb = glassbox.build_live(_EMPTY_STATE)
    assert gb.sent_known is False


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
